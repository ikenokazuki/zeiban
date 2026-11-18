#!/usr/bin/env python3
"""zeiban の Python の実装。Rust の実行ファイル（rs/）と同じ入力・同じ応答を返す。

一式（zeiban.bin・zeiban.json）から、語の一致の索引・節のベクトル・木・税率・確度の係数を読む。
3 層のエンコーダは、transformers で読む（リリースの zeiban-encoder。ふつうの 3 層の ModernBERT。config.json・model.safetensors・tokenizer.json）。
エンコーダの重みは、一式と同じ値（int8 に丸めた値を fp32 で持ったもの）なので、答えは Rust の実行ファイルと同じになる。

使い方:
  echo '{"name": "Mens crew neck t-shirt"}' | python3 py/zeiban.py bundle zeiban-encoder

コードから:
  from zeiban import Zeiban
  z = Zeiban('bundle', 'zeiban-encoder')
  print(z.classify({'name': 'Mens crew neck t-shirt', 'material': '100% cotton, knitted'})['hsCode'])

要るもの: numpy・torch・transformers（tokenizers は transformers と一緒に入る）。
"""
import argparse
import json
import math
import sys
import unicodedata
from pathlib import Path

import numpy as np

D, NONE, LEVELS = 256, 0xFFFF, 5
QN = 30                                                # 質問を作る候補の葉の数（応答に出す上位 k 件とは別。rs/src/main.rs の QN と同じ）


def build(item):
    """入力の 1 件 → (モデルに渡す文, 語の一致に使う文)。text があればそのまま。無ければ name・material・categories・description から組み立てる。"""
    if isinstance(item.get('text'), str):
        return item['text'], item['lex'] if isinstance(item.get('lex'), str) else item['text']
    field = lambda k: (item[k].strip() or None) if isinstance(item.get(k), str) else None
    cats = item.get('categories')
    cats = (' '.join(x for x in cats if isinstance(x, str)) or None) if isinstance(cats, list) else field('categories')
    name = field('name')
    if not name:
        raise ValueError('text か name が要る')
    text, lex = f'title: {name}', name
    for k, x in (('material', field('material')), ('category', cats), ('description', field('description'))):
        if x:
            text += f' | {k}: {x}'
            lex += ' ' + x
    return text, lex


def lex_norm(text):
    """語の一致に使う文の前処理: NFKC → 小文字 → 空白を 1 つにそろえ、先頭に空白を足す。"""
    return ' ' + ' '.join(unicodedata.normalize('NFKC', text).lower().split())


def is_ja(lex, share):
    """切り替えの決まり: 空白でない字のうち、かな・漢字（ひらがな・カタカナ・半角カタカナ・漢字）が share 以上か。"""
    body = [c for c in lex if c not in ' \t\n\r\x0b\x0c\u3000']
    ja = sum('\u3040' <= c <= '\u30ff' or '\uff66' <= c <= '\uff9f' or '\u3400' <= c <= '\u4dbf' or '\u4e00' <= c <= '\u9fff' or '\uf900' <= c <= '\ufaff' for c in body)
    return len(body) > 0 and ja >= share * len(body)


def unpack4(b):
    """int4 の区画（1 バイトに 2 個。値 −7〜7 に 8 を足したもの。下の 4 ビットが先）[行, n ÷ 2] → int8 [行, n]"""
    out = np.empty((b.shape[0], b.shape[1] * 2), np.int8)
    out[:, 0::2], out[:, 1::2] = (b & 15).astype(np.int8) - 8, (b >> 4).astype(np.int8) - 8
    return out


def unpack4_lin(b):
    """線形の層の int4 の区画 [出力 ÷ 16, 入力, 8] → int8 [出力, 入力]"""
    n, i, _ = b.shape
    return np.ascontiguousarray(unpack4(np.asarray(b).reshape(-1, 8)).reshape(n, i, 16).transpose(0, 2, 1)).reshape(n * 16, i)


def g6_state(sec, layers):
    """先頭 6 層の読み手（目録の enc.kind が g6）: 一式の int4 の重みを読んで、トークンの番号 → 状態 [768] を返す関数を作る。
    計算: 語の表 × √512 → 層ごとの入力（射影 × 512^-0.5 → 512 ごとに RMSNorm）→ 層（RMSNorm → 注意 → RMSNorm → 足す → RMSNorm → 門つきの MLP → RMSNorm → 足す
    → 層ごとの入力の足し込み → 層の倍率）→ RMSNorm → 射影（512 → 768）→ トークンの平均。注意は、問い 4 頭・問いと鍵に RMSNorm と回転の位置づけ・点の倍率 1。
    最後の層だけ「頭の幅 512・鍵と値は 1 頭・θ = 1,000,000」、ほかは「頭の幅 256・鍵と値は 2 頭・θ = 10,000」。"""
    import torch
    import torch.nn.functional as F
    H, HEADS = 512, 4
    lin = lambda k: torch.from_numpy(unpack4_lin(sec(k)).astype(np.float32) * sec(k + '.s')[:, None])
    vec = lambda k: torch.from_numpy(np.array(sec(k), np.float32))
    emb, emb_s = sec('g.emb'), sec('g.emb.s')
    W = {'ple': lin('g.ple'), 'ple.norm': vec('g.ple.norm'), 'norm': vec('g.norm'), 'eproj': lin('g.eproj')}
    for i in range(layers):
        W.update({f'{i}.{k}': lin(f'g.l{i}.{k}') for k in ('q', 'k', 'v', 'o', 'gate', 'up', 'down', 'pg', 'pp')})
        W.update({f'{i}.{k}': vec(f'g.l{i}.{k}') for k in ('n_in', 'n_q', 'n_k', 'n_att', 'n_pre', 'n_ff', 'n_ple', 'scalar')})
    rms = lambda x, w=None: (lambda y: y if w is None else y * w)(x * torch.pow(x.pow(2).mean(-1, keepdim=True) + 1e-6, -0.5))
    gelu = lambda x: F.gelu(x, approximate='tanh')

    def rot(x, n, dim, theta):
        f = torch.arange(n, dtype=torch.float32)[:, None] * (1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)))
        e = torch.cat((f, f), -1)
        return x * e.cos() + torch.cat((-x[..., dim // 2:], x[..., :dim // 2]), -1) * e.sin()

    def state(ids):
        with torch.no_grad():
            n = len(ids)
            x = torch.from_numpy(unpack4(emb[ids]).astype(np.float32) * emb_s[ids][:, None]) * (H ** 0.5)
            ple = rms(((x @ W['ple'].T) * H ** -0.5).view(n, layers, H), W['ple.norm'])
            for i in range(layers):
                hd, kv, theta = (512, 1, 1e6) if i == layers - 1 else (256, 2, 1e4)
                h = rms(x, W[f'{i}.n_in'])
                q = rot(rms((h @ W[f'{i}.q'].T).view(n, HEADS, hd).transpose(0, 1), W[f'{i}.n_q']), n, hd, theta)
                k = rot(rms((h @ W[f'{i}.k'].T).view(n, kv, hd).transpose(0, 1), W[f'{i}.n_k']), n, hd, theta).repeat_interleave(HEADS // kv, 0)
                v = rms((h @ W[f'{i}.v'].T).view(n, kv, hd).transpose(0, 1)).repeat_interleave(HEADS // kv, 0)
                a = (torch.softmax(q @ k.transpose(1, 2), -1) @ v).transpose(0, 1).reshape(n, HEADS * hd) @ W[f'{i}.o'].T
                x = x + rms(a, W[f'{i}.n_att'])
                h = rms(x, W[f'{i}.n_pre'])
                x = x + rms((gelu(h @ W[f'{i}.gate'].T) * (h @ W[f'{i}.up'].T)) @ W[f'{i}.down'].T, W[f'{i}.n_ff'])
                g = (gelu(x @ W[f'{i}.pg'].T) * ple[:, i]) @ W[f'{i}.pp'].T
                x = (x + rms(g, W[f'{i}.n_ple'])) * W[f'{i}.scalar']
            return (rms(x, W['norm']) @ W['eproj'].T).mean(0).numpy().astype(np.float64)
    return state


class Zeiban:
    def __init__(self, bundle, encoder=None):
        """bundle: zeiban.bin と zeiban.json のあるフォルダ。encoder: transformers で読むエンコーダのフォルダ
        （日本語の入力を読む 2 つ目の層のある一式では、その中の ja/ に 2 つ目のエンコーダが要る）。"""
        import torch
        from tokenizers import Tokenizer
        bundle = Path(bundle)
        man = self.man = json.loads((bundle / 'zeiban.json').read_text(encoding='utf-8'))
        raw = np.memmap(bundle / man['file'], mode='r')
        if raw.size != man['bytes']:
            raise ValueError('一式の大きさが目録と合わない')

        def sec(name):
            s = man['sections'][name]
            return np.frombuffer(raw, s['dtype'], int(np.prod(s['shape'])), s['offset']).reshape(s['shape'])

        self.n_nodes = man['n_nodes']
        self.offs = np.concatenate([[0], np.cumsum(self.n_nodes)])
        g6 = man.get('enc') if (man.get('enc') or {}).get('kind') == 'g6' else None   # 先頭 6 層の読み手の一式（読み手は 1 つ。エンコーダのフォルダは要らない）
        if not g6 and encoder is None:
            raise ValueError('この一式には、エンコーダのフォルダが要る')
        encoder = None if g6 else Path(encoder)
        # 節のベクトル: 番号のまま fp32 で持つ（約 39 MB）。node_s は、番号に掛けて長さ 1 のベクトルとの内積にする数
        self.node, self.node_s = (unpack4(sec('node4')).astype(np.float32), sec('node4.n')) if g6 else (sec('node').astype(np.float32), sec('node.s'))
        torch.set_num_threads(2)                                                      # 小さいモデルなので、スレッドを増やすと待ち合わせでかえって遅くなる

        def reader(pre, folder, calib, calib_prefix):
            """文を読む層ひとつ分: 問いの射影・目盛り・確度の係数と、エンコーダ（先頭トークンの状態を返す関数）。語の埋め込みと節のベクトルは、どの層でも同じ。"""
            from transformers import AutoModel
            w = sec(pre + 'q_proj')                                                   # [16 出力ごとのブロック, 入力, 16] → [出力, 入力]
            model = AutoModel.from_pretrained(folder).eval()

            def state(ids):
                """トークンの番号 → 先頭トークンの状態（最後の正規化のあと）[256]"""
                with torch.no_grad():
                    return model(input_ids=torch.tensor([ids])).last_hidden_state[0, 0].numpy().astype(np.float64)
            return {'q_w': w.transpose(0, 2, 1).reshape(-1, D).astype(np.float32) * sec(pre + 'q_proj.s')[:, None], 'q_b': sec(pre + 'q_proj.b'), 'scale': sec(pre + 'scale'), 'lam': sec(pre + 'lam'),
                    'state': state, 'calib': calib, 'calibPrefix': calib_prefix}

        if g6:
            self.base = {'q_w': unpack4_lin(sec('g.q_proj')).astype(np.float32) * sec('g.q_proj.s')[:, None], 'q_b': sec('g.q_proj.b'), 'scale': sec('g.scale'), 'lam': sec('g.lam'),
                         'state': g6_state(sec, g6['layers']), 'calib': man.get('calib'), 'calibPrefix': man.get('calibPrefix')}
            self.gtk, self.gid = Tokenizer.from_file(str(bundle / g6['tokenizer'])), sec('g.tok.id').astype(np.int64)      # 読み手の語の区切り（絞った語彙。番号はもとのまま）と、もとの番号 → 語の表の行
            self.gtk.no_padding()
            self.gtk.enable_truncation(max_length=man['q_len'])
        else:
            self.base = reader('', encoder, man.get('calib'), man.get('calibPrefix'))
        self.g6 = g6
        self.ja, self.ja_share = None, None                                           # 日本語の入力を読む 2 つ目の層（目録の route.ja。無い一式では、全部の入力をもとの層で読む）
        route = (man.get('route') or {}).get('ja')
        if route:
            if not (encoder / 'ja' / 'config.json').exists():
                raise ValueError(f'この一式には日本語用の 2 つ目の層がある。エンコーダのフォルダに ja/ が要る: {encoder / "ja"}')
            self.ja, self.ja_share = reader(route['prefix'], encoder / 'ja', route.get('calib'), route.get('calibPrefix')), route['share']
        self.leaf_path = sec('leaf_path').astype(np.int64)
        self.codes = np.array([bytes(r).decode('ascii') for r in sec('leaves')])
        self.labels = bytes(sec('label')).decode('utf-8').split('\n')
        self.duty = [dict(zip(('general', 'special', 'other'), (line.split('\t') + ['', '', ''])[:3])) for line in bytes(sec('duty')).decode('utf-8').split('\n')]
        self.leaf_duty = sec('leaf_duty')
        self.term = sec('tok.term')
        self.ptr, self.doc, self.dlen = sec('bmx.ptr').astype(np.int64), sec('bmx.doc').astype(np.int64), sec('bmx.len').astype(np.float64)
        self.tf = sec('bmx.tf').astype(np.float64)                                    # 回数。0 は bmx.big（位置, 回数）にある
        big = sec('bmx.big')
        self.tf[big[:, 0]] = big[:, 1]
        self.idf, self.ent = sec('bmx.idf'), sec('bmx.ent')
        self.avgdl, self.alpha, self.beta, self.norm1 = (float(x) for x in sec('bmx.const'))
        self.tk = Tokenizer.from_file(str(bundle / g6['bmx_tokenizer'] if g6 else encoder / 'tokenizer.json'))      # 語の一致（BMX）の語の区切り
        self.tk.no_padding()
        self.tk.no_truncation()

    def bmx_top(self, lex):
        """語の一致の点（BMX）の上位の葉 → (葉の番号, 点)。点の高い順、同じ点は番号の小さい順。"""
        terms = list(dict.fromkeys(int(t) for t in self.term[self.tk.encode(lex_norm(lex), add_special_tokens=False).ids] if t != NONE))
        if not terms:
            return np.zeros(0, np.int64), np.zeros(0, np.float32)
        ent = self.ent[terms]
        e = ent / ent.max() if ent.max() > 0 else np.zeros(len(terms))
        m, ebar, a = len(terms), e.mean(), self.alpha
        acc = np.zeros((3, self.n_nodes[4]))
        for w, ew in zip(terms, e):
            d, f = self.doc[self.ptr[w]:self.ptr[w + 1]], self.tf[self.ptr[w]:self.ptr[w + 1]]
            acc[0, d] += self.idf[w] * f * (a + 1) / (f + a * self.dlen[d] / self.avgdl + a * ebar)
            acc[1, d] += 1
            acc[2, d] += ew
        hit = np.flatnonzero(acc[1])
        s = acc[0, hit] + self.beta * acc[2, hit] * acc[1, hit] / m
        o = np.lexsort((hit, -s))[:self.man['k']]
        return hit[o], (s[o] / (m * self.norm1)).astype(np.float32)

    def predict(self, text, lex, k=5, prefix='', full=False):
        """→ 上位 k 件の (葉の番号, 葉の点, 読んだ層)。full なら、4 つ目に (葉の点の全部, prefix の中の葉の番号) を足す（質問の候補を、枝の中から取り直すのに使う）。葉の点 = 5 段の log softmax(scale × cos + λ × BMX) を、道筋に沿って足したもの。
        日本語の入力（is_ja）は、2 つ目の層のある一式では、その層で読む。"""
        r = self.ja if self.ja is not None and is_ja(lex, self.ja_share) else self.base
        if self.g6:                                                                   # 文の頭に決まった文を付け、前後に印を付けて、q_len までに切る。もとの番号 → 語の表の行
            raw = np.asarray(self.gtk.encode(self.g6['text_prefix'] + text).ids, np.int64)
            ids = np.searchsorted(self.gid, raw)
            assert (self.gid[ids] == raw).all()
        else:
            ids = [self.man['tok']['bos']] + self.tk.encode(text, add_special_tokens=False).ids[:self.man['q_len'] - 2] + [self.man['tok']['eos']]
        q = (r['q_w'].astype(np.float64) @ r['state'](ids) + r['q_b']).reshape(LEVELS, D)
        q = (q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)).astype(np.float32)
        leaf, val = self.bmx_top(lex)
        score = np.zeros(self.n_nodes[4], np.float32)
        for t in range(LEVELS):
            at, n = self.offs[t], self.n_nodes[t]
            bm = np.zeros(n, np.float32)
            np.maximum.at(bm, self.leaf_path[leaf, t], val)                           # 節の BMX の点は、その下の葉の点の最大
            s = (r['scale'][t] * ((self.node[at:at + n] @ q[t]) * self.node_s[at:at + n]) + r['lam'][t] * bm).astype(np.float64)
            score += (s - (s.max() + math.log(np.exp(s - s.max()).sum()))).astype(np.float32)[self.leaf_path[:, t]]
        idx = np.flatnonzero(np.char.startswith(self.codes, prefix)) if prefix else np.arange(self.n_nodes[4])
        top = idx[np.lexsort((idx, -score[idx]))[:k]]
        return (top, score[top], r, (score, idx)) if full else (top, score[top], r)

    def respond(self, top, score, fixed=0, reader=None, wide=None, full=None, confirm=False):
        """上位の候補 → 応答。fixed は、呼び手がもう答えた段の数（prefix の桁 ÷ 2）。reader は predict の返す「読んだ層」（確度の係数を持つ。無ければ、もとの層）。
        wide は、質問を作る広い候補の (葉の番号, 葉の点)（top を頭に含む、点の高い順の QN 葉。無ければ top と同じ）。full は predict の返す (葉の点の全部, prefix の中の葉の番号)。"""
        reader = reader or self.base
        man, lp = self.man, self.leaf_path
        label = lambda leaf, t: self.labels[self.offs[t] + lp[leaf, t]]
        desc = lambda leaf: ' > '.join(x for x in (label(leaf, t) for t in range(LEVELS)) if x)
        duty = lambda leaf: self.duty[self.leaf_duty[leaf]]
        code = lambda leaf: str(self.codes[leaf])
        out = {'codes': [code(i) for i in top], 'scores': [float(x) for x in score], 'htsRelease': man['htsRelease']}
        if not len(top):
            return out
        first, best = int(top[0]), float(score[0])
        p = np.exp((score.astype(np.float64) - best) / man['temp'])
        p /= p.sum()
        conf = None
        if reader['calib']:                                                           # 段 t の確度 = σ(a + b·logit(m) + c·1 位の点 + d·答えた段の数)。m は、候補の確率のうち 1 位と同じ節の分
            c, conf = reader['calibPrefix' if fixed else 'calib'], [0.0] * LEVELS
            for t in reversed(range(LEVELS)):
                m = min(max(float(p[lp[top, t] == lp[first, t]].sum()), 1e-3), 1 - 1e-3)
                x = 1 / (1 + math.exp(-(c[t][0] + c[t][1] * (math.log(m) - math.log1p(-m)) + c[t][2] * best + c[t][3] * fixed)))
                conf[t] = x if t == LEVELS - 1 else max(x, conf[t + 1])               # 上の桁の確度は、下の桁より低くしない
            conf[:fixed] = [1.0] * fixed
        out.update({'hsCode': {'code': code(first), 'description': desc(first)}, 'sixDigit': code(first)[:6], 'confidenceScore': conf[4] if conf else float(p[0])})
        if conf:
            out['sixDigitConfidence'] = conf[2]
        out['dutyRate'] = duty(first)
        out['alternates'] = [{'code': code(i), 'description': desc(i), 'probabilityMass': float(q), 'dutyRate': duty(i)} for i, q in zip(top[1:], p[1:])]
        out['questions'] = []
        # 質問: 確度が tau より低い最初の段 t0 を聞く（自信のある段は飛ばす）。候補は、1 位と t0 の上まで同じ葉のうち、点の高い QN 葉。その候補の頭の桁が分かれる最初の段が質問になる。
        # どの段も確度が tau 以上のとき（と、合わせ込みの無い一式）は、広い候補（上位 QN 葉）が分かれる最初の段。確率は、候補の中での割合
        wtop, wscore = wide if wide is not None else (top, score)
        split = lambda pool, start: next((t for t in range(start, LEVELS) if (lp[pool, t] != lp[first, t]).any()), None)
        t, t0 = None, next((t for t in range(fixed, LEVELS) if conf[t] < man['tau']), None) if conf and not confirm else None      # confirm: 自信のある段も飛ばさず、上の段から順に聞く
        if t0 is not None:
            wt, ws = np.asarray(wtop), np.asarray(wscore)
            if t0 and full is not None:                                               # 1 位と t0 の上まで同じ葉から、点の高い QN 葉を取り直す
                ix = full[1][lp[full[1], t0 - 1] == lp[first, t0 - 1]]
                pool = ix[np.lexsort((ix, -full[0][ix]))[:QN]]
                psc = full[0][pool]
            else:                                                                     # t0 が一番上の段か、葉の点の全部が無い: 広い候補のうち、枝の中のもの
                keep = lp[wt, t0 - 1] == lp[first, t0 - 1] if t0 else np.ones(len(wt), bool)
                pool, psc = wt[keep], ws[keep]
            t = split(pool, t0)
            if t is not None:
                wtop, wscore = pool, psc
        if t is None:
            t = split(wtop, 0)
        pw = np.exp((np.asarray(wscore, np.float64) - best) / man['temp'])
        pw /= pw.sum()
        if t is not None:
            opts = {}
            for i, q in zip(wtop, pw):
                o = opts.setdefault(int(lp[i, t]), [int(i), 0.0])
                o[1] += float(q)
            c = conf[t] if conf else next(iter(opts.values()))[1]
            out['questions'].append({'digits': 2 * t + 2, 'ask': c < man['tau'], 'confidence': c, 'options': [
                {'prefix': code(i)[:2 * t + 2], 'label': label(i, t), 'code': code(i), 'probabilityMass': q, 'dutyRate': duty(i)} for i, q in opts.values()]})
        if reader is self.ja:
            out['reader'] = 'ja'                                                      # 2 つ目の層（日本語）で読んだ印。もとの層で読んだ応答には付けない
        return out

    def classify(self, item, k=5):
        """入力の 1 件（name・material・categories・description・prefix）→ 応答。"""
        prefix = item.get('prefix') or ''
        if not (isinstance(prefix, str) and (prefix == '' or prefix.isascii() and prefix.isdigit() and len(prefix) in (2, 4, 6, 8, 10))):
            raise ValueError('prefix は 2・4・6・8・10 桁の数字')
        text, lex = build(item)
        top, score, reader, full = self.predict(text, lex, max(k, QN), prefix, full=True)   # 質問を作るために、応答に出す k 件より広く取る
        return self.respond(top[:k], score[:k], len(prefix) // 2, reader, (top, score), full, item.get('confirm') is True)


def demo():
    assert build({'name': ' cotton t-shirt ', 'material': 'cotton', 'categories': ['apparel', 'tops']}) == ('title: cotton t-shirt | material: cotton | category: apparel tops', 'cotton t-shirt cotton apparel tops')
    assert build({'text': 'title: a', 'lex': 'a'}) == ('title: a', 'a') and build({'text': 'x'}) == ('x', 'x')
    assert lex_norm('Ｃｏｔｔｏｎ\tＴシャツ  100％') == ' cotton tシャツ 100%' and lex_norm('') == ' '
    assert is_ja('綿のTシャツ', 0.1) and is_ja('調製食料品', 0.1) and not is_ja('Cotton T-shirt', 0.1) and not is_ja('LED lamp for car 新品 12V 6000K waterproof bulb', 0.1) and not is_ja(' \u3000', 0.1)
    print('demo ok')


def main():
    ap = argparse.ArgumentParser(description='zeiban: 商品の文から米国 HTS の 10 桁を出す（標準入力に 1 行 1 件の JSON）')
    ap.add_argument('bundle', help='zeiban.bin と zeiban.json のあるフォルダ。demo と書くと自己確認だけ')
    ap.add_argument('encoder', nargs='?', help='transformers で読むエンコーダのフォルダ（リリースの zeiban-encoder。先頭 6 層の読み手の一式では要らない）')
    ap.add_argument('-k', type=int, default=5, help='上位の件数')
    a = ap.parse_args()
    if a.bundle == 'demo':
        return demo()
    z = Zeiban(a.bundle, a.encoder)
    for line in sys.stdin:
        if line.strip():
            print(json.dumps(z.classify(json.loads(line), a.k), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
