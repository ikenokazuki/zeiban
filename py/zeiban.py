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


class Zeiban:
    def __init__(self, bundle, encoder):
        """bundle: zeiban.bin と zeiban.json のあるフォルダ。encoder: transformers で読むエンコーダのフォルダ
        （日本語の入力を読む 2 つ目の層のある一式では、その中の ja/ に 2 つ目のエンコーダが要る）。"""
        import torch
        from tokenizers import Tokenizer
        from transformers import AutoModel
        bundle, encoder = Path(bundle), Path(encoder)
        man = self.man = json.loads((bundle / 'zeiban.json').read_text(encoding='utf-8'))
        raw = np.memmap(bundle / man['file'], mode='r')
        if raw.size != man['bytes']:
            raise ValueError('一式の大きさが目録と合わない')

        def sec(name):
            s = man['sections'][name]
            return np.frombuffer(raw, s['dtype'], int(np.prod(s['shape'])), s['offset']).reshape(s['shape'])

        self.n_nodes = man['n_nodes']
        self.offs = np.concatenate([[0], np.cumsum(self.n_nodes)])
        self.node, self.node_s = sec('node').astype(np.float32), sec('node.s')        # int8 の値のまま fp32 で持つ（約 39 MB）
        torch.set_num_threads(2)                                                      # 小さいモデルなので、スレッドを増やすと待ち合わせでかえって遅くなる

        def reader(pre, folder, calib, calib_prefix):
            """文を読む層ひとつ分: 問いの射影・目盛り・確度の係数と、エンコーダ（先頭トークンの状態を返す関数）。語の埋め込みと節のベクトルは、どの層でも同じ。"""
            w = sec(pre + 'q_proj')                                                   # [16 出力ごとのブロック, 入力, 16] → [出力, 入力]
            model = AutoModel.from_pretrained(folder).eval()

            def state(ids):
                """トークンの番号 → 先頭トークンの状態（最後の正規化のあと）[256]"""
                with torch.no_grad():
                    return model(input_ids=torch.tensor([ids])).last_hidden_state[0, 0].numpy().astype(np.float64)
            return {'q_w': w.transpose(0, 2, 1).reshape(-1, D).astype(np.float32) * sec(pre + 'q_proj.s')[:, None], 'q_b': sec(pre + 'q_proj.b'), 'scale': sec(pre + 'scale'), 'lam': sec(pre + 'lam'),
                    'state': state, 'calib': calib, 'calibPrefix': calib_prefix}

        self.base = reader('', encoder, man.get('calib'), man.get('calibPrefix'))
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
        self.tk = Tokenizer.from_file(str(encoder / 'tokenizer.json'))
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

    def predict(self, text, lex, k=5, prefix=''):
        """→ 上位 k 件の (葉の番号, 葉の点, 読んだ層)。葉の点 = 5 段の log softmax(scale × cos + λ × BMX) を、道筋に沿って足したもの。
        日本語の入力（is_ja）は、2 つ目の層のある一式では、その層で読む。"""
        r = self.ja if self.ja is not None and is_ja(lex, self.ja_share) else self.base
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
        return top, score[top], r

    def respond(self, top, score, fixed=0, reader=None):
        """上位の候補 → 応答。fixed は、呼び手がもう答えた段の数（prefix の桁 ÷ 2）。reader は predict の返す「読んだ層」（確度の係数を持つ。無ければ、もとの層）。"""
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
        t = next((t for t in range(LEVELS) if (lp[top, t] != lp[first, t]).any()), None)     # 質問: 候補の頭の桁が分かれる最初の段
        if t is not None:
            opts = {}
            for i, q in zip(top, p):
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
        top, score, reader = self.predict(text, lex, k, prefix)
        return self.respond(top, score, len(prefix) // 2, reader)


def demo():
    assert build({'name': ' cotton t-shirt ', 'material': 'cotton', 'categories': ['apparel', 'tops']}) == ('title: cotton t-shirt | material: cotton | category: apparel tops', 'cotton t-shirt cotton apparel tops')
    assert build({'text': 'title: a', 'lex': 'a'}) == ('title: a', 'a') and build({'text': 'x'}) == ('x', 'x')
    assert lex_norm('Ｃｏｔｔｏｎ\tＴシャツ  100％') == ' cotton tシャツ 100%' and lex_norm('') == ' '
    assert is_ja('綿のTシャツ', 0.1) and is_ja('調製食料品', 0.1) and not is_ja('Cotton T-shirt', 0.1) and not is_ja('LED lamp for car 新品 12V 6000K waterproof bulb', 0.1) and not is_ja(' \u3000', 0.1)
    print('demo ok')


def main():
    ap = argparse.ArgumentParser(description='zeiban: 商品の文から米国 HTS の 10 桁を出す（標準入力に 1 行 1 件の JSON）')
    ap.add_argument('bundle', help='zeiban.bin と zeiban.json のあるフォルダ。demo と書くと自己確認だけ')
    ap.add_argument('encoder', nargs='?', help='transformers で読むエンコーダのフォルダ（リリースの zeiban-encoder）')
    ap.add_argument('-k', type=int, default=5, help='上位の件数')
    a = ap.parse_args()
    if a.bundle == 'demo':
        return demo()
    if not a.encoder:
        ap.error('エンコーダのフォルダが要る')
    z = Zeiban(a.bundle, a.encoder)
    for line in sys.stdin:
        if line.strip():
            print(json.dumps(z.classify(json.loads(line), a.k), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
