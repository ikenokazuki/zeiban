// zeiban.wasm を読む小さな包み（Node 18 以上・Bun・Deno・ブラウザ。依存なし）。答えは Rust の実行ファイルと同じ。
//   import { load } from './zeiban.mjs';
//   const z = await load(wasm, manifest, bundle);   // 3 つとも Uint8Array（zeiban.wasm・zeiban.json・zeiban.bin の中身）
//   const r = z.classify({ name: 'Mens crew neck t-shirt', material: '100% cotton' });   // 入力と応答は README の「使い方」と同じ
// コマンドとして: node js/zeiban.mjs <zeiban.wasm> <一式のフォルダ> [k]   （標準入力に 1 行 1 件の JSON → 標準出力に 1 行 1 件の JSON）
export async function load(wasm, manifest, bundle) {
  const { instance } = await WebAssembly.instantiate(wasm, {});
  const ex = instance.exports;
  // wasm のメモリは伸びると付け替わるので、使うたびに ex.memory.buffer を取り直す
  const put = (b) => {
    const p = ex.zb_alloc(b.length);
    new Uint8Array(ex.memory.buffer, p, b.length).set(b);
    return p;
  };
  const out = () => new TextDecoder().decode(new Uint8Array(ex.memory.buffer, ex.zb_out_ptr(), ex.zb_out_len()));
  const mp = put(manifest);
  const rc = ex.zb_load(mp, manifest.length, put(bundle), bundle.length); // 一式の場所は、wasm の側が持ち続ける
  ex.zb_free(mp, manifest.length);
  if (rc !== 0) throw new Error(`zeiban: ${out()}`);
  const enc = new TextEncoder();
  return {
    /** req: {name, material?, categories?, description?, prefix?, confirm?} → 応答。k は上位の件数（確度は 5 で合わせてある） */
    classify(req, k = 5) {
      const b = enc.encode(JSON.stringify(req));
      const p = put(b);
      const rc = ex.zb_classify(p, b.length, k);
      ex.zb_free(p, b.length);
      const s = out();
      if (rc !== 0) throw new Error(`zeiban: ${s}`);
      return JSON.parse(s);
    },
  };
}

if (typeof process !== 'undefined' && process.argv[1] && import.meta.url === (await import('node:url')).pathToFileURL(process.argv[1]).href) {
  const { readFileSync, writeFileSync } = await import('node:fs');
  const [wasm, dir, k] = process.argv.slice(2);
  if (!wasm || !dir) throw new Error('使い方: node js/zeiban.mjs <zeiban.wasm> <一式のフォルダ> [k]');
  const z = await load(readFileSync(wasm), readFileSync(`${dir}/zeiban.json`), readFileSync(`${dir}/zeiban.bin`));
  const out = readFileSync(0, 'utf8').split('\n').filter((line) => line.trim()).map((line) => JSON.stringify(z.classify(JSON.parse(line), Number(k) || 5)));
  // まとめて 1 回で書く（Bun 1.3 は、パイプへの console.log を続けると、まれに改行を落とす）
  if (out.length) writeFileSync(1, `${out.join('\n')}\n`);
}
