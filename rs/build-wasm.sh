#!/usr/bin/env bash
# zeiban.wasm を作る（WebAssembly。Bun・Node・Deno・ブラウザの中で、実行ファイルと同じ答えを出す）。
# 要るもの: Rust（wasm32-unknown-unknown の std つき。rustup なら `rustup target add wasm32-unknown-unknown`）と wasm-ld（LLVM の lld）。
# できるもの: rs/target/wasm32-unknown-unknown/release/zeiban.wasm（約 0.4 MB。外からの取り込みは無い）
# 組み立てる場所で中身が変わらないよう、ソースと依存の置き場のパスを、決まった名前に置き換える（--remap-path-prefix）。
set -eu
cd "$(dirname "$0")"
exports="zb_alloc zb_free zb_load zb_classify zb_out_ptr zb_out_len"
flags="-C target-feature=+simd128 -C linker=${WASM_LD:-wasm-ld} --remap-path-prefix=${CARGO_HOME:-$HOME/.cargo}=/cargo --remap-path-prefix=$PWD=/zeiban/rs"
for f in $exports; do flags="$flags -C link-arg=--export=$f"; done
RUSTFLAGS="$flags" exec cargo build --release --target wasm32-unknown-unknown "$@"
