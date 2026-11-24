# zeiban (税番)

[日本語](README.md) | English

A small model that predicts the 10-digit code of the US Harmonized Tariff Schedule (HTS) from a product text (name, material, categories, description). It runs on CPU only and uses no LLM and no network at inference time.

- **Small**: one 55.1 MB bundle, about 76 MB peak resident memory, 0.07–0.33 s per item on 2 CPU threads.
- **Answers with its doubts**: the 10-digit code, a confidence, alternates, the basic duty rate, and a **clarifying question** built from the official tariff text ("outer surface of textile materials — of cotton / of man-made fibers / other"), with the duty rate of each option.
- **Follows tariff revisions**: a daily watcher archives every HTS release and the differences between releases. The tables inside the model (tree, row texts, duty rates) are data, not weights.
- **English and Japanese** product texts, read by one multilingual reader.
- **Three ways to run it**: a Rust binary (fast, small), WebAssembly (Node, Bun, Deno, browsers), and a Python implementation. All give the same answers.

This file is a shorter English version of the [Japanese README](README.md), which is the complete write-up (every table, every experiment that did not help).

> **Pre-release: `v0.4.0-rc1` (Interim Preview)**
>
> While full `v0.4.0` retraining and the Stage-2 reranker are in progress, `v0.4.0-rc1` provides the **24-layer mixed 2/4-bit Stage-1 standalone bundle (`b32`, `avx2_int8_k64`)** on branch `v0.4.0-rc1`, GitHub Pre-release, and Hugging Face `revision="v0.4.0-rc1"`. Held-out sealed test splits are reserved for `v0.4.0`, so only development and calibration checks are reported here.
>
> - **Bundle & SHA-256 (`2026HTSRev22`)**: `zeiban.bin` `80,126,764` bytes (`80.1 MB`, sha256 `0f9f87623057465ff58b2eea96b1e84efbdbd8b87b2c397eaaa6e8ab12b55fc4`), `zeiban.json` `80,468` bytes (`c6bce1e00c4f36841b4b13f08198e8be53b64341197c3614904ec1971cf2ca40`), `zeiban.wasm` `489,171` bytes (`09db6b86c3138ec088345f4be050cc8a6911b5d6023800f043e669b6c6b2195a`).
> - **Dev `holdout` (568 rows, Rust AVX2 int8, 2 threads)**: top-1 at 10 digits `0.4137 → 0.5070` (+9.3 pts), 6 digits `0.5352 → 0.6285`, top-5 at 10 digits `0.6585 → 0.7465`.
> - **Calibration (5,884 non-training tune rows, CV ECE)**: ATLAS 6d `0.117 → 0.045` / 10d `0.062 → 0.046`, ruling rewrites 6d `0.069 → 0.023` / 10d `0.053 → 0.024`.
> - **Speed & memory (2 CPU threads, AVX2 int8, 128 items)**: load `18.22 ms`, median `118.35 ms`, **p95 `439.15 ms`**, peak RSS (`VmHWM`) `115.71 MB` (without AVX2, `portable` median is `946.3 ms`, p95 `3152.65 ms`).
> - **Runtime parity**: `ZEIBAN_FP32=1` matches `py/zeiban.py` `16/16` (max diff `5.9e-5`). Dynamic `int8` quantization across 24 layers can differ by `~0.2–0.3` in log-scores due to rounding ties (`AVX2 int8` vs Python `15/16` top-1, `Portable` vs `WASM` `8/8` top-1). Backward compatibility verified with `v0.3.0` (`8/8`) and `v0.2.1` (`4/4`).
> - **Try it**: build from branch `v0.4.0-rc1` (`cargo build --release`), use `v0.4.0-rc1` in GitHub Pre-release URLs, or `snapshot_download("ikeno-ada/zeiban", revision="v0.4.0-rc1")`.
>
> **Read this first**
>
> - The top-1 10-digit code is not accurate enough to file a customs entry as is. On held-out test items, top-1 accuracy is 46% at 10 digits and 55% at 6 digits for ruling-like text, and 27% / 42% for short e-commerce titles. The intended use is to look at the 6-digit code and the confidence, and to narrow the 10 digits with the alternates and the question. With questions answered correctly, 10-digit accuracy on e-commerce titles reaches 66% within three questions.
> - **v0.3.0 replaced the reader. On the test items, accuracy without questions is about the same as v0.2.1** (scored once, 2026-11-18). What improved is accuracy on rulings issued after the training data (10 digits, name + description: 27% → 39%) and accuracy after answering questions. Two targets set before development were not met: 30% at 10 digits on e-commerce titles, and 90% accuracy among answers with confidence ≥ 0.9.
> - Japanese customs-style text: 56–74% at 6 digits, 3–5 points below v0.2.1 for name-only and name + description inputs.
> - v0.3.0 is about 10 times slower than v0.2.1. If speed or Japanese customs-style text matters more to you, the v0.2.1 bundle works with the v0.3.0 binary (see [Get the bundle](#2-get-the-bundle)).
> - This is not a customs ruling or legal advice. Duty rates are the basic columns of the tariff schedule; additional duties are not included.

## Contents

- [Usage](#usage)
- [Accuracy](#accuracy)
- [Speed and size](#speed-and-size)
- [How it works](#how-it-works)
- [Data](#data)
- [Following tariff revisions](#following-tariff-revisions)
- [Strengths and limits](#strengths-and-limits)
- [What did not help](#what-did-not-help)
- [License and attribution](#license-and-attribution)

## Usage

### 1. Build the binary

Rust is required (checked with 1.91).

```sh
git clone https://github.com/ikenokazuki/zeiban.git
cd zeiban/rs
cargo build --release
```

This produces `rs/target/release/zeiban` (about 0.8 MB). Checked on x86-64 Linux. CPUs with AVX2 use dedicated kernels; others fall back to plain Rust automatically. ARM64 should work but has not been checked on real hardware.

### 2. Get the bundle

The bundle (weights, tariff tree, index) and its manifest go in one folder.

```sh
mkdir -p ../bundle && cd ../bundle
curl -LO https://github.com/ikenokazuki/zeiban/releases/download/v0.3.0/zeiban.bin
curl -LO https://github.com/ikenokazuki/zeiban/releases/download/v0.3.0/zeiban.json
echo "5413a838519cb1cf2d27ea4c9043e37d84aef5dcb85735da28628db9dbd919f1  zeiban.bin" | sha256sum -c
```

Earlier bundles and binaries (v0.1.0–v0.2.1) remain in the releases. The v0.3.0 binary also reads v0.2.x bundles: codes, confidences and duty rates are the same as with the v0.2.x binary, and only the questions follow the new logic. To use the v0.2.1 bundle (61.0 MB, 6–36 ms per item), replace `v0.3.0` with `v0.2.1` in the URLs above (sha256 of `zeiban.bin`: `ba2f7883e9d37315c2c64137bd3fbe8d420eb0d552aa9e808b6e43b78a9653e0`).

### 3. Run

One JSON object per line on standard input, one JSON object per line on standard output.

```sh
echo '{"name": "Womens tote bag"}' | ../rs/target/release/zeiban .
echo '{"name": "メンズ 半袖 Tシャツ", "material": "綿100% ニット"}' | ../rs/target/release/zeiban .
```

Input fields:

| Field | Content |
|---|---|
| `name` | Product name (required) |
| `material` | Material (optional) |
| `categories` | Shop categories: a string or a list of strings (optional) |
| `description` | Description (optional) |
| `prefix` | Leading digits you already know: 2, 4, 6, 8 or 10 digits (optional; see below) |
| `confirm` | `true` to be asked from the top level even where the model is confident (optional; see below) |

Material and categories work better in their own fields than inside the description. Input is cut at 256 tokens.

Response for the tote bag (reformatted; long texts shortened with "…", one alternate and 3 of the 15 options shown):

```json
{
  "hsCode": {
    "code": "4202924500",
    "description": "ARTICLES OF LEATHER; … > Trunks, suitcases, … > Other: > With outer surface of sheeting of plastics or of textile materials: > Travel, sports and similar bags: > Other (other than: With outer surface of textile materials)"
  },
  "sixDigit": "420292",
  "confidenceScore": 0.56,
  "sixDigitConfidence": 0.81,
  "dutyRate": {
    "general": "20%",
    "special": "Free (A,AU,BH,CL,CO,D,IL, JO,KR,MA,NP,OM,P,PA,PE,R,S, SG) 17.5% (E)",
    "other": "45%"
  },
  "alternates": [
    {"code": "4202923131", "description": "… > With outer surface of textile materials: > Of man-made fibers > Other (670)", "probabilityMass": 0.201, "dutyRate": {"general": "17.6%", "special": "…", "other": "65%"}}
  ],
  "questions": [
    {
      "digits": 8,
      "ask": true,
      "confidence": 0.57,
      "options": [
        {"prefix": "42029245", "label": "Travel, sports and similar bags: > Other (other than: With outer surface of textile materials)", "code": "4202924500", "probabilityMass": 0.466, "dutyRate": {"general": "20%", "special": "…", "other": "45%"}},
        {"prefix": "42029231", "label": "Travel, sports and similar bags: > With outer surface of textile materials: > Of man-made fibers", "code": "4202923131", "probabilityMass": 0.182, "dutyRate": {"general": "17.6%", "special": "…", "other": "65%"}},
        {"prefix": "42029215", "label": "Travel, sports and similar bags: > With outer surface of textile materials: > Of vegetable fibers and not of pile or tufted construction: > Of cotton (369)", "code": "4202921500", "probabilityMass": 0.044, "dutyRate": {"general": "6.3%", "special": "…", "other": "40%"}}
      ]
    }
  ],
  "htsRelease": "2026HTSRev22"
}
```

Response fields:

| Field | Content |
|---|---|
| `hsCode` | The top 10-digit code and its path in the tariff schedule |
| `sixDigit` | The first 6 digits of the top code (the internationally shared part of the HS) |
| `confidenceScore` | Estimated probability that the top 10-digit code is correct (0–1) |
| `sixDigitConfidence` | Estimated probability that the first 6 digits are correct |
| `dutyRate` | Duty columns of the schedule: `general`, `special`, `other` (column 2), in the official wording |
| `alternates` | Up to 4 alternates; `probabilityMass` is the share among the top 5 |
| `questions` | Zero or one question at the place where the candidates split |
| `htsRelease` | The HTS release this bundle was built from |
| `codes`, `scores` | The top 5 codes and scores (for checking) |

A second argument to the binary changes the number of top answers (default 5).

### Questions and follow-up calls

When the name alone does not decide the code, the model returns where it is unsure as a question.

- `digits`: the level of the question (8 digits in the example: is the outer surface textile; if so, cotton or man-made fibers).
- `options`: up to 30 options. `label` is the official tariff text, `code` is the code you get if you choose it, `dutyRate` is the rate in that case — the options alone show how the answer changes the duty (20%, 17.6%, 6.3% above). `probabilityMass` is the share among the candidates behind the options.
- `ask`: a hint on whether asking is worthwhile: `true` when `confidence` (the estimated probability that the answer is right down to that level) is below 0.7.
- After getting an answer, call again with the `prefix` of the chosen option. The model then chooses only among the rows below it.

```sh
echo '{"name": "Womens tote bag", "prefix": "42029215"}' | ../rs/target/release/zeiban .
```

This settles on 4202921500 (a cotton tote bag, 6.3%) with no further question.

**Which level is asked** (reworked in v0.3.0):

1. Going down from the top, find the first level where confidence is below 0.7. Confident levels are skipped (in the example, confidence down to 6 digits is 0.81, so the question is at 8 digits).
2. Gather up to 30 candidates that agree with the top answer above that level, in score order. The question is the first level where they split.
3. If every level is confident, the first split among the top 30 is returned with `ask` set to `false`.

Up to v0.2.x the question was the first split among the top 5; when the right branch was not among them, the options did not contain the right answer. With 30 candidates and confident levels skipped, more answers are reached with the same number of questions ([Answering questions](#answering-questions)).

Adding `"confirm": true` asks from the first split among the top 30 without skipping confident levels (6 digits in the example, with `ask` `false`). This takes more questions, but it lets a person correct levels where the model is confidently wrong.

Every call returns an answer, and the binary keeps no state. The question is an extra field; you can ignore it and use the top code.

### Confidence and duty rates

- `confidenceScore` and `sixDigitConfidence` were calibrated on rows not used for training (ruling-like text and rewrites of rulings). See [Calibration](#calibration).
- `probabilityMass` is a share among candidates, not a probability of being correct.
- `dutyRate` is the basic columns only. Chapter 99 additional duties (Section 232, Section 301 and others), anti-dumping duties, footnotes and units of quantity are not included. **It is not the amount you will pay.**

### Embedding it in another system

The binary answers one line per input line. Keep it running and feed it lines, and the load time (about 15 ms) is paid once. There is no HTTP server; add a thin wrapper if you need one.

### WebAssembly

The same computation is distributed as `zeiban.wasm` (about 0.5 MB). It runs inside any process that can run WebAssembly. The module has no imports: it touches no files and no network (the caller reads the bundle and passes it in).

```sh
# in the repository root; bundle/ is the folder from step 2
curl -LO https://github.com/ikenokazuki/zeiban/releases/download/v0.3.0/zeiban.wasm
echo "d891a0e8caf2f1ad44c4d7053ccd498eb9c608506bd1099d44354cd9b1cb1434  zeiban.wasm" | sha256sum -c

echo '{"name": "Womens tote bag"}' | node js/zeiban.mjs zeiban.wasm bundle
```

```js
import { readFileSync } from 'node:fs';
import { load } from './js/zeiban.mjs';

const z = await load(readFileSync('zeiban.wasm'), readFileSync('bundle/zeiban.json'), readFileSync('bundle/zeiban.bin'));
const r = z.classify({ name: 'Womens tote bag', material: 'cotton canvas' });
console.log(r.hsCode.code, r.sixDigit, r.dutyRate.general);   // 4202921500 420292 6.3%
```

- Same answers as the Rust binary: on 467 inputs (including follow-up calls) the top-5 codes and order, confidences, questions and duty rates were all identical (largest score difference 0.0002). Checked with Bun 1.3 and Node 22.
- It runs on one thread and is slow: about 0.5–0.6 s on average for inputs of ordinary length, and 1.0–1.2 s for the longest inputs (256 tokens). The 55 MB bundle lives inside the wasm memory.
- To build it yourself: `rs/build-wasm.sh` (needs the Rust `wasm32-unknown-unknown` target and LLVM's `wasm-ld`). The script remaps paths, so the same source gives the same file wherever it is built (Rust 1.91.1, LLD 21.1.7; other tool versions may give a different file).
- To call it from another language, there are six exports (`zb_alloc`, `zb_free`, `zb_load`, `zb_classify`, `zb_out_ptr`, `zb_out_len`); the calling order is described in the comment before `mod wasm` in `rs/src/main.rs`.

### Python

`py/zeiban.py` reads the same bundle: the reader in torch, the rest (word-match score, tree scoring, confidence, questions) in numpy. Put the tokenizer definitions next to the bundle.

```sh
pip install numpy torch tokenizers

# in the repository root; bundle/ is the folder from step 2
curl -L -o bundle/tokenizer.json https://github.com/ikenokazuki/zeiban/releases/download/v0.3.0/tokenizer.json
curl -L -o bundle/bmx-tokenizer.json https://github.com/ikenokazuki/zeiban/releases/download/v0.3.0/bmx-tokenizer.json

echo '{"name": "Womens tote bag"}' | python3 py/zeiban.py bundle
```

```python
import sys
sys.path.insert(0, 'py')
from zeiban import Zeiban

z = Zeiban('bundle')
r = z.classify({'name': 'Womens tote bag', 'material': 'cotton canvas'})
print(r['hsCode']['code'], r['sixDigit'], r['dutyRate']['general'])   # 4202921500 420292 6.3%
```

- Same answers as the Rust binary: checked on 3,805 inputs (including follow-up calls), against both the AVX2 and the plain Rust kernels (largest score difference 0.0004).
- About 50 ms per short item, about 3 s to load, about 650 MB of memory (torch). Use the Rust binary when a small footprint matters.
- Checked with Python 3.11, torch 2.9 (CPU), numpy 2.4, tokenizers 0.22.
- v0.2.x bundles are still read with the encoder folder as a second argument (`Zeiban('bundle', 'zeiban-encoder')`, needs transformers).

### Hugging Face

The same files are on Hugging Face: <https://huggingface.co/ikeno-ada/zeiban> (bundle, tokenizer definitions and `zeiban.py` in one folder).

```python
import sys
from huggingface_hub import snapshot_download

d = snapshot_download('ikeno-ada/zeiban')
sys.path.insert(0, d)
from zeiban import Zeiban

z = Zeiban(d)
r = z.classify({'name': 'Womens tote bag', 'material': 'cotton canvas'})
print(r['hsCode']['code'], r['sixDigit'], r['dutyRate']['general'])   # 4202921500 420292 6.3%
```

The previous version is at the tag `v0.2.1` of that repository.

## Accuracy

### How it was measured

- **Metric**: whether the top code equals the gold code, at 10 and at 6 digits.
- **Test items**, held out for final scoring:
  - Ruling-like text: the 200 test items of ATLAS (questions that an LLM wrote from US customs rulings). The main reading uses the 145 items left after removing 55 that have a near-identical text in the training split.
  - Short e-commerce titles: 422 of the 632 items of HSCodeComp (AliExpress), split off at random before any result was seen.
  - Everyday goods: 98 items from the evaluation set of the open-source project Sora (6-digit labels).
  - Japan Customs rulings: 1,788 items (6-digit labels).
- **v0.3.0 was scored on the test items once** (2026-11-18), after weights, calibration and binary were frozen; nothing was changed afterwards. This was the second use of the test items after the v0.1.0 scoring (the fourth for the Japan Customs test items). No model was ever selected on them.
- **New rulings**: 2,997 rows from rulings issued after the training data. One half was used for comparisons during development; the other half was not read until one final run.
- **Development items** (134 ATLAS validation items without near-duplicates, the other 210 HSCodeComp items, 52 everyday goods, 858 Japan Customs items): used a dozen times while choosing the shape of v0.1.0–v0.2.x. v0.3.0 looked at them once, to decide whether to adopt it, under a rule written down beforehand.
- The sets are small: differences of a few points are not differences. Intervals are 95%, from resampling items.

### Test results

v0.2.1 was read on the same items for paired differences (its numbers reproduce the v0.1.0 scoring; v0.1.0–v0.2.1 give identical answers for English input).

| | Needs at run time | Ruling-like text, 10 / 6 digits | E-commerce titles, 10 / 6 digits | Everyday goods, 6 digits |
|---|---|---|---|---|
| Items | | 145 | 422 | 98 |
| **zeiban v0.3.0** | CPU, 55.1 MB | **0.462** / **0.545** | **0.268** / **0.419** | **0.908** |
| – 95% interval | | 0.379–0.538 / 0.462–0.621 | 0.227–0.313 / 0.372–0.467 | 0.847–0.959 |
| – gold code in the top 5 | | 0.586 / 0.697 | 0.552 / 0.633 | 0.969 |
| – after one question answered correctly | | 0.531 / 0.655 | 0.434 / 0.607 | 0.949 |
| zeiban v0.1.0–v0.2.1 | CPU, 57.5–61.0 MB | 0.462 / 0.586 | 0.249 / 0.384 | 0.908 |
| – gold code in the top 5 | | 0.621 / 0.738 | 0.498 / 0.588 | 0.980 |
| – after one question answered correctly | | 0.517 / 0.683 | 0.363 / 0.536 | 0.959 |
| v0.3.0 − v0.2.1 (paired, 95% interval) | | ±0.000 [−0.069, +0.062] / −0.041 [−0.104, +0.028] | +0.019 [−0.024, +0.057] / +0.036 [−0.009, +0.081] | ±0.000 |
| Lexical index alone (the v0.2.x index, no model) | CPU | 0.352 / 0.448 | 0.156 / 0.270 | 0.765 |
| A public 6-digit model (570M parameters) | CPU | — / 0.731 | — / 0.353 | 0.786 |

On all 200 ATLAS test items, v0.3.0 scores 0.515 / 0.600 (v0.2.1: 0.520 / 0.640). The public 6-digit model is `Kenpache/hs-code-classifier-en`.

Targets set before development:

| Target | v0.3.0 | | v0.1.0 |
|---|---|---|---|
| Everyday goods, 6 digits ≥ 65% | 90.8% | met | 90.8%, met |
| ATLAS test (all 200), 10 digits above the 40.0% of Atlas | 51.5% | met | 52.0%, met |
| HSCodeComp test, 10 digits ≥ 30% | 26.8% | not met | 24.9%, not met |
| Accuracy ≥ 90% among answers with confidence ≥ 0.9 | 85.7% at 10 digits (14 items), 86.7% at 6 digits (30 items) | not met | 90.9% (22), 95.3% (85), met |
| Bundle ≤ 60 MB, resident memory ≤ 80 MB, ≤ 1 s per item | 55.1 MB, 75.6 MB, p95 0.33 s (0.66 s without AVX2) | met | 57.5 MB, 74.3 MB, p95 30 ms, met (v0.2.x: 61.0 MB, size not met) |

How to read this:

- **Accuracy without questions is about the same as v0.2.1.** Ruling-like text is equal at 10 digits and slightly lower at 6 digits and in the top 5; e-commerce titles are slightly higher at every level. All intervals include zero. The test numbers do not show an improvement.
- **The gain seen on the development items for e-commerce titles (+0.11 at 10 digits) was +0.02 on the test items.** The single development look was optimistic.
- **The 30% target for e-commerce titles was missed again** (26.8%). With questions answered correctly it rises to 43.4% (one), 57.8% (two), 66.4% (three).
- **Accuracy among answers with confidence ≥ 0.9 fell below the 90% target** (few items: 14 and 30). v0.1.0 met it.
- At 6 digits, ruling-like text is below the public 6-digit model (0.545 vs 0.731); e-commerce titles (0.419 vs 0.353) and everyday goods (0.908 vs 0.786) are above it.
- The 51.5% on all 200 ATLAS items is above the 40.0% reported for Atlas, but 55 of the 200 have a near-identical text in the training split, and a lexical index alone reaches 43.5%. Atlas has not published a number without those items.

### Rulings issued after the training data

A public dataset of product names and descriptions extracted from US customs rulings (`fklc/cbp-rulings-past-2012`) has 2,997 rows from rulings dated 2025-10-01 to 2026-07-23 that are in neither the training data nor the index. They were split in two by ruling number. All numbers are "v0.2.1 → v0.3.0".

| Rows | Input | Top-1, 10 digits | Top-1, 6 digits | Top-1, 4 digits | Gold 10-digit code in the top 5 |
|---|---|---|---|---|---|
| **Half never read before** (1,403 rows, scored once) | name + description | 0.270 → **0.387** (+0.116 [+0.088, +0.144]) | 0.411 → **0.533** | 0.529 → 0.649 | 0.528 → 0.659 |
| 〃 | name only | 0.149 → **0.225** (+0.076 [+0.055, +0.096]) | 0.232 → 0.316 | 0.324 → 0.435 | 0.300 → 0.410 |
| Half used during development (1,594 rows, used many times) | name + description | 0.280 → 0.419 | 0.402 → 0.545 | 0.499 → 0.657 | 0.540 → 0.659 |
| 〃 | name only | 0.144 → 0.236 | 0.206 → 0.314 | 0.279 → 0.393 | 0.287 → 0.401 |

- Rely on the half that was never read (0.387 at 10 digits, 0.533 at 6). The other half was used to choose the reader, the training length and the question logic, and reads about 0.03 higher.
- **The comparison favours v0.3.0**: older rulings from the same dataset (up to 2025-09), written in the same style, were added to its training data. v0.2.1 never saw that style.
- The top 2 digits (chapter) are right for 0.78 of the rows in the development half: one in five is already wrong at the chapter.
- Leaves with many training examples do much better: 0.54 at 10 digits for leaves with 20 or more examples, 0.18 for leaves with none (measured with the weights one step before the released ones).
- On the half never read, accuracy among answers with `confidenceScore` ≥ 0.9 is 0.986 (69 rows) for name + description and 0.810 (63 rows) for name only: confidence is too high for name-only inputs.

### Development items

| | Items | v0.2.1 | **v0.3.0** | Difference |
|---|---|---|---|---|
| Ruling-like text (ATLAS, no near-duplicates), 10 digits | 134 | 0.463 | **0.425** | −0.037 [−0.090, +0.008] |
| 〃 6 digits | 134 | 0.679 | **0.634** | −0.045 [−0.097, +0.007] |
| Ruling-like text (ATLAS, all), 10 digits | 200 | 0.525 | **0.485** | −0.040 [−0.080, −0.005] |
| E-commerce titles (HSCodeComp), 10 digits | 210 | 0.262 | **0.371** | +0.110 [+0.043, +0.176] |
| 〃 6 digits | 210 | 0.381 | **0.510** | +0.129 [+0.062, +0.200] |
| Everyday goods, 6 digits | 52 | 0.923 | **0.942** | +0.019 [+0.000, +0.058] |
| Main measure (mean of the two 10-digit numbers) | | 0.362 | **0.398** | +0.036 [−0.006, +0.079] |

The adoption rule, written before looking: the paired difference of the main measure must be at least −0.02 with the upper end of its interval at or above zero, and none of four watched measures (ATLAS without near-duplicates at 10 digits, HSCodeComp at 10 digits, everyday goods at 6 digits, Japan Customs name + description at 6 digits) may drop by 0.03 or more with an interval excluding zero. It passed. Two measures outside that list did drop clearly: ATLAS on all 200 items (above) and Japan Customs name-only input (0.639 → 0.603). On the same items, an LLM (Gemini Decision 1, choosing level by level; the teacher that labelled part of the training data) scores 0.299 / 0.500 on ruling-like text and 0.524 / 0.638 on e-commerce titles (10 / 6 digits).

### Answering questions

Top-1 accuracy at 10 digits when questions are answered correctly: if the options contain the right branch, it is chosen and the model is called again; if not, the flow stops. Three flows: **follow `ask`** (answer only when `ask` is `true`), **always answer**, and **confirm** (`"confirm": true`, answering from the top).

Test items (0 questions → up to 1 → up to 2 → up to 3 → up to 4):

| Items | Flow | 0 | 1 | 2 | 3 | 4 | Questions asked (mean) |
|---|---|---|---|---|---|---|---|
| Ruling-like text (145) | v0.1.0–v0.2.1 (always answer, old logic) | 0.462 | 0.517 | 0.593 | 0.648 | — | |
| 〃 | v0.3.0, follow `ask` | 0.462 | 0.531 | 0.614 | 0.614 | 0.621 | 1.20 |
| 〃 | v0.3.0, always answer | 0.462 | 0.531 | 0.648 | 0.676 | 0.690 | 2.17 |
| 〃 | v0.3.0, confirm | 0.462 | 0.524 | 0.634 | 0.731 | 0.772 | 3.12 |
| E-commerce titles (422) | v0.1.0–v0.2.1 (always answer, old logic) | 0.249 | 0.363 | 0.500 | 0.576 | — | |
| 〃 | v0.3.0, follow `ask` | 0.268 | 0.405 | 0.502 | 0.533 | 0.538 | 1.18 |
| 〃 | v0.3.0, always answer | 0.268 | 0.434 | 0.578 | 0.664 | 0.682 | 2.09 |
| 〃 | v0.3.0, confirm | 0.268 | 0.403 | 0.509 | 0.718 | 0.839 | 3.12 |

New rulings (the half used during development, 1,447 rows with a 10-digit gold code, name + description):

| Bundle and binary | Flow | 0 | 1 | 2 | 3 | 4 | Questions asked (mean) | Right branch missing from the options |
|---|---|---|---|---|---|---|---|---|
| v0.2.1 with its own binary (old logic) | follow `ask` | 0.280 | 0.400 | 0.478 | 0.498 | 0.500 | 1.17 | 37% |
| v0.2.1 with the v0.3.0 binary (new logic) | follow `ask` | 0.280 | 0.447 | 0.577 | 0.614 | 0.623 | 1.48 | 32% |
| **v0.3.0** | follow `ask` | 0.419 | 0.580 | 0.683 | 0.710 | 0.717 | 1.29 | 21% |
| **v0.3.0** | always answer | 0.419 | 0.585 | 0.713 | 0.766 | 0.780 | 2.18 | 22% |
| **v0.3.0** | confirm | 0.419 | 0.547 | 0.658 | 0.793 | 0.867 | 3.19 | 11% |

- **Questions move the 10 digits more than anything else.** For the test e-commerce titles: 0.268 without questions, 0.664 with up to three, 0.839 with confirm and up to four.
- The new question logic alone, with the same v0.2.1 model, raised the three-question number on new rulings from 0.498 to 0.614 (+0.116 [+0.100, +0.133]).
- "Answered correctly" is an upper bound: a user may not be able to choose even with the official text in front of them, and some questions have many options (up to 30).
- On the test items, the first question lacks the right option for 19% of ruling-like items and 27% of e-commerce titles (v0.2.x: 21% and 34%). `ask` is `true` for about 80% of both.

### Japanese

One reader reads every language. Test items: 1,788 advance classification rulings published by Japan Customs (processed 2023-11-16 to 2026-11-13; 70% of the 9,039 rows went to training, 10% to development, 20% to test, split by registration number). The label is the first 6 digits of the Japanese tariff number. Numbers are top-1 at 6 digits.

| Input | v0.1.0 | v0.2.1 | **v0.3.0** (95% interval) | v0.3.0 − v0.2.1 |
|---|---|---|---|---|
| Product name only | 0.345 | 0.633 | **0.584** (0.562–0.607) | −0.049 [−0.063, −0.035] |
| Name + description | 0.333 | 0.769 | **0.739** (0.718–0.760) | −0.030 [−0.048, −0.012] |
| Facts only (the description without its leading summary sentence) | 0.148 | 0.538 | **0.559** (0.535–0.581) | +0.021 [−0.001, +0.043] |

- **Name-only and name + description inputs got worse than v0.2.1**, which had a second reader retrained on these rulings. Facts-only input, the closest to what a user would write, is slightly better.
- These numbers need a discount. 39% of the test items have a near-identical ruling in the training part (the same kinds of goods are asked about repeatedly); for items without one, v0.3.0 scores 0.575 / 0.652 / 0.423. Names and summary sentences are written by customs after the classification and are close to the tariff wording.
- For name-only input, accuracy among answers with `sixDigitConfidence` ≥ 0.9 is 0.81 (95 items): confidence is too high.
- Whether Japan and the US ever classify the same goods under different 6-digit codes was not examined.

Source: Japan Customs website, "Advance Classification Ruling Cases" (<https://www.customs.go.jp/searchsv/jitsv001.jsp>), listing retrieved 2026-11-16 and processed. The texts are not in this repository.

### Calibration

Confidence was fitted on rows outside the development and test items (824 ruling-like rows and 5,060 rewrites of rulings, not used for training). On the test items (v0.3.0; accuracy and number of items per confidence band):

| Items | Digits | 0–0.5 | 0.5–0.7 | 0.7–0.9 | ≥ 0.9 |
|---|---|---|---|---|---|
| Ruling-like text (all 200) | 10 | 0.292 (96) | 0.600 (55) | 0.857 (42) | 0.857 (7) |
| Ruling-like text (all 200) | 6 | 0.333 (66) | 0.491 (53) | 0.892 (74) | 0.857 (7) |
| E-commerce titles | 10 | 0.140 (243) | 0.398 (103) | 0.464 (69) | 0.857 (7) |
| E-commerce titles | 6 | 0.236 (165) | 0.337 (101) | 0.658 (149) | 0.857 (7) |
| Everyday goods | 6 | 0.667 (3) | 0.714 (7) | 0.944 (72) | 0.875 (16) |

- Few answers get a confidence of 0.9 or more (14 of 622 at 10 digits, 30 of 720 at 6 digits), and their accuracy is 0.86–0.87, below the 0.90 target.
- **For e-commerce titles, confidence above 0.5 is too high** (0.464 accuracy in the 0.7–0.9 band at 10 digits). For this kind of text, do not read a confidence below 0.9 as "probably right". This has not improved since v0.2.x.

## Speed and size

Measured with the released bundle on an AMD EPYC (Milan) virtual machine, 2 threads, with other work running on the machine.

| | AVX2 kernels | Plain Rust kernels (forced on the same machine) |
|---|---|---|
| Inputs of ordinary length (median 60 tokens): median, p95 | 73 ms, 296 ms | 144 ms, 622 ms |
| 256-token inputs: median, p95 | 295 ms, 324 ms | 616 ms, 658 ms |
| Load (until the first item can be taken) | about 15–30 ms | about 15–30 ms |
| Peak resident memory | 75.6 MB | 75.0 MB |

| In the bundle | MB |
|---|---|
| Six layers of the reader (4-bit) | 16.5 |
| Token embeddings of the reader (4-bit) | 12.8 |
| Lexical index (BMX) | 12.3 |
| Node vectors (4-bit) | 5.1 |
| Row texts, tree, duty rates | 2.7 |
| Tokenizer tables of the lexical index, NFKC tables | 2.4 |
| Tokenizer tables of the reader (BPE) | 1.9 |
| Other parts of the reader | 1.5 |
| **Total** | **55.1** |

v0.2.1 took 6–36 ms per item: the reader went from 3 layers of width 256 to 6 layers of width 512, which costs about 10 times the time. The bundle is memory-mapped, so 55 MB of the resident memory is the file itself.

## How it works

```
product text (name, material, categories, description)
   │
   ├─(1) word-match path ─ BMX index over 19,926 leaf documents ─→ top 64 leaves and scores
   │
   └─(2) meaning path ─ tokenizer (BPE, at most 256 tokens)
                        → 6-layer reader (width 512) → mean of all tokens, projected to a 768-dim state h
                        → five query vectors q_t (2, 4, 6, 8, 10 digits)
                        → cosine with the node vectors e_t
   ▼
(3) level score   s_t = scale_t × cos(q_t, e_t) + λ_t × BMX_t
(4) leaf score    = sum of the five log softmax(s_t) along the path of the leaf
   ▼
top 5 codes → confidence, question, duty rates
```

The schedule is a tree: chapter (2 digits) → heading (4) → subheading (6) → 8 digits → 10 digits. All five levels are scored in one pass and the leaf with the best path sum wins, so a mistake at an upper level can be recovered by strong lower-level scores.

- **Reader**: the first 6 of the 24 text layers of `google/embeddinggemma-2` (Apache-2.0), a multilingual embedding model from Google DeepMind, fine-tuned for tariff classification. It replaced the 3-layer reader of v0.2.x built from `cl-nagoya/ruri-v3-30m`.
- **4-bit weights**: trained first without rounding, then with 4-bit rounding in the loop (the unrounded model as its own teacher). The released values are the integers at the end of training. All weights are 4-bit, about 70 million parameters in total.
- **Reduced vocabulary**: 49,106 of the 262,144 tokens are kept (those frequent in the training texts, plus single characters and bytes). Dropped tokens split into shorter ones; nothing becomes unknown.
- **Node vectors**: one 256-dim vector for each of the 38,276 nodes of the tree, shipped pre-computed.
- **BMX**: a lexical score in the BM25 family with entropy weighting (arXiv 2408.06643). Words such as "cotton", "knitted" or "stainless" often decide the row, so the word-match score is added to the meaning score. The lexical score alone reaches 0.287 at 10 digits on the new rulings (0.419 with the reader).
- **Confidence**: per level, a logistic regression on the share of top candidates that agree with the top answer and on its score. A separate fit is used after a `prefix` is given.
- **Training**: about 510,000 rows (rewrites of rulings, synthetic product names, ATLAS training questions) plus about 150,000 added rows (rulings, e-commerce texts, Japan Customs and UK rulings) repeated 4 times; 12,000 steps without rounding, then 3,000 steps with 4-bit rounding; about 1.5 hours on one L4 GPU. Losses: softmax over all nodes per level, softmax among siblings, and a contrast among the gold nodes in the batch. Rows with labels down to only some level (6-digit rulings) contribute only to those levels. Teacher-labelled e-commerce rows use hard labels down to the level where the teacher's probability is at least 0.7 and soft labels below.

## Data

| Data | Source | License | Use | Size |
|---|---|---|---|---|
| Tariff schedule (HTS) | US International Trade Commission, release 2026HTSRev22 | Work of the US Government | Tree, row texts, duty rates | 19,926 leaves, 38,276 nodes |
| Questions and answers made from rulings | `Dayanand314Krishna/cross_rulings_hts_dataset_for_tariffs` on Hugging Face (a copy of the ATLAS dataset) | Apache-2.0 | Training split for training and the index; validation and test for evaluation | 18,233 training rows |
| Texts of US customs (CBP) rulings | `orlandowhite/hts-cross-rulings` on Hugging Face | No license stated on the dataset; the rulings are US Government documents | Only LLM rewrites into product texts are used, for training and the index | 40,007 rewrites |
| Product names and descriptions extracted from rulings (from v0.3.0) | `fklc/cbp-rulings-past-2012` on Hugging Face | MIT | Rulings up to 2025-09 for training; later ones for evaluation only | 38,815 training rows, 2,997 evaluation rows |
| Synthetic product names | Written per leaf by an LLM (Claude Haiku 5.5), 10 in English and 10 in Japanese | Own work | Training and the index | 350,251 kept |
| E-commerce product texts | Amazon ESCI (`tasksource/esci`; US, JP, ES) and `Shopify/product-catalogue` | Apache-2.0 | Labelled by an LLM (Gemini Decision 1), for training | 23,588 products |
| E-commerce titles with gold codes | `ATH-MaaS/HSCodeComp` on Hugging Face | Apache-2.0 | Evaluation only | 632 |
| Japan Customs advance classification rulings | Search-result listing on the Japan Customs website | Terms of use of the website (usable with attribution) | Training (70%), evaluation (20% test, 10% development) | 9,039 rows |
| UK Advance Tariff Rulings (from v0.3.0) | Search-result listing of HM Revenue & Customs | Open Government Licence v3.0 | Training (6-digit labels) | 6,172 rulings |

Processing, in short:

- Chapters 98 and 99 are not targets. Gold codes that no longer exist in the current release are mapped to the deepest remaining digits.
- Ruling texts contain names and addresses of the people who asked. Only LLM rewrites into product texts are used; names and addresses are neither written nor kept, and the ruling texts are in neither the training data nor the bundle.
- Synthetic names are checked by a second LLM call that must pick the original leaf among close ones; 88% were kept.
- LLM labels on e-commerce texts are a teacher's answers, not gold (the same teacher is 52% correct at 10 digits on the development e-commerce titles).
- **Leak guards**: rulings that look like the source of an ATLAS validation or test item are left out of the rewrites (1,720 rulings) and of the extracted rows (11,684 rows; every row sharing the first 8 digits with such an item is dropped without looking at the text). Rows that look like the source of a calibration row (2,157) and rulings dated 2025-10-01 or later (2,997 rows) are left out as well. No prompt sent to an LLM contained development gold codes, test items or ruling texts.
- Training data and training scripts are not in this repository; the runtime source, the Python implementation and the bundle are.

## Following tariff revisions

The HTS changes often: 33 releases in 2025 and 24 by mid-November 2026. Of the 10-digit codes in 55,000 rulings from 2000–2025, only 41% still exist in the current release.

- **Release watcher** (`.github/workflows/watch.yml`, daily): saves each new release and the difference from the previous one. Past releases cannot be downloaded later from the source, so not missing one matters.
- **Archive** (`releases/`, `diffs/`): 56 releases from the 2025 basic edition on, and the 55 differences, each classified as A (no change to product rows, 44 of 55), B (small changes, 8) or C (40 or more leaves added or removed, 3). B and C open an issue.
- **Remapping gold codes** (`hts/rev_remap.py`): moves a 10-digit code from an old release to a new one, falling back to the deepest remaining digits when a row was split, merged or removed.
- **Ruling revocations** (`.github/workflows/bulletin.yml`, weekly): extracts notices of revoked or modified rulings from the Customs Bulletin (numbers, dates and codes only; no ruling text).
- **Not done yet**: rebuilding the bundle when product rows change is not automated. The current bundle was built from 2026HTSRev22; Rev23 has the same product rows, so the codes still apply. For v0.3.0, making node vectors for newly added rows from their text has not been verified; until the bundle is rebuilt such rows are found by the lexical score only. There is no measured result yet for a simulated revision.

## Strengths and limits

Strengths:

- Runs anywhere: CPU only, 55.1 MB, about 76 MB resident, no GPU, no network, no API cost; product texts never leave the machine.
- Deterministic: the same input gives the same answer.
- Shows its doubts: confidence, alternates, a question about where it is unsure, and the duty rate of each option.
- States the tariff release it used, and tracks revisions daily.
- The data sources, the processing, the measurements and the experiments that did not help are written down.

Limits:

- **The top-1 10-digit code is not at filing quality**, especially for short e-commerce titles (27% on the test items; experts are reported at 95% on the same kind of items).
- **Replacing the reader did not improve test accuracy without questions.** The gains are on rulings issued after the training data and after answering questions.
- Short titles often lack what decides the 10 digits (material, use, size). Questions or the input fields are meant to fill that in.
- Confidence is too high for e-commerce titles and short name-only inputs.
- Ruling-like text at 6 digits is below a public 6-digit model (0.545 vs 0.731 on the test items).
- US HTS only.
- Japanese customs-style text is 3–5 points below v0.2.1 for name-only and name + description inputs.
- About 10 times slower than v0.2.1; over a second for the longest inputs in WebAssembly.
- Section and chapter notes and the General Rules of Interpretation are not read as text; goods decided by notes (bags, parts, sets) are easy to get wrong.
- Duty rates are the basic columns only.
- Not checked on real ARM64 hardware.

## What did not help

For v0.3.0, measured on the development half of the new rulings (10 digits, top-1; released model 0.419):

| Tried | Result |
|---|---|
| Training twice as long and adding Japan Customs and UK rulings | 0.414 → 0.419, no visible difference; recall in the top 10 fell from 0.763 to 0.739. Released anyway because the single reader needs the Japanese rows |
| A memory of training examples (adding scores from the leaves of nearby examples) | −0.005 with input-dependent mixing, −0.020 with fixed mixing; hurts leaves with few examples |
| A margin loss against the most confusable wrong node at the chapter and heading levels | ±0.000 |
| No layers at all: reading with the mean of token vectors (distilled from the 6-layer reader) | 0.116, below the lexical score alone (0.287) |

What helped in v0.3.0 was replacing the reader with the first layers of a larger pre-trained model (0.280 → 0.419 on those rows) and reworking the questions. Accuracy without questions did not move from 0.42 with longer training, more data or loss changes. The experiments for v0.1.0–v0.2.x (looped layers, the full 10-layer base, a memory of chapter notes, a re-ranking stage, model averaging and others — none of which helped) are in the [Japanese README](README.md#試して入れなかったもの).

## License and attribution

- Code and the bundle (which contains weights) are under the Apache License 2.0 (`LICENSE`, `NOTICE`).
- The tariff schedule, the Customs Bulletin and the rulings are works of the US Government.
- Base model of the reader (from v0.3.0): [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2) (Apache-2.0). Only the first six layers of the text encoder are used; the model was fine-tuned, quantized to 4 bits, and its vocabulary reduced. Its model card states: "Deployments must adhere to the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy)".
- The vocabulary and tokenizer of the lexical index and the starting values of the node vectors derive from [`cl-nagoya/ruri-v3-30m`](https://huggingface.co/cl-nagoya/ruri-v3-30m) (Apache-2.0), built on [`sbintuitions/modernbert-ja-30m`](https://huggingface.co/sbintuitions/modernbert-ja-30m) (MIT).
- Training and index data: "CROSS Rulings HTS Dataset for Tariff Classification by Flexify.AI Inc. (https://www.flexify.ai)" (Apache-2.0, with changes), the Amazon Shopping Queries Dataset (ESCI, Apache-2.0), Shopify product-catalogue (Apache-2.0), [`fklc/cbp-rulings-past-2012`](https://huggingface.co/datasets/fklc/cbp-rulings-past-2012) (MIT).
- Japan Customs: source "Japan Customs website (<https://www.customs.go.jp/>), Advance Classification Ruling Cases"; the search-result listing was processed.
- HM Revenue & Customs Advance Tariff Rulings (<https://www.tax.service.gov.uk/search-for-advance-tariff-rulings>). Contains public sector information licensed under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
- Part of the training data was produced with LLMs (Anthropic Claude Haiku 5.5, Google Gemini Decision 1).
- The archived releases 2025HTSBasic–2026HTSRev21 under `releases/` come from Budget-Lab-Yale/tariff-rate-tracker (MIT; `releases/LICENSE.Budget-Lab-Yale`).

The answers of this model are a reference for classification. They are not legal advice and not a decision of a customs authority.

References: ATLAS (arXiv:2509.18400), HSCodeComp (arXiv:2510.19631), BMX (arXiv:2408.06643). The full list is in the [Japanese README](README.md#参考にした研究).
