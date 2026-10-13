# כתב-יד AI — Technical Handoff for AI Analysis

> **Purpose of this file.** A complete, self-contained technical briefing so any
> AI model or engineer can understand, run, analyze, and extend this project
> **with zero prior context**. Hebrew HTR (handwritten text recognition) system,
> fully offline, CPU-first. Companion docs: `PROJECT_SUMMARY.md` (running change
> log, in Hebrew) and `USER_GUIDE.md` (end-user manual, in Hebrew).

---

## 1. One-paragraph summary

An offline Hebrew **handwriting → text** system with a CRNN+CTC optical model
(3.4M params), synthetic pretraining from Windows Hebrew fonts, per-writer
fine-tuning with **regression-gated** training (a new model is saved only if it
beats the old one on a fixed held-out set), a statistical **language-model
post-corrector** (word counts + bigrams + trigrams over a 2.5M-line rabbinic
Hebrew corpus, driving a beam-search line reranker), a desktop GUI (local HTTP
server + standalone Edge/Chrome app window by default, optional Qt WebEngine), and a PyInstaller
one-file EXE for mass distribution that ships with an **empty
personal profile** and grows via a self-learning correction loop. A separate
**typed-text/screenshot** optical model handles printed text. Hardware target:
i5-9500, 16GB RAM, **no GPU** — but the code auto-detects CUDA and serves both
tracks from one codebase.

Current measured results (CER = character error rate, lower is better):
- Personal writer model: **3.49%** on a fixed val split.
- General model (all writers): **3.67%**.
- Full-page pipeline on real scans: **~5–8%** (outliers inflated by "page
  furniture" the ground truth skips — a measurement artifact, not model error).
- User's real screenshot: general **15.9%** → dedicated typed model **10.3%**.
- Synthetic clean screenshot: typed model **0.46%**.

---

## 2. Repository map (each file, what it does)

**Core model & data**
- `charset.py` — character set (Hebrew + finals + digits + Latin + punctuation +
  `[]–`), CTC encode/decode, and a **minimal custom bidi** (`to_visual` /
  `to_logical`). The model is trained on **visual order**; output is converted
  back to logical Hebrew. New chars are appended **only at the end** to preserve
  existing checkpoints (`migrate_checkpoint.py` extends a trained model).
- `model.py` — CRNN: CNN (5 blocks, input height `IMG_H=48`, width/4 via
  `DOWNSAMPLE_W=4`) → BiLSTM (2×256) → linear → CTC. `save_checkpoint` /
  `load_checkpoint`.
- `synth.py` — synthetic line renderer. `render_line` (handwriting fonts, jitter,
  shear, blur, screenshot-downscale augment) and `render_screen_line` +
  `sample_screen_text` (screen fonts incl. bold weights, UI-Hebrew/English/mixed
  text) for the typed model. `get_fonts` / `get_screen_fonts` (the latter filters
  fonts lacking real Hebrew glyphs via a tofu check). `TextSampler` samples the
  corpus.
- `dataset.py` — `RealLines` (a writer's labeled crops), `MixedSampler`
  (real+synthetic mix at a given ratio), `ScreenSampler` (pure synthetic screen),
  `make_batch` (CTC-padded batch; **skips** samples whose image is too narrow for
  their label rather than truncating the label — a silent-corruption fix),
  `eval_batch_from_items`.

**Segmentation & inference**
- `segment.py` — page → line crops. Pipeline: `load_gray` → optional
  auto-invert (light-on-dark) → `deskew` → Otsu → `_clean_linework`
  (**progressive in-place** rule removal; a real-rule guard active only on small
  images so screenshot text isn't eaten) → ink-profile bands →
  `split_tall_bands` → `_reattach_orphan_strips` → crop + **geometry** per line.
  `segment_page_ex` returns `(PIL, meta)` where meta has `x0,x1,y0,y1,page_w,
  page_h` and optionally `em_spans` (underline-emphasis spans, see §7). Thin-band
  filtering drops underline/smear strips. `min_height` adapts to resolution
  (8px small images, 12px scans).
- `infer.py` — `transcribe_with_conf` (mean max-prob confidence; a `collect`
  hook returns the CTC path for word-position mapping), `transcribe_rescued`
  (double-crop rescue: when confidence is low and the crop is tall, split at an
  empty valley; narrow top → merge as an above-line "funnel" insertion by ink-X),
  `visual_word_spans` / `bold_word_indices` (map underline spans → logical word
  indices for bold export), `INFER_MAX_W=3200` (dense printed lines).

**Post-correction (the "language intelligence")**
- `postfix.py` — the statistical corrector. Loads (from disk caches, **not the
  raw corpus at runtime**): word `_counts`, `_vocab` (label words always; corpus
  words with freq ≥ 2), `_bigrams`, `_trigrams`. `fix_text` = conservative
  auto-fixes (single unambiguous candidate: gershayim insertion, edit-distance-1,
  swallowed-space split, cut-word join). `suggest_line` = **beam search (width 6)
  over the whole line**: each word may stay, be replaced by a candidate, or join
  the next; scored by trigram+bigram chain + capped frequency + label/page
  bonuses; every proposed change must add ≥ `min_score` to the final-context
  chain score (iterative weakest-revert), and a word known only via a prefix
  gets a protection margin. Cross-line context via `prev_tail`/`prev_tail2`
  (`advance_tail`). Page context via `page_words`. **Not applied
  automatically** — surfaced as 💡 suggestions. `suggest_line_v1` = the older
  per-word engine, kept for A/B. `page_word_counts` = document vocabulary.
- `fixes.py` — the **learned** correction model. User edits are harvested into
  (wrong→right) pairs (`harvest_pairs`, with join/split-fragment guards).
  Stored per-writer + a general file, **multiple candidates per wrong word with
  counts**; `load_merged` resolves by **majority vote**. `mark_seen` (display
  clear only — learning stays live), `remove`, `override`, `clear` (full, tests
  only). Merged into the general pool on import; shipped in the EXE.

**Training**
- `train.py` — modes: `base` (synthetic pretrain), `writer <name>`
  (per-writer fine-tune, 70% real / 30% synth), `general` (union of all writers),
  `typed` (screen model from base), `chain <name>` (writer then general),
  `rollback <name|general|base>`. **Regression gate**: measures the loaded
  model's CER on a persistent val split (`val_split.json`) before training;
  saves only on improvement; backs up the prior version to `models/history`.
  Detached process (PID in `models/train.pid`, logs `models/train_<name>.log`).
  Auto CUDA detection.

**App, packaging, data collection**
- `app.py` — stdlib `ThreadingHTTPServer` on loopback (prefers port 8765, then
  8766–8774, then an OS-assigned port), the standalone Edge/Chrome app-window GUI by default
  (optional embedded Qt WebEngine)
  (single big HTML string), and all `/api/*` endpoints.
  Live transcription jobs (threaded), model auto-selection (writer/general/typed
  with page-height auto-detect for screenshots), Word export (plain + layout-
  preserving + bold runs), the quarantine UI, training dashboard, fixes tab,
  writer add/delete, deep page-correction, upload-to-dev.
- `paths.py` — separates code from data. Frozen EXE → data in
  the executable directory with first-run seeding; user data lives in adjacent
  `data` and `models` folders so the distribution is portable. `SYNTH_CORPUS` = the lite
  corpus if present (the full 320MB corpus is **not** shipped; the LM works from
  caches).
- `main.py` — EXE entry point; dispatches `--train` for the frozen detached
  trainer.
- `build_exe.bat` / `KtavYadAI.spec` — PyInstaller one-file build. Bundles the
  Python runtime and required packages, with Qt WebEngine only for the optional
  embedded build. Runtime DLLs are explicitly sourced from Windows System32 to
  avoid stale copies discovered on PATH. Ships
  base+general+typed models, `corpus_lite.txt`, the three `.lm_*.pkl` caches,
  `fixes_general.tsv`, `config.json`. On first run it seeds writable `data/`
  and `models/` folders beside the EXE, where user training is retained.
  **No writer data, no personal model, no history/quarantine** — users start
  with an empty personal profile.
- `build_corpus.py` / `expand_corpus.py` — corpus builders from Wikisource +
  the Otzaria library (github.com/Sivan22/otzaria-library). `make_corpus_lite.py`
  — strided 156K-line sample for shipping.
- `gscript/Code.js` — Apps Script web app that receives training-data uploads to
  a Drive folder (constraint: uses `this['Drive'+'App']`, manual oauthScopes).

**Test & analysis tooling**
- `test_v3.py` (9 E2E), `test_final.py` (10 endpoint), `test_user_sim.py`
  (full user-simulation via the API, saves to a throwaway writer),
  `measure_suggest.py` (**the mandatory gate before any postfix change**: raw /
  fix / v1 / v2 / v2+page / v2+cross-line CER + beneficial/harmful counts),
  `test_screenshot.py` (persistent synthetic screenshot benchmark),
  `analyze_pages.py` (fresh-page error taxonomy), `compare10.py`, `eval12.py`.

---

## 3. Data flow (transcription)

```
image/PDF ─► segment_page_ex ─► [line crop + geometry(+em_spans)]
          ─► transcribe_rescued (CRNN→CTC→visual→logical, +conf, +CTC path)
          ─► postfix.fix_text (conservative auto)         [skipped for screen pages]
          ─► postfix.suggest_line (beam, page+cross-line)  → 💡 suggestions
          ─► bold_word_indices (em_spans × CTC word X)     → bold metadata
          ─► GUI rows ─► user edits ─► /api/save (gates) ─► training data
                                    └► /api/save_fixes ─► learned corrections
```

## 4. Data flow (training, with the safety gate)

```
labeled crops ─► load_persistent_split (fixed val, val_split.json)
             ─► MixedSampler (70% real / 30% synth)
             ─► CRNN train loop ─► every N steps: eval on val
                                   └► save ONLY if CER ≤ baseline (else keep old)
                                   └► first overwrite backs up to models/history
```

## 5. Safety model ("cannot degrade")

1. **Gate 1 (format)** — valid image, non-empty text after charset filtering.
2. **Gate 2 (contradiction)** — transcribe the sample; if CER vs the label > 65%,
   quarantine (save-corrections only; import skips this so genuinely new writing
   isn't rejected).
3. **Gate 3 (consistency)** — pixel hash: same image with a different label →
   quarantine; exact duplicate → silent skip.
4. **Gate 4 (regression)** — training saves only on val improvement; history +
   rollback. **Demonstrated live**: a weak short run left the model unchanged.

## 6. The language model (why it is statistical, not neural)

The corpus is 2.5M lines / ~336MB of rabbinic Hebrew (full Shas, Rishonim,
Acharonim incl. contemporary authors, Shulchan Aruch + nosei kelim, Mishneh
Torah, Tur, responsa, mussar). At runtime the LM is three pickled dicts —
`_counts` (305K words), `_bigrams` (1.8M), `_trigrams` (1.0M) — **decoupled from
the raw corpus file** (the EXE ships only the caches + a 21MB lite corpus for
synth). A neural LM (BEREL/DictaBERT) was **deliberately rejected**: +400–700MB
and 10–30s/page on this CPU, and HuggingFace is blocked by the user's NetFree
content filter. The trigram model gives most of the contextual benefit at ~2ms
per line. This is the natural upgrade point if a GPU becomes available.

## 7. Underline → bold (metadata-only feature)

`segment._emphasis_spans` finds hand-drawn underline strokes: short segments in
the cleaned-rule mask that are dark (hand ink) and, crucially, **do not continue
far to both sides** (a printed notebook rule does; a hand underline stops at the
word). These spans map to logical word indices via the CTC path
(`infer.bold_word_indices`), and Word export emits `<w:b/>` runs. The **text
path is untouched** — verified that raw CER is byte-identical with the feature
on. Conservative by design: a wavy underline may be partially caught; a miss is
preferred over a false positive.

## 8. External-expert review findings (applied + roadmap)

An independent research pass confirmed CRNN+CTC is still correct for CPU HTR
(the Kraken/PyLaia architecture) and that fine-tuning from the general model is
the right writer-adaptation strategy (25% relative CER gain at 16 lines, 50% at
256 — arXiv:2302.06308). **Applied fixes**: silent label truncation in
`make_batch`; regression-gate val enlarged from 64 to the full split;
`charset.clean` now normalizes typographic quotes/dashes instead of deleting
them. **Roadmap, ranked by (expected gain)/(cost)**: (1) beam decoding with the
LM **on the CTC logits** (not just the final string) — the biggest remaining win
(~12–13% relative, PyLaia); (2) stronger geometric augmentation
(elastic/grid/±2°) for the small real set; (3) better CTC confidence (drop blank
frames, use 1−p2/p1); (4) self-training on unlabeled pages behind the gates;
(5) expand the synthetic font set to 100+; (6) input height 64 (uncertain).

## 9. How to run / test / build

```bash
# runtime/build dependencies are listed in requirements.txt; pywebview[qt] is optional
python -m venv venv                       # Python 3.13 or 3.14; only needed once
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe app.py            # system browser; install pywebview[qt] for embedded mode
venv\Scripts\python.exe train.py chain <writer>     # personal then general
venv\Scripts\python.exe measure_suggest.py 8 21 33  # postfix gate (run before any postfix edit)
venv\Scripts\python.exe test_v3.py && ... test_final.py && ... test_user_sim.py 15
# build_exe.bat selects browser mode and runs --diagnose-startup, logging to data/startup.log
```

## 10. Known pitfalls (hard-won)

1. Orphan `python train.py` / stale `KtavYadAI.exe` hold CPU/ports — clean by
   CommandLine before port tests; never kill a live train.py.
2. No two trainings at once.
3. PowerShell mangles Hebrew — scripts do `sys.stdout.reconfigure(encoding=
   "utf-8")`; `wmic` is gone on this Windows (use `Get-CimInstance`).
4. Charset change = append at end + `migrate_checkpoint.py`.
5. `ingest_kuntres.py` rebuilds the lines dir from the TSV — destroys gui_/appr_
   crops.
6. `_clean_linework` must stay progressive/in-place (batch-marking slices text
   rows mid-height → gibberish).
7. NetFree blocks large POSTs, Sefaria/HuggingFace, and needs truststore for SSL.
8. The user is a Hebrew speaker distributing to a community; the UI prizes
   maximum simplicity.
