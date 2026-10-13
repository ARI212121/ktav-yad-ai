# כתב-יד AI — Offline Hebrew Handwriting Recognition

מערכת מקומית (offline) להמרת **כתב יד עברי** — וגם טקסט מוקלד וצילומי מסך —
לטקסט במחשב. העיבוד והתמלול מקומיים. לחלופין, המשתמש יכול לבקש במפורש הצעות
תיקון מ-Google AI Studio; רק אז תמונות השורות וטקסט ה-OCR נשלחים ל-Google.

An offline Hebrew handwriting-recognition (HTR) system: CRNN+CTC optical model,
synthetic pretraining from Hebrew fonts, per-writer fine-tuning with
regression-gated training, a statistical (bigram+trigram) language-model
post-corrector over a 2.5M-line rabbinic Hebrew corpus, a local desktop GUI,
and a PyInstaller EXE for distribution. CPU-first (auto-detects CUDA).

## מסמכים / Docs

- **[USER_GUIDE.md](USER_GUIDE.md)** — מדריך למשתמש (עברית, לכולם).
- **[AI_HANDOFF.md](AI_HANDOFF.md)** — technical briefing for AI analysis / handoff (English).
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** — יומן שינויים מלא ומפורט (עברית).

## הורדה / Download

התוכנה המוכנה להרצה (Windows EXE) מופצת כ-ZIP (‏243MB):
- **GitHub Releases** (בריפו הזה) — נכס `KtavYadAI.zip`.
- **Google Drive** — קישור בתיאור ה-Release ואצל בעל הפרויקט.

מחלצים את ה-ZIP ומריצים `KtavYadAI.exe`. בהפעלה הראשונה התוכנה מכינה את עצמה
ונפתחת בדפדפן ברירת המחדל במחשב. התיקיות `data`
ו-`models` נוצרות לצד ה-EXE ושומרות את האימונים והמודלים, כך שאפשר להעביר אותן
יחד עם ה-EXE למחשב אחר.

גרסה **1.0.1** · יוצר: [ARI212121](https://github.com/ARI212121).

## הרצה מהמקור / Run from source

```bash
# Python 3.13/3.14 virtual environment
python -m venv venv
venv\Scripts\pip install -r requirements.txt
python app.py                     # Opens a standalone Edge/Chrome app window
python cli.py --help              # OCRHE command-line interface
python train.py chain <writer>    # fine-tune: personal then general
python measure_suggest.py 8 21 33 # postfix quality gate (run before any postfix change)
python test_v3.py ; python test_final.py ; python test_user_sim.py 15
```

## OCRHE command line

Install the cross-platform command with `python -m pip install .`; it is named
`OCRHE` (command names are case-insensitive on Windows). The CLI can also run
directly from source with `python cli.py`. Every command and subcommand has
`--help`. For machine-readable output, use `--json`, or `ocr --format json|jsonl`.
Results go to stdout; progress and errors go to stderr. Destructive changes
require the explicit `--yes` flag.

```text
OCRHE --help
OCRHE ocr "scan.pdf" --page --model auto --format json
OCRHE ocr "scan.png" --page --data-dir "D:\OCRHE-data" --format json
OCRHE writers list --json
OCRHE examples list --writer "אישי" --json
OCRHE examples edit --writer "אישי" --file "gui_123.png" --text "תיוג נכון"
OCRHE examples delete --writer "אישי" --file "gui_123.png" --yes
OCRHE quarantine list --writer "אישי" --json
OCRHE train start --writer "אישי" --scope both
OCRHE train status --json
OCRHE learn "D:\scans\page.png" --writer "שמואל" --accuracy 95
OCRHE models evaluate --name typed --json
OCRHE writers purge-data --name "אישי" --yes
OCRHE fixes list --writer "אישי" --json
OCRHE templates generate --text-file sentences.txt
OCRHE templates ingest --input scan.png --template page_001 --writer "אישי"
OCRHE writers export --name "אישי" --output training-pack.zip
OCRHE models history --json
OCRHE ai keys add --name "מפתח עבודה" --key-env OCRHE_GEMINI_API_KEY
OCRHE ai keys rename --id KEY_ID --name "מפתח חדש"
OCRHE ai settings get --json
OCRHE ai settings set --batch-size 3 --temperature 0.2 --top-p 0.9
OCRHE ocr "scan.png" --page --format json --include-images -o ocr.json
OCRHE ai correct --input ocr.json --lines 2,4-6 --batch-size 3 --key-name "מפתח עבודה" --output suggestions.json
```

The OCRHE command shares the application's OCR, sample-safety gates, corrections,
and training data. Saved labels can be reviewed and changed in the application's
training tab or by CLI; changing/deleting a label does not rewrite an already
trained model's weights. Retrain or restore a previous model after correcting
training examples. OCR output is not added to training automatically: review it
first, then explicitly use `examples add` or `parallel --save` if it is correct.
Use `--data-dir` to point a command at an isolated copy of `data` and `models`
when testing mutations. `models evaluate` reports character error rate without
changing model weights or creating a persistent validation split.

`OCRHE learn <image>` is the guided image-to-training workflow: provide the
approved transcript with `--text-file` (UTF-8, one line per handwritten line),
`--text`, or the interactive prompt, select a writer, and set the character
accuracy threshold with `--accuracy`. The command checks that detected image
lines match transcript lines, saves examples through the normal safety gates,
and trains only that writer's personal model until whole-page OCR reaches the
chosen threshold or `--max-hours` expires (up to 400). Progress reports the
current training-step percentage, elapsed time-budget percentage, and latest
whole-page character accuracy. It resumes a partial checkpoint when restarted;
completed rounds start from the best saved personal model. Defaults are 500
steps and batch size 2; adjust with `--steps` and `--batch`. Accuracy is
character-level edit accuracy after Unicode and whitespace normalization, with
line breaks preserved; it is not the model's training-set CER.

For the verified five-line Samuel page, run
`python train_samuel_until_exact.py --data-dir .\portable --max-hours 400 --steps-per-round 100 --batch-size 2`.
The script trains only Samuel's personal model, checks the page after every
round, and stops early when all five lines match exactly or at the time limit.
It can resume an interrupted round; the current best model remains protected
by the normal validation gate. A long run is useful only while the machine
remains available, and CPU training may take a long time. Set `--data-dir` to
train the copy used by a different installation.

When the GUI and `OCRHE.exe` are in the same portable folder, both read and
update the same `data` and `models` directories. API keys, their names and
selected models, and saved AI request settings are shared with the GUI. Add a
key through a hidden terminal prompt, or through an environment variable as
shown above; the secret is never accepted as a command-line argument.

For line-level AI correction, create OCR JSON with `--page --include-images`,
then choose original 1-based line numbers with `--lines` (for example
`2,4-6`). JSONL OCR results also include line numbers and optional images.
Page numbers in JSON and JSONL output are 1-based.
Corrections are suggestions only; the input file and training labels are not
changed. The API key can be selected by saved name or ID, or left automatic
for normal key rotation. Request settings can be saved for both interfaces with
`ai settings set`; per-request flags override them. Supported controls are batch
size (1–16), temperature (0–2), top-p (0–1), top-k (1–1000), maximum output
tokens (1–65536), and request timeout (1–300 seconds). JSON response mode and
the correction schema remain fixed so responses can be safely mapped back to
their original line numbers.

Commands that contact external services are explicit: `ai check`, `ai models`,
`ai test`, and `ai correct` call Google AI Studio; `writers share` uploads a
writer's training pack. Do not use them for offline testing, and review the
data before confirming an upload with `--yes`.

The editable Python install works on supported Python platforms. PyInstaller
builds are native to the OS/architecture where they run; invoke the platform's
native build process on Windows, macOS, or Linux (the Windows `build_exe.bat`
builds both Windows executables).

## בניית EXE נייד / Build portable EXE

On Windows with Python 3.13 or 3.14 installed, run `build_exe.bat`. It creates
the project `venv` if needed and installs/verifies packages automatically.
The final multi-resolution application icon is included as `icon.ico` and used
directly by the build. The GUI embeds its application window and does not
require an installed browser. The build also creates `portable/KtavYadAI.exe`
and `portable/OCRHE.exe` in a folder beside the official `release` directory.
Those portable executables share `portable/data` and `portable/models`; existing
release data/models are copied there on first portable build without replacing
an existing portable data directory. The Python console command can also be
installed with `python -m pip install .`. Each OS/architecture must be built on
that OS/architecture. The embedded window remembers its last size. The GUI EXE
includes its application icon and version metadata (1.0.1, creator ARI212121).
On first launch it creates data/model folders beside the portable executables;
keep them together when moving the app or preserving training. Startup timings and any
startup errors are recorded in `data/startup.log`; the post-build diagnostic also
records total EXE launch time, including one-file extraction before Python
starts. Every build runs a startup
diagnostic (including PyTorch, the general model, the local server, and icon);
the build is reported as failed if that check does not pass.

Optional AI correction uses Google AI Studio only after an explicit button
click. Choose a number of lines or pages; page selection sends only the
segmented line crops, never the full-page image. Requests are split into
configurable batches of 1, 2, 4, or 8 lines. Add one or more named API keys in the
training tab; Windows DPAPI encrypts them for the current Windows account. If a
key reaches its quota, the app tries the next configured key. Suggested text
is shown as an editable draft and is never saved to training data until the user
applies/reviews it and presses the save button. Each run displays elapsed time
and token usage reported by Google per batch; remaining quota is not returned
by the correction API. Repeating a run to compare batch sizes resends the images
and consumes quota again.

The key manager can explicitly check one key or refresh all connection checks.
This lightweight GET validates access to the configured model without sending
images or generating content; it does not verify remaining generation quota.
Keys belonging to the same Google Cloud project share that project's quota.
For each key, use “טען מודלים” to explicitly fetch its available
`generateContent` models and choose one. The selection is saved per key and is
used for connection checks and when that key is selected during failover; model
discovery is not performed automatically at startup.

## מבנה / Layout

מודל: `model.py` `charset.py` · סינתזה: `synth.py` · נתונים: `dataset.py` ·
פילוח: `segment.py` · אינפרנס: `infer.py` · תיקון-אחרי: `postfix.py` `fixes.py` ·
אימון: `train.py` · אפליקציה: `app.py` · נתיבים/אריזה: `paths.py` `main.py`
`build_exe.bat` · builds a bundled GUI EXE and a separate `OCRHE.exe` CLI into
the portable sibling folder. They share `data/` and `models/`, use an embedded
application window, and do not require an external browser. If the preferred
local port is busy, the app automatically uses an available port. · קורפוס:
`build_corpus.py` `expand_corpus.py` · בוט שיתוף:
`gscript_share/`.

## תוצאות (CER — אחוז שגיאת תווים, נמוך = טוב)

| מודל | CER |
|---|---|
| אישי (כותב יחיד) | **3.49%** |
| כללי (כל הכותבים) | **3.67%** |
| מוקלד / צילום מסך | ‏0.46% סינתטי · ‏10.3% צילום מסך אמיתי |
| עמוד כתב-יד מלא (צינור מלא) | ‏~5–8% |

> נתוני האימון האישיים (כתב היד, הקונטרס) **אינם** בריפו — ראו `.gitignore`.
> ההפצה יוצאת עם פרופיל אישי **ריק**; המשתמש בונה את המודל שלו בעצמו.
