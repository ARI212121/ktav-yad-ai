# -*- coding: utf-8 -*-
r"""ניהול נתיבים מרכזי — ההפרדה בין קבצי התוכנה לקבצי הנתונים.

בהרצה רגילה (פיתוח): הנתונים בתיקיית הפרויקט.
בהרצה כ-EXE ארוז: הנתונים נשמרים בתיקיות data/ ו-models/ לצד ה-EXE,
כך שאפשר להעביר את התוכנה ואת נתוני האימון שלה יחד למחשב אחר.
"""
import os
import sys
import shutil

FROZEN = getattr(sys, "frozen", False)

# בהרצה בלי חלון קונסולה (pythonw / EXE חלוני) אין stdout — נתעל לריק
# כדי שהדפסות לא יפילו את התוכנה
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

def _user_data_dir():
    if os.name == "nt":
        root = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if root:
            return os.path.join(root, "KtavYadAI")
    elif sys.platform == "darwin":
        return os.path.expanduser("~/Library/Application Support/KtavYadAI")
    root = os.environ.get("XDG_DATA_HOME")
    return os.path.join(root, "ktav-yad-ai") if root else os.path.expanduser(
        "~/.local/share/ktav-yad-ai")


if FROZEN:
    # PyInstaller one-file extracts its read-only bundle to _MEIPASS.
    BUNDLE = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    executable_dir = os.path.dirname(os.path.abspath(sys.executable))
    # The GUI and OCRHE CLI in the portable folder share this data root.
    BASE = os.path.abspath(os.path.expanduser(
        os.environ.get("OCRHE_HOME") or executable_dir))
else:
    BUNDLE = os.path.dirname(os.path.abspath(__file__))
    SOURCE_TREE = (os.path.isdir(os.path.join(BUNDLE, "data"))
                   and os.path.isdir(os.path.join(BUNDLE, "models")))
    BASE = os.path.abspath(os.path.expanduser(
        os.environ.get("OCRHE_HOME") or
        (BUNDLE if SOURCE_TREE else _user_data_dir())))

RESOURCE_DIR = BUNDLE if FROZEN or SOURCE_TREE else os.path.join(
    sys.prefix, "share", "ktav-yad-ai")

DATA_DIR = os.path.join(BASE, "data")
MODELS_DIR = os.path.join(BASE, "models")
WRITERS_DIR = os.path.join(DATA_DIR, "writers")
TEMPLATES_DIR = os.path.join(DATA_DIR, "templates")
CORPUS = os.path.join(DATA_DIR, "corpus_he.txt")
# קורפוס לאימון הסינתטי וליצירת תבניות: המוקטן (corpus_lite) אם קיים —
# בהפצה נשלח רק הוא (21MB), הקורפוס המלא (320MB) לא נארז. מודל השפה
# עובד מהמטמונים (.lm_*.pkl) ולא נוגע בקורפוס כלל.
SYNTH_CORPUS = os.path.join(DATA_DIR, "corpus_lite.txt")
if not os.path.exists(SYNTH_CORPUS) and os.path.exists(CORPUS):
    SYNTH_CORPUS = CORPUS
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")


def _seed_once():
    """Create writable folders and seed missing packaged assets."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(WRITERS_DIR, exist_ok=True)
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    # קורפוס מוקטן (לאימון הסינתטי) + תצורה + מאגר התיקונים + שלושת מטמוני
    # מודל השפה. הקורפוס המלא לא נארז — מודל השפה עובד מהמטמונים בלבד.
    for name in ("corpus_lite.txt", "config.json", "fixes_general.tsv",
                 ".lm_counts.pkl", ".lm_bigrams.pkl", ".lm_trigrams.pkl"):
        src = os.path.join(RESOURCE_DIR, "data", name)
        dst = os.path.join(DATA_DIR, name)
        if BASE != BUNDLE and not os.path.exists(dst) and os.path.exists(src):
            shutil.copy2(src, dst)
    # Never replace an existing model: it may contain the user's training.
    src_models = os.path.join(RESOURCE_DIR, "models")
    if BASE != BUNDLE and os.path.isdir(src_models):
        for f in os.listdir(src_models):
            if not f.endswith(".pt"):
                continue
            src = os.path.join(src_models, f)
            dst = os.path.join(MODELS_DIR, f)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
    # כותבים ארוזים (נתוני אימון התחלתיים) — זריעה רק לכותבים שלא קיימים
    src_writers = os.path.join(RESOURCE_DIR, "data", "writers")
    if os.path.isdir(src_writers):
        for name in os.listdir(src_writers):
            dst = os.path.join(WRITERS_DIR, name)
            if not os.path.exists(dst):
                shutil.copytree(os.path.join(src_writers, name), dst)


_seed_once()
