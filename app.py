# -*- coding: utf-8 -*-
"""תמלול כתב יד — תוכנה מקומית מלאה (גרסה 3).

הרצה:  start_app.bat  (או python app.py) — נפתח בחלון אפליקציה ייעודי.

עיקרי היכולות:
- תמלול חי של הרבה קבצים יחד (תמונות + PDF) עם התקדמות שורה-שורה
- עריכה רציפה כמו וורד (חצים/אנטר בין שורות) צמוד לתמונות המקור
- ייצוא: TXT / Word / העתקה — בשלושה מצבים (כמו במקור / רציף / שורה אחרי נקודה)
- אימון "משולב": כל אימון מעדכן גם את המודל האישי וגם את הכללי,
  רץ כתהליך עצמאי שממשיך גם כשהתוכנה סגורה, וניתן לעצירה מהתוכנה
- תבניות איסוף, טקסט מקביל, ייצוא/ייבוא חבילות אימון ומודלים
"""
import os
import io
import re
import json
import base64
import ctypes
import importlib
import logging
import shutil
import zipfile
import threading
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote
from xml.sax.saxutils import escape as xml_escape

import torch
from PIL import Image

from paths import BUNDLE, DATA_DIR, MODELS_DIR as MODELS, UPLOADS_DIR, FROZEN
from charset import CHARSET, NUM_CLASSES, clean, decode_ctc, to_logical
from model import load_checkpoint
from segment import (segment_page, segment_page_ex, WRITERS_DIR, TEMPLATES_DIR,
                     ingest_template)
from infer import (pil_to_array, transcribe_with_conf, transcribe_rescued,
                   bold_word_indices)
import ai_correction
import postfix
import gates
import fixes
import records


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HERE = os.path.dirname(os.path.abspath(__file__))
VERSION = "1.0.1"
WINDOW_SIZE_PATH = os.path.join(DATA_DIR, "window_size.json")
DEFAULT_WINDOW_SIZE = (1220, 880)
PORT = 8765
PID_FILE = os.path.join(MODELS, "train.pid")
TRAIN_STOP_FILE = os.path.join(MODELS, "train.stop")

_state = {"models": {}}
_jobs = {}
_jobs_lock = threading.Lock()


# ---------------------------------------------------------------- מודלים
def get_model(writer=None, prefer=None):
    """בוחר מודל: אישי של הכותב / כללי / מוקלד / בסיס.
    prefer='general'/'typed' כופה; אחרת אישי עם נפילה לכללי ולבסיס."""
    candidates = []
    if prefer == "typed":
        candidates.append(("__typed__", os.path.join(MODELS, "typed.pt")))
    if prefer == "general":
        candidates.append(("__general__", os.path.join(MODELS, "general.pt")))
    if writer and prefer not in ("general", "typed"):
        candidates.append((writer, os.path.join(MODELS, f"writer_{writer}.pt")))
    candidates.append(("__general__", os.path.join(MODELS, "general.pt")))
    candidates.append(("__base__", os.path.join(MODELS, "base_synth.pt")))
    for key, path in candidates:
        if os.path.exists(path):
            mtime = os.path.getmtime(path)
            cached = _state["models"].get(key)
            if cached and cached[1] == mtime:
                return cached[0], key
            m, _ = load_checkpoint(path, NUM_CLASSES, CHARSET)
            m.to(DEVICE)
            m.eval()
            _state["models"][key] = (m, mtime)
            return m, key
    return None, None


def img_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- כותבים
def _parse_cers(log_path):
    cers = []
    try:
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                m = re.search(r"CER [^:]*: ([\d.]+)%", line)
                if m:
                    cers.append(float(m.group(1)))
    except OSError:
        pass
    return cers


_meta_cache = {}


def _model_val_cer(path):
    """ה-CER המאומת של קובץ המודל עצמו (מתוך המטא של הצ'קפוינט) —
    מקור האמת לטבלה, בשונה מהגרף שמציג את ריצת האימון החיה."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    cached = _meta_cache.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    val = None
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        v = (ck.get("meta") or {}).get("val_cer")
        if v:
            val = round(float(v) * 100, 2)
    except Exception:
        val = None
    _meta_cache[path] = (mtime, val)
    return val


def list_writers():
    if not os.path.isdir(WRITERS_DIR):
        return []
    out = []
    for name in sorted(os.listdir(WRITERS_DIR)):
        if not os.path.isdir(os.path.join(WRITERS_DIR, name)):
            continue
        labels = os.path.join(WRITERS_DIR, name, "labels.tsv")
        n = 0
        if os.path.exists(labels):
            with open(labels, encoding="utf-8") as f:
                n = sum(1 for l in f if l.strip())
        model_path = os.path.join(MODELS, f"writer_{name}.pt")
        best = _model_val_cer(model_path)
        if best is None:
            cers = _parse_cers(os.path.join(MODELS, f"train_{name}.log"))
            if cers:
                best = min(cers)
        out.append({"name": name, "samples": n,
                    "trained": os.path.exists(model_path),
                    "best_cer": best})
    return out


def save_lines(writer, lines):
    """שמירת שורות לאימון דרך שערי ההזנה הבטוחה (gates.py).
    מחזיר (נשמרו, עברו להסגר, כפילויות שדולגו)."""
    wdir = os.path.join(WRITERS_DIR, writer)
    ldir = os.path.join(wdir, "lines")
    os.makedirs(ldir, exist_ok=True)
    pairs = []
    for item in lines:
        try:
            img = Image.open(io.BytesIO(base64.b64decode(item["img"]))).convert("L")
        except Exception:
            img = None
        pairs.append((img, item.get("text", "")))
    # שער 2 נבדק מול המודל הטוב ביותר שקיים לכותב (אישי/כללי/בסיס)
    model, _ = get_model(writer)
    predict = None
    if model is not None:
        def predict(pil):
            text, _ = transcribe_with_conf(model, pil_to_array(pil))
            return text
    accepted, quarantined, dups = gates.check_lines(writer, pairs, predict)
    labels = os.path.join(wdir, "labels.tsv")
    existing = 0
    if os.path.exists(labels):
        with open(labels, encoding="utf-8") as f:
            existing = sum(1 for _ in f)
    saved = 0
    with open(labels, "a", encoding="utf-8") as f:
        for img, text in accepted:
            fname = f"gui_{int(time.time())}_{existing + saved:05d}.png"
            img.save(os.path.join(ldir, fname))
            f.write(f"{fname}\t{text}\n")
            saved += 1
    gates.quarantine_add(writer, quarantined)
    return saved, len(quarantined), dups


# ---------------------------------------------------------------- אימון
def _pid_alive(pid):
    if os.name != "nt":
        try:
            os.kill(int(pid), 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
    SYNCHRONIZE = 0x00100000
    h = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, int(pid))
    if not h:
        return False
    STILL_ACTIVE = 259
    code = ctypes.c_ulong()
    PQI = 0x1000  # PROCESS_QUERY_LIMITED_INFORMATION
    ctypes.windll.kernel32.CloseHandle(h)
    h2 = ctypes.windll.kernel32.OpenProcess(PQI, False, int(pid))
    if not h2:
        return True  # קיים אך אין הרשאה לשאול — נניח שרץ
    ok = ctypes.windll.kernel32.GetExitCodeProcess(h2, ctypes.byref(code))
    ctypes.windll.kernel32.CloseHandle(h2)
    return bool(ok) and code.value == STILL_ACTIVE


def _train_pid():
    try:
        with open(PID_FILE) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def training_running():
    pid = _train_pid()
    if pid is None:
        return False
    if _pid_alive(pid):
        return True
    try:
        os.remove(PID_FILE)
    except OSError:
        pass
    return False


def start_training(writer, scope="both", resume=True):
    """מפעיל אימון עצמאי לפי היקף: both (אישי+כללי) / personal / general.
    התהליך מנותק מהתוכנה וממשיך גם אם סוגרים אותה."""
    if training_running():
        return False, "אימון כבר רץ ברקע — אפשר לעצור אותו בכפתור העצירה"
    logs = {"both": [f"train_{writer}.log", "train_general.log"],
            "personal": [f"train_{writer}.log"],
            "general": ["train_general.log"]}.get(scope, [])
    for f in logs:
        open(os.path.join(MODELS, f), "w", encoding="utf-8").close()
    try:
        os.remove(TRAIN_STOP_FILE)
    except OSError:
        pass
    flags = (subprocess.DETACHED_PROCESS |
             subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0
    run_mode = "--resume" if resume else "--fresh"
    if FROZEN:
        cmd = [sys.executable, "--train", writer or "-", run_mode]
        if scope == "personal":
            cmd.append("--personal-only")
        elif scope == "general":
            cmd.append("--general-only")
    else:
        if scope == "general":
            cmd = [sys.executable, os.path.join(HERE, "train.py"), "general", run_mode]
        elif scope == "personal":
            cmd = [sys.executable, os.path.join(HERE, "train.py"), "writer", writer, run_mode]
        else:
            cmd = [sys.executable, os.path.join(HERE, "train.py"), "chain", writer, run_mode]
    process_kwargs = {"creationflags": flags} if os.name == "nt" else {
        "start_new_session": True}
    process = subprocess.Popen(
        cmd, cwd=HERE, close_fds=True, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, **process_kwargs)
    with open(PID_FILE, "w", encoding="ascii") as f:
        f.write(str(process.pid))
    what = {"both": "אישי + כללי", "personal": "אישי בלבד",
            "general": "הכללי בלבד"}.get(scope, scope)
    return True, f"האימון החל ({what}). רץ ברקע — אפשר לסגור את התוכנה."


# ---------------------------------------------------------------- תצורה
CONFIG_PATH = None  # מוגדר למטה אחרי ייבוא הנתיבים


def load_config():
    from paths import DATA_DIR
    p = os.path.join(DATA_DIR, "config.json")
    try:
        # utf-8-sig — עמיד גם לקבצים שנשמרו עם BOM (עריכה ידנית/PowerShell)
        with open(p, encoding="utf-8-sig") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_config(cfg):
    from paths import DATA_DIR
    p = os.path.join(DATA_DIR, "config.json")
    cur = load_config()
    cur.update({k: v for k, v in cfg.items() if v is not None})
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cur, f, ensure_ascii=False, indent=1)
    return cur


def upload_pack_to_dev(writer):
    """שולח את חבילת האימון של הכותב לשרת ההעלאה (Apps Script) של המפתח."""
    import urllib.request
    url = load_config().get("upload_url", "").strip()
    if not url.startswith("https://"):
        return False, "לא הוגדרה כתובת שרת העלאה (בהגדרות למטה)"
    data = export_writer_zip(writer, for_upload=True)
    if not data:
        return False, "אין נתונים לכותב זה"
    payload = json.dumps({"writer": writer,
                          "data": base64.b64encode(data).decode()}).encode()
    req = urllib.request.Request(url, data=payload,
                                 headers={"Content-Type": "application/json"})
    import ssl
    ctx = None
    try:
        # מאגר התעודות של Windows — עובד גם עם אנטי-וירוס שמצוטט HTTPS
        import truststore
        ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        pass
    if ctx is None:
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=180, context=ctx) as r:
            resp = json.loads(r.read().decode())
        if resp.get("ok"):
            return True, "הנתונים נשלחו בהצלחה — תודה רבה על התרומה! 💝"
        return False, "השרת דחה את ההעלאה — נסה שוב מאוחר יותר"
    except Exception as e:
        return False, f"שגיאת שליחה: {e}"


def delete_training_data(writer):
    """מוחק את קובצי האימון של הכותב לפינוי מקום בדיסק. נמחקים: תמונות
    השורות, התיוגים, חלוקת האימות, אינדקס הכפילויות וההסגר. נשארים:
    המודלים המאומנים ומודל התיקונים — התמלול ממשיך לעבוד כרגיל."""
    if training_running():
        return False, "אימון רץ כעת — המתן לסיומו (או עצור אותו) לפני מחיקה"
    wdir = os.path.join(WRITERS_DIR, writer)
    if not os.path.isdir(wdir):
        return False, "כותב לא נמצא"
    import shutil
    targets = [os.path.join(wdir, "lines"), os.path.join(wdir, "quarantine"),
               os.path.join(wdir, "labels.tsv"),
               os.path.join(wdir, "val_split.json"),
               os.path.join(wdir, ".hashes.json")]
    freed = 0
    for t in targets:
        if os.path.isdir(t):
            freed += sum(os.path.getsize(os.path.join(r, f))
                         for r, _, fs in os.walk(t) for f in fs)
            shutil.rmtree(t, ignore_errors=True)
        elif os.path.exists(t):
            freed += os.path.getsize(t)
            try:
                os.remove(t)
            except OSError:
                pass
    os.makedirs(os.path.join(wdir, "lines"), exist_ok=True)
    postfix.refresh_vocab()
    return True, f'קובצי האימון של "{writer}" נמחקו — התפנו {freed / 1e6:.1f}MB'


def delete_writer(writer):
    """מחיקה מלאה של כותב מהמערכת: נתוני האימון + המודל האישי + גרסאות
    הארכיון והיומנים שלו. בלתי הפיך — הממשק מציע גיבוי (חבילת ייצוא) קודם."""
    if training_running():
        return False, "אימון רץ כעת — המתן לסיומו (או עצור אותו) לפני מחיקה"
    wdir = os.path.join(WRITERS_DIR, writer)
    if not os.path.isdir(wdir):
        return False, "כותב לא נמצא"
    import shutil
    shutil.rmtree(wdir, ignore_errors=True)
    removed = ["נתוני האימון"]
    model_f = os.path.join(MODELS, f"writer_{writer}.pt")
    for f in (model_f, model_f + ".last", model_f + ".last.previous",
              os.path.join(MODELS, f"{writer}.meta.json"),
              os.path.join(MODELS, f"train_{writer}.log")):
        try:
            os.remove(f)
            if f == model_f:
                removed.append("המודל האישי")
        except OSError:
            pass
    hist = os.path.join(MODELS, "history")
    if os.path.isdir(hist):
        for f in os.listdir(hist):
            if f.startswith(f"{writer}__"):
                try:
                    os.remove(os.path.join(hist, f))
                except OSError:
                    pass
    _state["models"].pop(writer, None)
    postfix.refresh_vocab()
    return True, f'הכותב "{writer}" נמחק ({" + ".join(removed)})'


def stop_training():
    pid = _train_pid()
    if not pid or not _pid_alive(pid):
        return False
    with open(TRAIN_STOP_FILE, "w", encoding="ascii") as f:
        f.write("stop")
    return True


def _latest_log():
    best, best_m = None, 0
    if os.path.isdir(MODELS):
        for f in os.listdir(MODELS):
            if f.startswith("train_") and f.endswith(".log") and f != "train_base.log":
                m = os.path.getmtime(os.path.join(MODELS, f))
                if m > best_m:
                    best, best_m = f, m
    return best, best_m


def train_status():
    running = training_running()
    log_f, mtime = _latest_log()
    if not log_f:
        return {"running": running}
    name = log_f[len("train_"):-len(".log")]
    label = ("המודל הכללי (כל הכותבים)" if name == "general" else
             "מודל הטקסט המוקלד/צילומי מסך" if name == "typed" else
             f"מודל אישי: {name}")
    phase = "2/2" if name == "general" else "1/2"
    step, total, loss, tail, cers = 0, 0, None, "", []
    with open(os.path.join(MODELS, log_f), encoding="utf-8") as f:
        lines_all = f.readlines()
    tail = "".join(lines_all[-10:])
    for line in lines_all:
        m = re.search(r"צעד (\d+)/(\d+) \| loss=([\d.]+)", line)
        if m:
            step, total, loss = int(m.group(1)), int(m.group(2)), float(m.group(3))
        m = re.search(r"CER [^:]*: ([\d.]+)%", line)
        if m:
            cers.append(float(m.group(1)))
    gen_cers = _parse_cers(os.path.join(MODELS, "train_general.log"))
    return {"running": running, "label": label, "phase": phase, "step": step,
            "total": total, "loss": loss, "cers": cers, "log": tail,
            "general_best": min(gen_cers) if gen_cers else None}


# ---------------------------------------------------------------- תמלול חי
def _decode_inputs(files):
    """[{name, data}] -> [(תווית, תמונת PIL)] — תמונות ו-PDF."""
    pages = []
    for f in files:
        raw = base64.b64decode(f["data"])
        name = f.get("name", "קובץ")
        if name.lower().endswith(".pdf"):
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(raw)
            for i in range(len(pdf)):
                pil = pdf[i].render(scale=300 / 72).to_pil().convert("L")
                pages.append((f"{name} — עמוד {i + 1}", pil))
            pdf.close()
        else:
            pages.append((name, Image.open(io.BytesIO(raw)).convert("L")))
    return pages


def _run_transcribe_job(job_id, files, page_mode, writer, prefer, use_fix):
    job = _jobs[job_id]
    try:
        model, key = get_model(writer, prefer)
        if model is None:
            job.update(error="אין אף מודל מאומן עדיין", done=True)
            return
        job["model"] = key
        pages = _decode_inputs(files)
        job["pages_total"] = len(pages)
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        # הקשר עמוד/רב-עמודי: אוצר המילים של כל השורות שתומללו עד כה בעבודה —
        # מועמד שחוזר במקום אחר בעמוד מקבל חיזוק במנוע ההצעות
        page_counts = postfix.page_word_counts([])
        for p_idx, (label, pil) in enumerate(pages):
            job["cur"] = label
            if page_mode:
                tmp = os.path.join(UPLOADS_DIR, f"_job{job_id}.png")
                pil.save(tmp)
                try:
                    items = segment_page_ex(tmp)
                finally:
                    try:
                        os.remove(tmp)
                    except FileNotFoundError:
                        pass
            else:
                items = [(pil, None)]
            imgs = [im for im, _ in items]
            # שורה נמוכה משמעותית מהחציון = כנראה הערת שוליים/כותרת קטנה
            heights = sorted(im.height for im in imgs)
            med_h = heights[len(heights) // 2] if heights else 1
            # זיהוי עמוד מוקלד/צילום מסך: שורות נמוכות מאוד (טקסט מסך 10-30px;
            # סריקת כתב יד מנורמלת ל-2600 נותנת שורות 40px+). כשקיים מודל
            # מוקלד ומצב auto — עוברים אליו לעמוד הזה בלבד.
            page_model, page_key = model, key
            if prefer in (None, "", "auto") and page_mode and med_h < 34:
                tm, tk = get_model(None, "typed")
                if tk == "__typed__":
                    page_model, page_key = tm, tk
            with _jobs_lock:
                job["model"] = page_key
                if page_key not in job["models_used"]:
                    job["models_used"].append(page_key)
            for k, (im, geom) in enumerate(items):
                path_info = {} if geom and geom.get("em_spans") else None
                rescued = transcribe_rescued(page_model, im, med_h,
                                             collect=path_info)
                for im2, text, conf in rescued:
                    # מילים עם קו-הדגשה מתחתן -> כתב מודגש (Bold) בייצוא
                    bold = []
                    if path_info and path_info.get("pred") and text.strip():
                        try:
                            bold = bold_word_indices(
                                path_info["pred"], path_info["arr_w"],
                                im.width, geom["em_spans"],
                                len(text.split()))
                        except Exception:
                            bold = []
                    sugg = None
                    is_typed_page = page_key == "__typed__"
                    n_words_raw = len(text.split())
                    if use_fix and not is_typed_page:
                        # בעמוד-מסך (מודל typed) שכבת ההצעות מדולגת: נמדד
                        # שה-LM התלמודי מזיק לעברית מודרנית של ממשקים
                        # (התוכנה->התכונה, גלול->גדול; 10.3%->11.1%)
                        text = postfix.fix_text(text, writer=writer)
                        if bold and len(text.split()) != n_words_raw:
                            bold = []   # פיצול/איחוי הזיז אינדקסים — מוותרים
                        # הצעת תיקון הקשרית (מודל שפה + תיקונים שנלמדו) —
                        # לא מוחלת אוטומטית, מוצגת למשתמש כ-💡 לאישור בלחיצה
                        s_text, s_changes = postfix.suggest_line(
                            text, writer=writer, page_words=page_counts)
                        if s_changes:
                            sugg = s_text
                        page_counts += postfix.page_word_counts([text])
                    is_note = len(items) > 3 and im2.height < med_h * 0.62
                    with _jobs_lock:
                        job["lines"].append({"img": img_to_b64(im2), "text": text,
                                             "src": label, "page": p_idx,
                                             "ai_crop": page_mode,
                                             "conf": round(conf, 3),
                                             "note": is_note, "geom": geom,
                                             "sugg": sugg, "bold": bold,
                                             "typed": is_typed_page})
                with _jobs_lock:
                    # התקדמות אמיתית ברמת שורות: עמודים שהושלמו + חלק העמוד הנוכחי
                    job["progress"] = (p_idx + (k + 1) / max(1, len(items))) / len(pages)
            job["pages_done"] += 1
        if use_fix:
            # מעבר שני: רענון ההצעות עם אוצר המילים המלא של כל העבודה
            # (בלי מילות השורה עצמה — שגיאה עקבית לא תחזק את עצמה) + רצף
            # ההקשר חוצה-השורות. ההבדלים נשלחים כ"עדכוני 💡" לשורות שהוצגו.
            suggs, pt2, pt, prev_src = {}, None, None, None
            for idx, l in enumerate(job["lines"]):
                if l.get("src") != prev_src:
                    pt2, pt, prev_src = None, None, l.get("src")
                if l.get("typed"):
                    continue    # עמוד-מסך: שכבת ההצעות מדולגת (נמדד מזיק)
                ctx = page_counts - postfix.page_word_counts([l["text"]])
                s_text, s_changes = postfix.suggest_line(
                    l["text"], writer=writer, page_words=ctx,
                    prev_tail=pt, prev_tail2=pt2)
                new_sugg = s_text if s_changes else None
                if new_sugg != l["sugg"]:
                    l["sugg"] = new_sugg
                    suggs[idx] = new_sugg
                pt2, pt = postfix.advance_tail(pt2, pt, l["text"])
            with _jobs_lock:
                job["suggs"] = suggs
        job["progress"] = 1.0
        job["done"] = True
    except Exception as e:
        job.update(error=str(e), done=True)


def start_transcribe(files, page_mode, writer, prefer, use_fix=True):
    job_id = str(int(time.time() * 1000))
    _jobs[job_id] = {"pages_total": 0, "pages_done": 0, "cur": "", "lines": [],
                     "done": False, "error": None, "model": None,
                     "models_used": [], "progress": 0.0}
    threading.Thread(target=_run_transcribe_job,
                     args=(job_id, files, page_mode, writer, prefer, use_fix),
                     daemon=True).start()
    return job_id


# ---------------------------------------------------------------- Word
def build_docx(paragraphs, title="תמלול כתב יד"):
    """יוצר קובץ Word מעוצב (RTL, גופן David, כותרת, הערות מוטות) — ללא תלויות."""
    import datetime
    date_str = datetime.date.today().strftime("%d/%m/%Y")

    def run_props(sz, bold=False, italic=False, color=None):
        p = f'<w:rFonts w:ascii="David" w:hAnsi="David" w:cs="David"/>'
        if bold:
            p += "<w:b/><w:bCs/>"
        if italic:
            p += "<w:i/><w:iCs/>"
        if color:
            p += f'<w:color w:val="{color}"/>'
        return (f'<w:rPr>{p}<w:rtl/><w:sz w:val="{sz}"/>'
                f'<w:szCs w:val="{sz}"/></w:rPr>')

    parts = []
    # כותרת ראשית + תאריך
    parts.append('<w:p><w:pPr><w:bidi/><w:jc w:val="center"/>'
                 '<w:spacing w:after="60"/></w:pPr>'
                 f'<w:r>{run_props(40, bold=True)}'
                 f'<w:t xml:space="preserve">{xml_escape(title)}</w:t></w:r></w:p>')
    parts.append('<w:p><w:pPr><w:bidi/><w:jc w:val="center"/>'
                 '<w:spacing w:after="240"/></w:pPr>'
                 f'<w:r>{run_props(20, color="777777")}'
                 f'<w:t xml:space="preserve">נוצר בתוכנת כתב-יד AI · {date_str}</w:t></w:r></w:p>')
    def runs_with_bold(text, bold_idx, sz):
        """ריצות עם מודגש: מילים שזוהה קו-הדגשה מתחתן בכתב היד -> <w:b/>."""
        words = text.split(" ")
        bold_set = set(bold_idx or [])
        segs, cur, cur_b = [], [], None
        for i, w in enumerate(words):
            b = i in bold_set
            if cur_b is None or b == cur_b:
                cur.append(w)
                cur_b = b
            else:
                segs.append((cur_b, " ".join(cur)))
                cur, cur_b = [w], b
        if cur:
            segs.append((cur_b, " ".join(cur)))
        out = []
        for j, (b, seg) in enumerate(segs):
            seg_t = seg + (" " if j < len(segs) - 1 else "")
            out.append(f'<w:r>{run_props(sz, bold=b)}'
                       f'<w:t xml:space="preserve">{xml_escape(seg_t)}</w:t></w:r>')
        return "".join(out)

    for p in paragraphs:
        bold_idx = []
        if isinstance(p, dict):
            bold_idx = p.get("bold") or []
            p = p.get("text") or ""
        is_note = p.strip().startswith("(הערה")
        if is_note:
            parts.append('<w:p><w:pPr><w:bidi/><w:ind w:right="567"/>'
                         '<w:spacing w:after="80"/></w:pPr>'
                         f'<w:r>{run_props(22, italic=True, color="666666")}'
                         f'<w:t xml:space="preserve">{xml_escape(p)}</w:t></w:r></w:p>')
        else:
            parts.append('<w:p><w:pPr><w:bidi/><w:jc w:val="both"/>'
                         '<w:spacing w:after="120" w:line="320" w:lineRule="auto"/></w:pPr>'
                         + runs_with_bold(p, bold_idx, 26) + '</w:p>')
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f'<w:body>{"".join(parts)}'
        '<w:sectPr><w:bidi/><w:pgMar w:top="1134" w:right="1134" w:bottom="1134" '
        'w:left="1134" w:header="708" w:footer="708" w:gutter="0"/></w:sectPr>'
        '</w:body></w:document>')
    ct = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
          '<Default Extension="xml" ContentType="application/xml"/>'
          '<Override PartName="/word/document.xml" ContentType='
          '"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/'
            'officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ct)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", document)
    return buf.getvalue()


def _docx_pack(document_xml):
    """עוטף document.xml מוכן לקובץ docx תקני."""
    ct = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
          '<Default Extension="xml" ContentType="application/xml"/>'
          '<Override PartName="/word/document.xml" ContentType='
          '"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/'
            'officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ct)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", document_xml)
    return buf.getvalue()


def build_docx_layout(lines, title="תמלול כתב יד"):
    """ייצוא Word שמשחזר את פריסת הדף המקורי מהגיאומטריה של השורות:
    הזחות (שורה שלא מתחילה בשוליים הימניים), מרכוז (כותרות/מספרי עמוד),
    הערות צד (שורות צרות בשוליים), ורווחי פסקאות (מרווח אנכי חריג)."""

    def run_props(sz, bold=False, italic=False, color=None):
        p = '<w:rFonts w:ascii="David" w:hAnsi="David" w:cs="David"/>'
        if bold:
            p += "<w:b/><w:bCs/>"
        if italic:
            p += "<w:i/><w:iCs/>"
        if color:
            p += f'<w:color w:val="{color}"/>'
        return (f'<w:rPr>{p}<w:rtl/><w:sz w:val="{sz}"/>'
                f'<w:szCs w:val="{sz}"/></w:rPr>')

    USABLE_TWIPS = 9638  # רוחב טקסט של A4 עם שוליים של 2 ס"מ
    # חציון גובה שורה ורוחב מלא — הבסיס לזיהוי חריגות
    geoms = [l.get("geom") for l in lines if l.get("geom")]
    heights = sorted(g["y1"] - g["y0"] for g in geoms) or [1]
    med_h = heights[len(heights) // 2]
    parts = []
    prev_y1 = None
    for l in lines:
        text = (l.get("text") or "").strip()
        if not text:
            continue
        g = l.get("geom")
        ppr = "<w:bidi/>"
        sz, italic, color = 26, False, None
        spacing_before = 0
        if g:
            pw = max(1, g["page_w"])
            width_r = (g["x1"] - g["x0"]) / pw
            right_off = (pw - g["x1"]) / pw          # RTL: ההתחלה מימין
            center_off = abs((g["x0"] + g["x1"]) / 2 - pw / 2) / pw
            if prev_y1 is not None and g["y0"] - prev_y1 > 1.8 * med_h:
                spacing_before = 240                  # פסקה חדשה (רווח אנכי גדול)
            prev_y1 = g["y1"]
            if width_r < 0.55 and center_off < 0.10:
                ppr += '<w:jc w:val="center"/>'       # כותרת/מספר עמוד ממורכזים
                sz = 24
            elif width_r < 0.35 and (right_off > 0.25 or g["x0"] / pw > 0.55):
                # שורה צרה הרחק מהשוליים — הערת צד
                indent = int(min(0.5, right_off) * USABLE_TWIPS)
                ppr += f'<w:ind w:right="{indent}"/>'
                sz, italic, color = 20, True, "555555"
            elif right_off > 0.07:
                ppr += f'<w:ind w:right="{int(right_off * USABLE_TWIPS)}"/>'
        if l.get("note") and not italic:
            sz, italic, color = 22, True, "666666"
        ppr += f'<w:spacing w:before="{spacing_before}" w:after="60" ' \
               'w:line="300" w:lineRule="auto"/>'
        bold_set = set(l.get("bold") or [])
        if bold_set and not italic:
            # מילים עם קו-הדגשה בכתב היד -> ריצות מודגשות
            words = text.split(" ")
            runs, cur, cur_b = [], [], None
            for i, wd in enumerate(words):
                b = i in bold_set
                if cur_b is None or b == cur_b:
                    cur.append(wd)
                    cur_b = b
                else:
                    runs.append((cur_b, " ".join(cur)))
                    cur, cur_b = [wd], b
            if cur:
                runs.append((cur_b, " ".join(cur)))
            body = "".join(
                f'<w:r>{run_props(sz, bold=b, color=color)}'
                f'<w:t xml:space="preserve">'
                f'{xml_escape(seg + (" " if j < len(runs) - 1 else ""))}'
                f'</w:t></w:r>' for j, (b, seg) in enumerate(runs))
            parts.append(f'<w:p><w:pPr>{ppr}</w:pPr>{body}</w:p>')
        else:
            parts.append(f'<w:p><w:pPr>{ppr}</w:pPr>'
                         f'<w:r>{run_props(sz, italic=italic, color=color)}'
                         f'<w:t xml:space="preserve">{xml_escape(text)}</w:t></w:r></w:p>')
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f'<w:body>{"".join(parts)}'
        '<w:sectPr><w:bidi/><w:pgMar w:top="1134" w:right="1134" w:bottom="1134" '
        'w:left="1134" w:header="708" w:footer="708" w:gutter="0"/></w:sectPr>'
        '</w:body></w:document>')
    return _docx_pack(document)


# ---------------------------------------------------------------- ייצוא/ייבוא
def export_writer_zip(name, for_upload=False):
    """חבילת אימון. for_upload: גרסה קלה לשליחה — תמונות דחוסות ובלי קובץ מודל
    (מגבלת גודל של שרת ההעלאה)."""
    wdir = os.path.join(WRITERS_DIR, name)
    if not os.path.isdir(wdir):
        return None
    labels_map = {}
    labels = os.path.join(wdir, "labels.tsv")
    if os.path.exists(labels):
        with open(labels, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    labels_map[parts[0]] = parts[1]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        out_labels = []
        ldir = os.path.join(wdir, "lines")
        if os.path.isdir(ldir):
            for f in sorted(os.listdir(ldir)):
                # שורות שיובאו מאחרים (קידומת imp) לא מיוצאות שוב —
                # כל חבילה מכילה רק את הנתונים המקוריים של הכותב עצמו
                if f.startswith("imp") or f not in labels_map:
                    continue
                src = os.path.join(ldir, f)
                if for_upload:
                    img = Image.open(src).convert("L")
                    jb = io.BytesIO()
                    img.save(jb, "JPEG", quality=72)
                    new_name = os.path.splitext(f)[0] + ".jpg"
                    z.writestr(f"lines/{new_name}", jb.getvalue())
                    out_labels.append(f"{new_name}\t{labels_map[f]}")
                else:
                    z.write(src, f"lines/{f}")
                    out_labels.append(f"{f}\t{labels_map[f]}")
        z.writestr("labels.tsv", "\n".join(out_labels) + "\n")
        # קובץ מודל התיקונים של הכותב מצורף תמיד — כולל בשליחה למפתח,
        # כך שהמאגר הכללי לומד גם מתיקוני המשתמשים
        fx = fixes.export_tsv(name)
        if fx:
            z.writestr("fixes.tsv", fx)
        if not for_upload:
            model = os.path.join(MODELS, f"writer_{name}.pt")
            if os.path.exists(model):
                z.write(model, "model.pt")
        z.writestr("info.txt", f"writer={name}\ncharset_len={len(CHARSET)}\n")
    return buf.getvalue()


def import_writer_zip(zip_b64, writer):
    wdir = os.path.join(WRITERS_DIR, writer)
    ldir = os.path.join(wdir, "lines")
    os.makedirs(ldir, exist_ok=True)
    z = zipfile.ZipFile(io.BytesIO(base64.b64decode(zip_b64)))
    names = z.namelist()
    if "labels.tsv" not in names:
        return 0, "בחבילה אין labels.tsv — זו אינה חבילת אימון תקינה"
    by_base = {os.path.basename(n): n for n in names
               if n.startswith("lines/") and
               n.lower().endswith((".png", ".jpg", ".jpeg"))}
    pairs = []
    for line in z.read("labels.tsv").decode("utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0] in by_base and parts[1].strip():
            try:
                img = Image.open(io.BytesIO(z.read(by_base[parts[0]]))).convert("L")
            except Exception:
                img = None
            pairs.append((img, parts[1].strip()))
    # בייבוא נבדקים שערים 1+3 בלבד — שער 2 יפסול בטעות כתב יד חדש שהמודל טרם למד
    accepted, quarantined, dups = gates.check_lines(writer, pairs, None)
    prefix = f"imp{int(time.time())}_"
    added = 0
    with open(os.path.join(wdir, "labels.tsv"), "a", encoding="utf-8") as out:
        for img, text in accepted:
            fname = f"{prefix}{added:05d}.png"
            img.save(os.path.join(ldir, fname))
            out.write(f"{fname}\t{text}\n")
            added += 1
    gates.quarantine_add(writer, quarantined)
    n_fix = 0
    if "fixes.tsv" in names:
        # תיקונים מיובאים מתמזגים למאגר הכללי בלבד (לא לאישי של אף אחד)
        n_fix = fixes.import_tsv(z.read("fixes.tsv").decode("utf-8", "replace"))
    extra = (f" · {len(quarantined)} בהסגר" if quarantined else "") + \
            (f" · {dups} כפילויות דולגו" if dups else "") + \
            (f" · {n_fix} תיקונים מוזגו למאגר הכללי" if n_fix else "")
    return added, (f"נוספו {added} שורות מתויגות לכותב '{writer}'{extra}"
                   " — מומלץ להריץ אימון עכשיו")


def import_model_file(pt_b64, name):
    safe = re.sub(r"[^\w\-אבגדהוזחטיכךלמםנןסעפףצץקרשת]", "", name) or "imported"
    tmp = os.path.join(MODELS, "_import_check.pt")
    with open(tmp, "wb") as f:
        f.write(base64.b64decode(pt_b64))
    try:
        load_checkpoint(tmp, NUM_CLASSES, CHARSET)
    except Exception as e:
        os.remove(tmp)
        return None, f"המודל אינו תואם: {e}"
    os.replace(tmp, os.path.join(MODELS, f"writer_{safe}.pt"))
    _state["models"].pop(safe, None)
    return safe, f"המודל נשמר כמודל של הכותב '{safe}'"


# ---------------------------------------------------------------- תבניות
def templates_zip_bytes():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        if os.path.isdir(TEMPLATES_DIR):
            for f in sorted(os.listdir(TEMPLATES_DIR)):
                if f.endswith(".png"):
                    z.write(os.path.join(TEMPLATES_DIR, f), f)
    return buf.getvalue()


def list_templates():
    if not os.path.isdir(TEMPLATES_DIR):
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(TEMPLATES_DIR)
                  if f.endswith(".png"))


# ---------------------------------------------------------------- HTML
HTML = r"""<!DOCTYPE html>
<html dir="rtl" lang="he"><head><meta charset="utf-8">
<link rel="icon" href="/favicon.ico">
<title>כתב-יד AI 1.0.1 — תמלול אישי</title>
<style>
 :root{--navy:#141e3c;--blue:#1d4e89;--teal:#0e7c7b;--green:#17805a;--bg:#eef1f6;--card:#fff;--line:#e3e7ef}
 *{box-sizing:border-box}
 body{font-family:'Segoe UI',Arial;margin:0;background:var(--bg);color:#182135}
 header{background:linear-gradient(105deg,#141e3c 0%,#1d4e89 70%,#0e7c7b 100%);color:#fff;padding:16px 28px;display:flex;gap:18px;align-items:center;flex-wrap:wrap}
 header h1{font-size:21px;margin:0} header .sub{font-size:13px;opacity:.85}
 header select{padding:7px 10px;border-radius:8px;border:0;font-size:15px}
 nav{display:flex;gap:8px;background:transparent;padding:14px 24px 0}
 nav button{background:#fff;border:1px solid var(--line);padding:10px 22px;font-size:15.5px;cursor:pointer;border-radius:24px;transition:.15s}
 nav button:hover{border-color:var(--blue)}
 nav button.active{background:var(--blue);color:#fff;border-color:var(--blue);font-weight:600}
 main{max-width:1200px;margin:16px auto;padding:0 20px}
 .card{background:var(--card);border-radius:14px;padding:20px 22px;margin-bottom:16px;border:1px solid var(--line);box-shadow:0 2px 8px rgba(20,30,60,.05)}
 .card b.title{font-size:17px}
 button.btn{background:var(--blue);color:#fff;border:0;border-radius:9px;padding:10px 20px;font-size:15px;cursor:pointer;box-shadow:0 1px 3px rgba(0,0,0,.15);transition:.15s}
 button.btn:hover{filter:brightness(1.12);transform:translateY(-1px)}
 button.btn:disabled{background:#a9b2c3;cursor:default;transform:none}
 button.green{background:var(--green)} button.red{background:#b03636} button.teal{background:var(--teal)}
 select,input[type=text],input[type=number]{padding:9px;border-radius:8px;border:1px solid #c8cfdd;font-size:15px}
 textarea{width:100%;min-height:130px;border-radius:8px;border:1px solid #c8cfdd;font-size:16px;padding:10px}
 .drop{border:2px dashed #b9c3d8;border-radius:12px;padding:22px;text-align:center;color:#5a6residual覆}
 .drop{color:#56617a;background:#f7f9fd;cursor:pointer;transition:.15s}
 .drop.over{border-color:var(--teal);background:#eefaf9}
 .filelist{font-size:13.5px;color:#385;margin-top:6px}
 .srcHead{background:#f0f4fb;border-radius:8px;padding:5px 12px;margin:10px 0 3px;font-weight:600;color:var(--blue);font-size:13.5px}
 #lines,#parLines{max-height:70vh;overflow-y:auto;border:1px solid var(--line);border-radius:10px;padding:2px 6px;margin-top:8px}
 .line{display:flex;gap:8px;align-items:flex-start;padding:1px 3px;border-radius:5px}
 .line:nth-child(even){background:#f8fafd}
 .line.focused{background:#eef6ff;outline:1.5px solid #bcd7f7}
 .line img{max-width:40%;max-height:42px;border:1px solid var(--line);border-radius:3px;background:#fff;object-fit:contain;margin-top:2px;cursor:zoom-in}
 .line img.zoom{max-width:75%;max-height:140px;cursor:zoom-out}
 .line .num{min-width:20px;text-align:center;color:#98a1b5;font-size:10.5px;margin-top:6px}
 .line .aiRowSelect{display:none;margin-top:7px;flex:0 0 auto}
 body.aiManualSelection .line .aiRowSelect{display:inline-block}
 .line textarea.lt{flex:1;font-size:14.5px;line-height:1.4;direction:rtl;border:0;background:transparent;
  padding:3px 6px;border-bottom:1px solid transparent;resize:none;overflow:hidden;min-height:24px;font-family:inherit}
 .line textarea.lt:focus{outline:none;border-bottom:2px solid var(--teal)}
 .line textarea.lt.lowconf{background:#fff3f0;border-radius:4px}
 .noteTag{background:#fdf1d7;color:#8a6100;border-radius:10px;padding:1px 8px;font-size:11.5px;white-space:nowrap}
 .drop.disabled{opacity:.5;pointer-events:none}
 #log{background:#10141f;color:#7fe3a1;font-family:consolas,monospace;font-size:13px;padding:12px;border-radius:8px;white-space:pre-wrap;direction:ltr;text-align:left;max-height:190px;overflow-y:auto}
 .muted{color:#66708a;font-size:14px} .big{font-size:21px;font-weight:700}
 .bar{background:#e2e6ee;border-radius:10px;height:24px;overflow:hidden}
 .bar>div{background:linear-gradient(90deg,#17805a,#2fb37e);height:100%;color:#fff;font-size:13px;text-align:center;line-height:24px;transition:width .5s;white-space:nowrap}
 table{border-collapse:collapse;width:100%} td,th{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;font-size:14.5px}
 .grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
 .pill{display:inline-block;background:#e8f4f1;color:var(--green);border-radius:14px;padding:3px 12px;font-size:13px;font-weight:600}
 .pill.gray{background:#eceff5;color:#66708a}
 .exportBar{display:flex;gap:14px;align-items:center;flex-wrap:wrap;background:#f7f9fd;border-radius:10px;padding:12px 14px;margin-top:12px}
 .exportBar label{font-size:14px}
 #magBar{display:none;position:sticky;top:0;background:#fff;z-index:5;border:1px solid var(--line);border-radius:8px;padding:4px;text-align:center;margin-bottom:4px}
 #magBar img{max-width:100%;max-height:120px;object-fit:contain}
 .mini{border:0;background:#eceff5;border-radius:6px;cursor:pointer;font-size:12px;padding:2px 7px;margin-top:5px;color:#44506b}
 .mini:hover{background:#d6dcea}
 .mini.sugg{background:#fdf3d0;font-size:14px}
 .mini.sugg:hover{background:#f8e6a0}
 #appDialog{position:fixed;inset:0;z-index:10000;display:none;align-items:center;justify-content:center;padding:20px;background:rgba(12,20,39,.52)}
 #appDialog.open{display:flex}
 #appDialogPanel{width:min(540px,100%);max-height:min(80vh,680px);overflow:auto;background:#fff;border:1px solid #dbe1ec;border-radius:16px;box-shadow:0 18px 60px rgba(10,20,40,.28);padding:24px}
 #appDialogTitle{margin:0 0 12px;color:var(--navy);font-size:19px}
 #appDialogMessage{margin:0;white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.6;color:#344058}
 #appDialogInput{display:none;width:100%;margin-top:16px}
 #appDialogActions{display:flex;gap:10px;justify-content:flex-start;margin-top:22px}
 #appDialogActions button{min-width:92px}
 @media(max-width:600px){#appDialogPanel{padding:18px}#appDialogActions button{flex:1}}
 @media(max-width:820px){.grid2{grid-template-columns:1fr}.line img{max-width:230px}}
</style></head><body>
<header>
 <div><h1>🖋️ כתב-יד AI</h1><div class="sub">תמלול כתב יד אישי — הכול מקומי, שום דבר לא יוצא מהמחשב</div></div>
 <span style="margin-right:auto">כותב:</span><select id="writer" onchange="refreshAll()"></select>
 <select id="modelPick" title="באיזה מודל להשתמש בתמלול">
  <option value="auto">אוטומטי (מומלץ)</option>
  <option value="personal">המודל האישי של הכותב</option>
  <option value="general">המודל הכללי</option>
  <option value="typed">טקסט מוקלד / צילום מסך</option>
 </select>
  <span id="modelUsed" class="pill gray" title="המודל או המודלים שנבחרו בפועל לעבודה האחרונה"></span>
  <span id="writerInfo" class="pill gray"></span>
</header>
<nav>
 <button class="active" onclick="showTab('tabScan',this)">📄 תמלול</button>
 <button onclick="showTab('tabTrain',this)">🎓 אימון</button>
 <button onclick="showTab('tabTmpl',this)">📋 תבניות</button>
 <button onclick="showTab('tabPar',this)">📚 טקסט מקביל</button>
 <button onclick="showTab('tabFix',this)">🧠 תיקונים</button>
</nav>
<main>

<div id="tabScan">
 <div class="card">
  <b class="title">העלאת קבצים לתמלול</b>
  <div class="muted" style="margin:4px 0 10px">אפשר לבחור הרבה קבצים יחד — תמונות (PNG/JPG) וגם PDF. מומלץ סריקה 300dpi בגווני אפור.</div>
  <div class="drop" id="drop" onclick="document.getElementById('file').click()">
   📁 גרור לכאן קבצים או לחץ לבחירה
   <input type="file" id="file" accept="image/*,.pdf" multiple style="display:none" onchange="fileChosen()">
   <div class="filelist" id="fileList"></div>
  </div>
  <div style="margin-top:12px">
   <label><input type="checkbox" id="pageMode" checked> עמודים מלאים (פילוח אוטומטי לשורות)</label>
   <label><input type="checkbox" id="fixMode" checked> תיקון מילוני אוטומטי</label>
   <button class="btn" onclick="transcribe()" id="btnGo">🚀 תמלל הכול</button>
  </div>
 </div>
 <div class="card" id="scanProg" style="display:none">
  <b class="title">מתמלל...</b> <span id="scanCount" class="muted"></span>
  <div class="bar" style="margin-top:10px"><div id="scanBar" style="width:0%">0%</div></div>
  <div class="muted" style="margin-top:6px">השורות מופיעות למטה ברגע שהן מוכנות — אפשר לערוך ולהעתיק תוך כדי.</div>
 </div>
 <div class="card" id="resultsCard" style="display:none">
  <b class="title">תוצאות</b> <span class="muted">— עריכה רציפה: אנטר/חצים עוברים בין השורות · ✖ מוחק שורת סרק · ⤴ ממזג לשורה הקודמת</span>
  <div id="magBar"><img id="magImg" alt=""></div>
  <div id="lines"></div>
  <div id="aiReview" class="card" style="display:none;margin-top:12px">
   <b class="title">הצעות תיקון מ-Google AI Studio — טיוטה לבדיקה</b>
   <p class="muted">התמונות והטקסט שזוהה נשלחו ל-Google רק בעקבות לחיצה על „הצע תיקון AI”.
    ההצעות אינן נשמרות או מאומנות אוטומטית. בדוק כל שינוי, החל רק את המתאימים,
    ערוך לפי הצורך, ורק אז שמור לאימון.</p>
    <button class="btn green" onclick="applyAllAICorrections()">החל את כל ההצעות לטיוטה</button>
    <div id="aiReviewList"></div>
   <span id="aiReviewMsg" class="muted"></span>
  </div>
  <div class="exportBar">
   <b>ייצוא:</b>
   <label><input type="radio" name="expMode" value="orig" checked> שורות כמו במקור</label>
   <label><input type="radio" name="expMode" value="flow"> טקסט רציף (בלי ירידות שורה)</label>
   <label><input type="radio" name="expMode" value="dots"> ירידת שורה רק אחרי נקודה</label>
   <button class="btn" style="background:#6a3fa0" onclick="deepSuggest()"
    title="ניתוח כל המסמך במקביל: אוצר המילים של כל העמודים + רצף ההקשר חוצה-השורות">🧠 תיקון מורחב (ניתוח כל המסמך)</button>
   <label>היקף AI:
    <select id="aiScope" onchange="updateAISelectionControls()">
     <option value="lines">שורות ראשונות</option>
     <option value="pages">עמודים ראשונים (כל חיתוכי השורות)</option>
     <option value="manual">בחירה ידנית של שורות</option>
    </select>
   </label>
   <label id="aiAmountControl">כמות:
    <input id="aiAmount" type="number" value="8" min="1" step="1"
     style="width:76px" onchange="updateAISelectionControls()">
   </label>
   <span id="aiManualControls" style="display:none">
    <button class="mini" type="button" onclick="setAISelection(true)">בחר הכול</button>
    <button class="mini" type="button" onclick="invertAISelection()">הפוך בחירה</button>
    <button class="mini" type="button" onclick="setAISelection(false)">נקה בחירה</button>
   </span>
   <label>מספרי שורות:
    <input id="aiLineNumbers" type="text" placeholder="למשל 1,3-5"
     style="width:110px">
   </label>
   <button class="mini" type="button" onclick="selectAILinesByNumber()">סמן לפי מספרים</button>
   <label>מפתח API:
    <select id="aiKeyChoice" title="בחר מפתח שמור או השתמש בסבב האוטומטי">
     <option value="">בחירה אוטומטית</option>
    </select>
   </label>
   <label title="מספר חיתוכי השורות בכל בקשת רשת; יותר שורות בבקשה מפחית בקשות נפרדות">
    שורות לבקשה:
    <input id="aiBatchSize" type="number" min="1" max="16" value="8"
     style="width:70px" onchange="saveAISettings()">
   </label>
   <details style="display:inline-block">
    <summary style="cursor:pointer">פרמטרים מתקדמים</summary>
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:8px">
     <label>Temperature
      <input id="aiTemperature" type="number" min="0" max="2" step="0.1"
       value="0" style="width:72px" onchange="saveAISettings()">
     </label>
     <label>Top-P
      <input id="aiTopP" type="number" min="0" max="1" step="0.01"
       style="width:72px" onchange="saveAISettings()">
     </label>
     <label>Top-K
      <input id="aiTopK" type="number" min="1" max="1000" step="1"
       style="width:72px" onchange="saveAISettings()">
     </label>
     <label>טוקנים מרביים
      <input id="aiMaxOutputTokens" type="number" min="1" max="65536" step="1"
       style="width:92px" onchange="saveAISettings()">
     </label>
     <label>פסק זמן (שניות)
      <input id="aiRequestTimeout" type="number" min="1" max="300" step="1"
       value="90" style="width:76px" onchange="saveAISettings()">
     </label>
     <button class="mini" type="button" onclick="resetAISettings()">אפס לברירות מחדל</button>
    </div>
   </details>
   <button class="btn" style="background:#7352a3" id="aiCorrectBtn" onclick="aiCorrect()">✨ הצע תיקון AI</button>
   <span id="aiSelectionInfo" class="muted"></span>
   <button class="btn" style="background:#b8860b" onclick="applyAllSuggs()">💡 החל את כל ההצעות</button>
   <button class="btn" onclick="copyText()">📋 העתק</button>
   <button class="btn" onclick="downloadTxt()">⬇️ TXT</button>
   <button class="btn teal" onclick="downloadDocx()">⬇️ Word</button>
   <button class="btn teal" onclick="downloadDocxLayout()" title="משחזר הזחות, כותרות ממורכזות והערות צד מהדף המקורי">⬇️ Word בפריסת המקור</button>
   <button class="btn green" onclick="saveAll('lines','saveMsg')">💾 שמור תיקונים לאימון</button>
   <button class="btn teal" onclick="saveFixes()">🧠 שמור למודל התיקונים</button>
   <select id="fixScope" title="להיכן נלמדים התיקונים">
    <option value="both">אישי + כללי</option>
    <option value="personal">אישי בלבד</option>
    <option value="general">כללי בלבד</option>
   </select>
   <span id="saveMsg" class="muted"></span>
  </div>
 </div>
</div>

<div id="tabTrain" style="display:none">
 <div class="card">
  <b class="title">✨ תיקון תמלול בעזרת Google AI Studio</b>
  <p class="muted">הוסף מפתחות API מ-Google AI Studio; לכל מפתח אפשר לתת שם.
   אם מפתח מגיע למכסה שלו, התוכנה תנסה מפתח אחר שהוספת.
   המפתחות נשמרים מוצפנים לחשבון Windows הנוכחי ואינם נכללים בחבילות האימון.
   אפשר לבחור מספר שורות או עמודים; חיתוכי השורות נשלחים בבקשות קטנות ברצף.</p>
  <p class="muted">מפתחות ששייכים לאותו פרויקט חולקים את אותה מכסה; הוספת מפתחות
   מאותו פרויקט אינה מגדילה אותה. בזמן תיקון, שגיאת מכסה או הרשאה תעביר למפתח הבא;
   שגיאת חיבור תוצג ולא תגרום לניסיונות מיותרים עם כל המפתחות.</p>
  <p class="muted"><a href="https://aistudio.google.com/rate-limit?timeRange=last-28-days"
   target="_blank" rel="noreferrer">צפה במכסות ובשימוש הפעיל ב-Google AI Studio</a>
   — יתרת המכסה אינה זמינה בתשובת ה-API.</p>
  <p class="muted"><b>פרטיות:</b> תמונות השורות והטקסט שזוהה נשלחים ל-Google רק כשלוחצים
   על „✨ הצע תיקון AI”. אין שליחה אוטומטית בזמן תמלול, שמירה או אימון.</p>
  <div>
   <input type="text" id="aiKeyName" maxlength="60" placeholder="שם למפתח, למשל אישי">
   <input type="password" id="aiKeyValue" autocomplete="new-password" placeholder="מפתח API של Google AI Studio">
   <button class="btn" onclick="addAIKey()">➕ שמור מפתח</button>
   <span id="aiKeyMsg" class="muted"></span>
  </div>
  <button class="btn" id="aiKeysCheckAll" onclick="checkAllAIKeys()">🔄 בדוק/רענן חיבור לכל המפתחות</button>
  <span class="muted">בדיקת החיבור אינה שולחת תמונות ואינה בודקת מכסת יצירה; היא מאמתת גישה למודל.</span>
  <div id="aiKeyList" class="muted" style="margin-top:10px"></div>
 </div>
 <div class="grid2">
  <div class="card"><b class="title">מצב האימון</b>
   <div id="trainIdle" class="muted" style="margin-top:8px">אין אימון פעיל.</div>
   <div id="trainLive" style="display:none">
    <div style="margin:8px 0"><span id="trainLabel" class="big"></span>
     <span id="trainPhase" class="pill"></span>
     <span id="trainBg" class="pill">רץ ברקע — אפשר לסגור את התוכנה</span></div>
    <div class="bar"><div id="trainBar" style="width:0%">0%</div></div>
    <div class="muted" style="margin-top:6px">loss: <span id="trainLoss">—</span></div>
   </div>
   <div style="margin-top:14px">
    <select id="trainScope">
     <option value="both">אימון מלא: אישי + כללי (מומלץ)</option>
     <option value="personal">רק המודל האישי</option>
     <option value="general">רק המודל הכללי (הזרמת הנתונים הקיימים אליו)</option>
    </select>
    <button class="btn green" onclick="train(undefined,true)">&#1492;&#1502;&#1513;&#1498; &#1488;&#1497;&#1502;&#1493;&#1503; / &#1488;&#1502;&#1503; &#1495;&#1491;&#1513;</button>
    <button class="btn" onclick="train(undefined,false)" title="Start over from the base/general model; the interrupted checkpoint is kept as a backup">&#1488;&#1514;&#1495;&#1500; &#1488;&#1497;&#1502;&#1493;&#1503; &#1502;&#1495;&#1491;&#1513;</button>
    <button class="btn red" onclick="stopTrain()">⏹ עצור אימון</button>
   </div>
   <div class="muted" style="margin-top:8px">מומלץ אימון מלא: המודל הכללי לומד מכל הכותבים יחד,
   ולעיתים הוא דווקא מתמלל טוב יותר — בדוק את שניהם בתפריט שלמעלה. האפשרות "רק הכללי" שימושית
   אחרי עדכון מודל כללי חדש, כדי להזרים אליו את נתוני האימון שכבר קיימים אצלך.</div>
  </div>
  <div class="card"><b class="title" id="cerTitle">גרף דיוק (CER)</b>
   <div id="cerChart" style="margin-top:10px" class="muted">אין נתונים עדיין</div>
   <div class="muted">אחוז שגיאת התווים — ככל שנמוך יותר, המודל מדויק יותר.
   הגרף מציג את האימון הפעיל/האחרון; "שגיאה מיטבית" בטבלה למטה שייכת למודל האישי.</div>
   <div id="genBest" class="muted" style="margin-top:6px"></div>
  </div>
 </div>
 <div class="card"><b class="title">הכותבים במערכת</b>
  <table id="writersTable"><tr><th>שם</th><th>שורות מתויגות</th><th>מודל אישי</th><th>שגיאה מיטבית</th></tr></table>
  <div style="margin-top:10px">
   <input type="text" id="newWriter" placeholder="שם כותב חדש">
   <button class="btn" onclick="addWriter()">➕ הוסף כותב</button>
   <button class="btn red" style="margin-right:18px" onclick="deleteWriter()"
    title="מוחק את הכותב הנבחר למעלה: נתוני האימון + המודל האישי. בלתי הפיך!">🗑 מחק את הכותב הנבחר</button>
   <span id="delWriterMsg" class="muted"></span>
  </div>
 </div>
 <div class="card">
  <b class="title">🗂️ היסטוריית דוגמאות האימון — צפייה, עריכה ומחיקה</b>
  <p class="muted">כאן אפשר לתקן תיוג שנשמר לא נכון או להסיר דוגמה. השינוי מתעדכן
   בקובצי האימון ובאינדקס הכפילויות; הוא לא מוחק ידע שכבר נכנס למשקולות המודל.
   לאחר תיקון נתונים, הפעל אימון מחדש או שחזר גרסת מודל קודמת.</p>
  <div><button class="btn" onclick="refreshExamples()">רענן דוגמאות</button>
   <span id="examplesSummary" class="muted"></span></div>
  <div id="examplesList"></div>
  <div style="display:flex;gap:8px;align-items:center;margin-top:10px">
   <button class="btn" id="examplesPrev" onclick="pageExamples(-1)">הקודם</button>
   <span id="examplesPage" class="muted"></span>
   <button class="btn" id="examplesNext" onclick="pageExamples(1)">הבא</button>
  </div>
 </div>
 <div class="card" id="quarCard" style="display:none"><b class="title">🛃 הסגר — שורות שממתינות לאישור</b>
  <p class="muted">שורות שנעצרו באחד משערי הבטיחות (תיוג חשוד, סותר או כפול). אשר רק אם הטקסט
  באמת תואם לתמונה — אפשר לתקן את הטקסט לפני האישור. שורה שנדחית נמחקת ולא נכנסת לאימון.</p>
  <div id="quarLines"></div>
 </div>
 <div class="card"><b class="title">🌐 שיתוף ושיפור מרכזי</b>
  <p class="muted">הפץ את התוכנה, קבל מאחרים "חבילות אימון", ייבא אותן כאן ואמן — המודל הכללי משתפר מכל כתב יד שנוסף.</p>
  <div class="grid2">
   <div><b>ייצוא</b><br><br>
    <button class="btn" onclick="exportPack()">📦 חבילת אימון של הכותב הפעיל</button><br><br>
    <button class="btn" onclick="location.href='/api/export_model?which=general'">⬇️ המודל הכללי (.pt)</button>
    <button class="btn" onclick="location.href='/api/export_model?which=base'">⬇️ מודל הבסיס (.pt)</button>
   </div>
   <div><b>ייבוא</b><br><br>
    חבילת אימון: <input type="file" id="packFile" accept=".zip">
    <input type="text" id="packWriter" placeholder="שם הכותב שיקבל">
    <button class="btn green" onclick="importPack()">📥 ייבא</button>
    <div id="packMsg" class="muted"></div><br>
    <b>עדכון המודל הכללי</b> <span class="muted">(מקובץ עדכון של מפתח התוכנה — לא נוגע במודל האישי שלך)</span><br>
    <input type="file" id="generalFile" accept=".pt">
    <button class="btn teal" onclick="importGeneral()">🔄 עדכן מודל כללי</button>
    <div id="generalMsg" class="muted"></div><br>
    מודל ככותב (.pt): <input type="file" id="modelFile" accept=".pt">
    <input type="text" id="modelName" placeholder="שם לכותב">
    <button class="btn" onclick="importModel()">📥 ייבא מודל</button>
    <div id="modelMsg" class="muted"></div>
   </div>
  </div>
 </div>
 <div class="card"><b class="title">💝 תרומת נתונים למפתח התוכנה</b>
  <p class="muted">שליחת נתוני האימון שלך (רק מה שאתה יצרת — תמונות שורות + התיקונים שלך)
  עוזרת מאוד לשפר את המודל הכללי עבור <b>כל</b> המשתמשים. תודה על כל תרומה!</p>
  <button class="btn green" onclick="uploadPack()">🚀 שלח את נתוני האימון שלי למפתח</button>
  <span id="uploadMsg" class="muted"></span>
  <details style="margin-top:8px"><summary class="muted">הגדרות שרת העלאה (למפתח)</summary>
   <input type="text" id="uploadUrl" placeholder="כתובת סקריפט ההעלאה (https://script.google.com/...)" style="width:70%">
   <button class="btn" onclick="saveConfig()">שמור</button>
  </details>
 </div>
 <div class="card"><b class="title">🧹 פינוי מקום — מחיקת קובצי האימון</b>
  <p class="muted">מוחק את תמונות השורות והתיוגים של הכותב הפעיל. <b>לא מומלץ</b> —
  הקבצים משמשים גיבוי ויידרשו לאימון מחדש אחרי עדכון מודל עתידי. המודלים המאומנים
  ומודל התיקונים נשארים, והתמלול ממשיך לעבוד כרגיל.</p>
  <button class="btn red" onclick="deleteTrainData()">🧹 מחק את קבצי האימון</button>
  <span id="cleanMsg" class="muted"></span>
 </div>
 <div class="card"><b class="title">יומן אימון</b><div id="log">—</div></div>
</div>

<div id="tabTmpl" style="display:none">
 <div class="grid2">
  <div class="card"><b class="title">דפי איסוף מובנים</b>
   <p class="muted">דפים עם משפטים מוכנים. מדפיסים ← ממלאים בכתב יד ← סורקים ← תיוג אוטומטי.</p>
   <button class="btn" onclick="makeTemplates()">🖨️ צור 10 דפים חדשים</button>
   <button class="btn green" onclick="location.href='/api/templates/zip'">⬇️ הורד הכול (ZIP)</button>
   <div id="tmplMsg" class="muted" style="margin-top:8px"></div>
  </div>
  <div class="card"><b class="title">תבנית מתוכן אישי</b>
   <p class="muted">הקלד משפטים משלך (משפט בשורה) — המערכת תיצור מהם דפי איסוף:</p>
   <textarea id="customText" placeholder="משפט ראשון&#10;משפט שני..."></textarea>
   <button class="btn" onclick="makeCustom()">צור דפים מהתוכן שלי</button>
   <div id="customMsg" class="muted" style="margin-top:8px"></div>
  </div>
 </div>
 <div class="card"><b class="title">קליטת תבנית סרוקה</b>
  <p class="muted">העלה סריקה של דף תבנית מלא — השורות ייכנסו לאימון עם תיוג אוטומטי.</p>
  <input type="file" id="tmplFile" accept="image/*">
  <select id="tmplName"></select>
  <button class="btn green" onclick="ingestTemplate()">קלוט לכותב הפעיל</button>
  <div id="ingestMsg" class="muted" style="margin-top:8px"></div>
 </div>
</div>

<div id="tabFix" style="display:none">
 <div class="card"><b class="title">🧠 מודל התיקונים הלומד</b>
  <p class="muted">המערכת לומדת מהתיקונים שלך: כשאתה מתקן שורות ולוחץ
  "שמור למודל התיקונים" (בלשונית התמלול), ההבדלים נלמדים כזוגות שגוי→נכון.
  בתמלולים הבאים, טעות שכבר תיקנת בעבר תוצע מיד (💡). ההצעות מבוססות גם על
  מודל שפה תלמודי שנבנה מגמרות, ראשונים, אחרונים וקונטרסים בני-זמננו.</p>
  <p class="muted">כשאותה מילה תוקנה בדרכים שונות — <b>הרוב קובע</b>: תיקון
  שנשמר פעמיים גובר על תיקון חד-פעמי, כך שטעות בודדת לא משבשת את המודל.
  הטבלה מציגה את מה שנלמד מאז הניקוי האחרון; ✏️ מתקן צורה שגויה שנלמדה,
  ‏🗑 מסיר אותה מהמודל לגמרי.</p>
  <div style="margin:8px 0"><span class="pill">פעילים: <span id="fixCntA">0</span></span>
   <span class="pill">אישי: <span id="fixCntP">0</span></span>
   <span class="pill gray">כללי: <span id="fixCntG">0</span></span>
   <button class="btn" style="margin-right:14px" onclick="clearFixes()"
    title="מנקה רק את הטבלה שלמטה — הלמידה עצמה נשארת פעילה במלואה">🧹 נקה את הרשימה (התיקונים נשארים פעילים)</button>
   <span id="fixClearMsg" class="muted"></span></div>
  <table id="fixTable"><tr><th>שגוי</th><th>נכון</th><th>פעמים</th><th></th></tr></table>
 </div>
 <div class="card"><b class="title">שיתוף מודל התיקונים</b>
  <p class="muted">קובץ התיקונים שלך מצורף אוטומטית לחבילת האימון ולשליחה
  למפתח. אפשר גם לייצא/לייבא ידנית — ייבוא מתמזג תמיד למאגר הכללי.</p>
  <button class="btn" onclick="exportFixes()">⬇️ ייצוא קובץ התיקונים שלי</button>
  &nbsp; ייבוא: <input type="file" id="fixFile" accept=".tsv,.txt">
  <button class="btn green" onclick="importFixes()">📥 ייבא למאגר הכללי</button>
  <span id="fixMsg" class="muted"></span>
 </div>
</div>

<div id="tabPar" style="display:none">
 <div class="card">
  <b class="title">אימון מטקסט מקביל</b>
  <p class="muted">סריקה + הטקסט שלה המוקלד (שורה מול שורה) = חומר אימון מיידי.</p>
  <input type="file" id="parFile" accept="image/*"><br><br>
  <textarea id="parText" placeholder="הדבק את הטקסט — כל שורה מול שורה בדף"></textarea>
  <button class="btn" onclick="parallelMatch()">🔗 הצמד שורות</button>
 </div>
 <div class="card" id="parCard" style="display:none">
  <b class="title">בדוק את ההצמדה</b>
  <div id="parLines"></div><br>
  <button class="btn green" onclick="saveAll('parLines','parMsg')">💾 שמור לאימון</button>
  <span id="parMsg" class="muted"></span>
 </div>
</div>

</main>
<div id="appDialog" role="presentation">
 <section id="appDialogPanel" role="dialog" aria-modal="true" aria-labelledby="appDialogTitle"
  aria-describedby="appDialogMessage">
  <h2 id="appDialogTitle">כתב-יד AI</h2>
  <p id="appDialogMessage"></p>
  <input id="appDialogInput" type="text">
  <div id="appDialogActions">
   <button id="appDialogCancel" class="btn teal" type="button">ביטול</button>
   <button id="appDialogAccept" class="btn" type="button">אישור</button>
  </div>
 </section>
</div>
<script>
let appDialogQueue=Promise.resolve();
function showAppDialog(message,{title='כתב-יד AI',confirm=false,input=null,initialValue=''}={}){
 const openDialog=()=>new Promise(resolve=>{
  const overlay=document.getElementById('appDialog'),inputEl=document.getElementById('appDialogInput');
  const cancel=document.getElementById('appDialogCancel'),accept=document.getElementById('appDialogAccept');
  document.getElementById('appDialogTitle').textContent=title;
  document.getElementById('appDialogMessage').textContent=message;
  inputEl.style.display=input===null?'none':'block';
  inputEl.value=initialValue;
  inputEl.placeholder=input||'';
  cancel.style.display=confirm||input!==null?'inline-block':'none';
  accept.textContent=input!==null?'שמירה':'אישור';
  overlay.classList.add('open');
  const finish=value=>{
   overlay.classList.remove('open');
   cancel.onclick=null;accept.onclick=null;overlay.onclick=null;
   document.removeEventListener('keydown',onKeyDown);
   resolve(value);
  };
  const onKeyDown=event=>{
   if(event.key==='Escape'&&(confirm||input!==null)){event.preventDefault();finish(input!==null?null:false);}
   else if(event.key==='Enter'&&input!==null){event.preventDefault();finish(inputEl.value);}
  };
  cancel.onclick=()=>finish(input!==null?null:false);
  accept.onclick=()=>finish(input!==null?inputEl.value:true);
  overlay.onclick=event=>{if(event.target===overlay&&!confirm&&input===null)finish(true);};
  document.addEventListener('keydown',onKeyDown);
  requestAnimationFrame(()=>input!==null?inputEl.focus():accept.focus());
 });
 const queued=appDialogQueue.then(openDialog,openDialog);
 appDialogQueue=queued.then(()=>undefined,()=>undefined);
 return queued;
}
function appAlert(message,title){return showAppDialog(String(message),{title:title||'כתב-יד AI'});}
function appConfirm(message,title){return showAppDialog(String(message),{title:title||'אישור נדרש',confirm:true});}
function appPrompt(message,initialValue='',title){return showAppDialog(String(message),{
 title:title||'עריכת ערך',confirm:true,input:'הקלד ערך',initialValue});}
let lastLines={lines:[],parLines:[]};
let chosenFiles=[];
async function api(path,body){const r=await fetch(path,{method:body?'POST':'GET',
 headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});
 return r.json();}
function showTab(id,btn){for(const t of ['tabScan','tabTrain','tabTmpl','tabPar','tabFix'])
 document.getElementById(t).style.display=t===id?'block':'none';
 document.querySelectorAll('nav button').forEach(b=>b.classList.remove('active'));
 btn.classList.add('active');if(id==='tabTrain')refreshExamples();}

/* ---------- קבצים ---------- */
const drop=document.getElementById('drop');
drop.addEventListener('dragover',e=>{e.preventDefault();drop.classList.add('over');});
drop.addEventListener('dragleave',()=>drop.classList.remove('over'));
drop.addEventListener('drop',e=>{e.preventDefault();drop.classList.remove('over');
 addFiles(e.dataTransfer.files);});
function fileChosen(){addFiles(document.getElementById('file').files);}
function addFiles(fl){for(const f of fl)chosenFiles.push(f);
 document.getElementById('fileList').textContent=
  chosenFiles.length?('נבחרו '+chosenFiles.length+' קבצים: '+chosenFiles.map(f=>f.name).join(', ')):'';}
function readAsB64(f){return new Promise(res=>{const r=new FileReader();
 r.onload=()=>res(r.result.split(',')[1]);r.readAsDataURL(f);});}
async function readFile(id){const f=document.getElementById(id).files[0];
 if(!f){await appAlert('בחר קובץ');return null;}
 return new Promise((res,rej)=>{const r=new FileReader();
  r.onload=()=>res(r.result.split(',')[1]);
  r.onerror=()=>rej(new Error('לא ניתן לקרוא את הקובץ שנבחר'));
  r.readAsDataURL(f);});}

/* ---------- שורות + ניווט מקלדת ---------- */
function autoGrow(el){el.style.height='auto';el.style.height=Math.max(24,el.scrollHeight)+'px';}
function appendLines(divId,arr,offset){const div=document.getElementById(divId);
 arr.forEach((l,k)=>{const i=offset+k;
  if(l.src&&l.src!==(div.dataset.lastSrc||'')){
   const h=document.createElement('div');h.className='srcHead';h.textContent='📄 '+l.src;div.appendChild(h);
   div.dataset.lastSrc=l.src;}
  const row=document.createElement('div');row.className='line';
  const num=document.createElement('span');num.className='num';num.textContent=i+1;
  if(divId==='lines'){
   const select=document.createElement('input');select.type='checkbox';
   select.className='aiRowSelect';select.title='כלול שורה זו בבקשת תיקון AI';
   select.checked=l.ai_crop!==false;select.disabled=l.ai_crop===false;
   select.dataset.aiIndex=i;
   select.addEventListener('change',updateAISelectionControls);
   row.appendChild(select);
  }
  const img=document.createElement('img');img.src='data:image/png;base64,'+l.img;
  img.onclick=()=>img.classList.toggle('zoom');
  const inp=document.createElement('textarea');inp.className='lt';inp.rows=1;
  inp.value=l.text;inp.dataset.i=i;inp.dataset.div=divId;
  if(l.conf!=null&&l.conf<0.75)inp.classList.add('lowconf');
  if(l.conf!=null)inp.title='ביטחון: '+Math.round(l.conf*100)+'%';
  inp.addEventListener('input',()=>autoGrow(inp));
  inp.addEventListener('focus',()=>{document.querySelectorAll('.line').forEach(x=>x.classList.remove('focused'));
   row.classList.add('focused');row.scrollIntoView({block:'nearest'});
   if(divId==='lines'){const mb=document.getElementById('magBar');
    document.getElementById('magImg').src=img.src;mb.style.display='block';}});
  inp.addEventListener('keydown',e=>{
   if(e.key==='Enter'||e.key==='ArrowDown'){e.preventDefault();moveFocus(divId,i,1);}
   else if(e.key==='ArrowUp'){e.preventDefault();moveFocus(divId,i,-1);}});
  row.appendChild(num);row.appendChild(img);row.appendChild(inp);
  if(l.note){const t=document.createElement('span');t.className='noteTag';t.textContent='הערה?';row.appendChild(t);}
  if(l.bold&&l.bold.length){const b=document.createElement('span');b.className='noteTag';
   b.style.fontWeight='bold';b.textContent='𝐁';
   b.title='זוהה קו הדגשה מתחת ל: '+l.bold.map(i=>(l.text.split(' ')[i]||'')).join(', ')+
    '\nהמילים ייוצאו ל-Word בכתב מודגש';row.appendChild(b);}
  if(l.sugg&&l.sugg!==l.text){const sg=document.createElement('button');sg.className='mini sugg';
   sg.textContent='💡';sg.title='הצעת תיקון לפי הקשר לשוני:\n'+l.sugg+'\n(לחץ להחלה — אפשר לערוך אחר כך)';
   sg.onclick=()=>{inp.value=l.sugg;l.text=l.sugg;autoGrow(inp);sg.remove();};
   row.appendChild(sg);}
  const del=document.createElement('button');del.className='mini';del.textContent='✖';
  del.title='מחק שורה (סרק/קו הדגשה)';
  del.onclick=()=>{inp.value='';l.text='';row.style.display='none';
   if(divId==='lines')updateAISelectionControls();};
  const mrg=document.createElement('button');mrg.className='mini';mrg.textContent='⤴';
  mrg.title='מזג את הטקסט לשורה הקודמת';
  mrg.onclick=()=>{const prev=document.querySelector('#'+divId+' textarea.lt[data-i="'+(i-1)+'"]');
   if(!prev){appAlert('אין שורה קודמת');return;}
   prev.value=(prev.value.trim()+' '+inp.value.trim()).trim();autoGrow(prev);
   const it=lastLines[divId][i-1];if(it)it.text=prev.value;
   inp.value='';l.text='';row.style.display='none';
   if(divId==='lines')updateAISelectionControls();};
  row.appendChild(del);row.appendChild(mrg);
  div.appendChild(row);requestAnimationFrame(()=>autoGrow(inp));});
 if(divId==='lines')updateAISelectionControls();}
function attachSugg(divId,i,sugg){/* עדכון 💡 לשורה שכבר הוצגה (מעבר ההקשר השני) */
 const inp=document.querySelector('#'+divId+' textarea.lt[data-i="'+i+'"]');
 if(!inp)return;const row=inp.closest('.line');const l=lastLines[divId][i];if(!l)return;
 l.sugg=sugg;const old=row.querySelector('.mini.sugg');if(old)old.remove();
 if(!sugg||sugg===inp.value)return;
 const sg=document.createElement('button');sg.className='mini sugg';sg.textContent='💡';
 sg.title='הצעת תיקון לפי הקשר לשוני:\n'+sugg+'\n(לחץ להחלה — אפשר לערוך אחר כך)';
 sg.onclick=()=>{inp.value=sugg;l.text=sugg;autoGrow(inp);sg.remove();};
 row.appendChild(sg);}
function moveFocus(divId,i,d){const nxt=document.querySelector('#'+divId+' textarea.lt[data-i="'+(i+d)+'"]');
 if(nxt){nxt.focus();const p=nxt.value.length;nxt.setSelectionRange(p,p);}}
function collect(divId){document.querySelectorAll('#'+divId+' textarea.lt').forEach(inp=>{
  const it=lastLines[divId][+inp.dataset.i];if(it)it.text=inp.value;});return lastLines[divId];}

/* ---------- תיקון טיוטה עם AI ---------- */
let aiCorrections=[],aiMetricsSummary='';
function getVisibleAILines(syncText=true){
 const items=syncText?collect('lines'):lastLines.lines,visible=[];
 document.querySelectorAll('#lines textarea.lt').forEach(inp=>{
  const row=inp.closest('.line');
  if(!row||row.style.display==='none')return;
  const index=Number(inp.dataset.i),item=items[index];
  if(item)visible.push({
   index,item,pageKey:item.page==null?item.src:item.page,
   aiCrop:item.ai_crop!==false,
   selected:!!row.querySelector('.aiRowSelect')?.checked});
 });
 return visible;
}
function setAISelection(selected){
 document.querySelectorAll('#lines .line .aiRowSelect:not(:disabled)').forEach(input=>{
  if(input.closest('.line').style.display!=='none')input.checked=selected;
 });
 updateAISelectionControls();
}
function invertAISelection(){
 document.querySelectorAll('#lines .line .aiRowSelect:not(:disabled)').forEach(input=>{
  if(input.closest('.line').style.display!=='none')input.checked=!input.checked;
 });
 updateAISelectionControls();
}
async function selectAILinesByNumber(){
 const raw=document.getElementById('aiLineNumbers').value.trim(),requested=new Set();
 if(!raw){await appAlert('הזן מספרי שורות, למשל 1,3-5.');return;}
 for(const part of raw.split(',')){
  const match=part.trim().match(/^(\d+)(?:-(\d+))?$/);
  if(!match){await appAlert(`בחירת שורות לא תקינה: ${part}`);return;}
  const start=Number(match[1]),end=Number(match[2]||match[1]);
  if(start<1||end<start||end>lastLines.lines.length){
   await appAlert(`מספרי שורות חייבים להיות בין 1 ל-${lastLines.lines.length}.`);return;}
  for(let number=start;number<=end;number++)requested.add(number);
 }
 const inputs=[...document.querySelectorAll('#lines .line .aiRowSelect:not(:disabled)')]
  .filter(input=>input.closest('.line').style.display!=='none');
 const available=new Set(inputs.map(input=>Number(input.dataset.aiIndex)+1));
 const missing=[...requested].filter(number=>!available.has(number));
 if(missing.length){await appAlert(`השורות ${missing.join(', ')} אינן זמינות לשליחה ל-AI.`);return;}
 document.getElementById('aiScope').value='manual';
 for(const input of inputs)
  input.checked=requested.has(Number(input.dataset.aiIndex)+1);
 updateAISelectionControls();
}
function readAISettings(){
 const number=(id,optional=false)=>{
  const value=document.getElementById(id).value.trim();
  return value===''&&optional?null:Number(value);
 };
 return {batch_size:number('aiBatchSize'),temperature:number('aiTemperature'),
  top_p:number('aiTopP',true),top_k:number('aiTopK',true),
  max_output_tokens:number('aiMaxOutputTokens',true),
  request_timeout:number('aiRequestTimeout')};
}
async function saveAISettings(showMessage=true){
 const msg=document.getElementById('aiKeyMsg');
 try{
  const settings=readAISettings();
  const r=await api('/api/ai_settings',settings);
  if(r.error)throw Error(r.error);
  if(showMessage)msg.textContent='הגדרות תיקון AI נשמרו ומשותפות לתוכנה ול-CLI.';
  return settings;
 }catch(e){
  msg.textContent='שמירת הגדרות AI נכשלה: '+e.message;
  return null;
 }
}
async function loadAISettings(){
 try{
  const s=await api('/api/ai_settings');
  if(s.error)throw Error(s.error);
  document.getElementById('aiBatchSize').value=s.batch_size;
  document.getElementById('aiTemperature').value=s.temperature;
  document.getElementById('aiTopP').value=s.top_p==null?'':s.top_p;
  document.getElementById('aiTopK').value=s.top_k==null?'':s.top_k;
  document.getElementById('aiMaxOutputTokens').value=
   s.max_output_tokens==null?'':s.max_output_tokens;
  document.getElementById('aiRequestTimeout').value=s.request_timeout;
 }catch(e){document.getElementById('aiKeyMsg').textContent=
  'שגיאה בטעינת הגדרות AI: '+e.message;}
}
async function resetAISettings(){
 if(!await appConfirm('לאפס את פרמטרי תיקון ה-AI לברירות המחדל?'))return;
 try{
  const r=await api('/api/ai_settings/reset',{});
  if(r.error)throw Error(r.error);
  const s=r.settings;
  document.getElementById('aiBatchSize').value=s.batch_size;
  document.getElementById('aiTemperature').value=s.temperature;
  document.getElementById('aiTopP').value='';
  document.getElementById('aiTopK').value='';
  document.getElementById('aiMaxOutputTokens').value='';
  document.getElementById('aiRequestTimeout').value=s.request_timeout;
  document.getElementById('aiKeyMsg').textContent='הגדרות AI אופסו.';
 }catch(e){document.getElementById('aiKeyMsg').textContent=
  'איפוס הגדרות AI נכשל: '+e.message;}
}
function updateAISelectionControls(){
 const allVisible=getVisibleAILines(false),visible=allVisible.filter(x=>x.aiCrop);
 const pages=[...new Set(visible.map(x=>x.pageKey))];
 const scope=document.getElementById('aiScope'),amount=document.getElementById('aiAmount');
 if(!scope||!amount)return;
 const manual=scope.value==='manual';
 document.body.classList.toggle('aiManualSelection',manual);
 document.getElementById('aiManualControls').style.display=manual?'inline-flex':'none';
 document.getElementById('aiAmountControl').style.display=manual?'none':'inline';
 const max=scope.value==='pages'?pages.length:visible.length;
 amount.max=String(max);
 amount.value=String(Math.min(Math.max(1,Number(amount.value)||1),Math.max(1,max)));
 document.getElementById('aiSelectionInfo').textContent=
  !visible.length&&allVisible.length
   ? 'לתיקון AI יש לתמלל עם פילוח אוטומטי לשורות.'
   : manual
    ? `נבחרו ${visible.filter(x=>x.selected).length} מתוך ${visible.length} שורות זמינות לשליחה.`
   : scope.value==='pages'
    ? `${pages.length} עמודים זמינים; יישלחו חיתוכי השורות של ${Math.min(Number(amount.value),pages.length)} עמודים.`
    : `${visible.length} שורות זמינות; יישלחו ${Math.min(Number(amount.value),visible.length)} חיתוכים.`;
}
async function aiCorrect(){
 updateAISelectionControls();
 const allVisible=getVisibleAILines(),visible=allVisible.filter(x=>x.aiCrop);
 const scope=document.getElementById('aiScope').value;
 const amount=Number(document.getElementById('aiAmount').value);
 const pages=[...new Set(visible.map(x=>x.pageKey))];
 const selected=scope==='manual'
  ? visible.filter(x=>x.selected)
  : scope==='pages'
   ? visible.filter(x=>pages.slice(0,amount).includes(x.pageKey))
   : visible.slice(0,amount);
 if(!selected.length){
  await appAlert(allVisible.some(x=>!x.aiCrop)
   ? 'כדי לשלוח חיתוכי שורות ל-AI, סמן "עמודים מלאים (פילוח אוטומטי לשורות)" ותמלל מחדש.'
   : scope==='manual'?'סמן לפחות שורה אחת לשליחה לתיקון AI.':'אין שורות גלויות לשליחה לתיקון');
  return;
 }
 const btn=document.getElementById('aiCorrectBtn'),msg=document.getElementById('saveMsg');
 btn.disabled=true;msg.textContent=`שולח ${selected.length} חיתוכי שורות ל-Google AI Studio...`;
 try{
  const settings=await saveAISettings(false);
  if(!settings)throw Error(document.getElementById('aiKeyMsg').textContent);
  const r=await api('/api/ai_correct',{
   lines:selected.map(x=>({img:x.item.img,text:x.item.text})),
   ...settings,key_id:document.getElementById('aiKeyChoice').value||null});
  if(r.error)throw Error(r.error);
  aiCorrections=(r.corrections||[]).map(c=>({
   ...c,index:selected[c.index].index,lineNumber:selected[c.index].index+1,
   original:selected[c.index].item.text,applied:false,skipped:false}));
  renderAIReview(r.keys_used||[],r.metrics||{},r.warning||'');
  msg.textContent=`התקבלו ${aiCorrections.length} הצעות. הן עדיין טיוטה ולא נשמרו.`+
   (r.warning?' '+r.warning:'');
 }catch(e){msg.textContent='';await appAlert('תיקון AI נכשל: '+e.message,'תיקון AI נכשל');}
 finally{btn.disabled=false;}}
function renderAIReview(keys,metrics,warning){
 const panel=document.getElementById('aiReview'),list=document.getElementById('aiReviewList');
 list.replaceChildren();
 const usage=metrics.total_tokens==null?'טוקנים: לא דווחו על ידי השירות':
  `טוקנים: ${metrics.total_tokens} בסך הכול (קלט ${metrics.prompt_tokens}, פלט ${metrics.output_tokens})`;
 const batches=(metrics.batches||[]).map((b,i)=>
  `בקשה ${i+1}: ${b.line_count} שורות, ${(b.image_bytes/1024).toFixed(1)}KB תמונות, `+
  `${(b.request_bytes/1024).toFixed(1)}KB נשלחו, `+
  `${b.elapsed_seconds}s, ${b.total_tokens==null?'טוקנים לא דווחו':b.total_tokens+' טוקנים'}`).join(' · ');
 const generation=metrics.generation||{};
 const stats=`${metrics.line_count||0} שורות ב-${metrics.batch_count||0} בקשות; `+
  `זמן כולל ${metrics.elapsed_seconds||0}s; ${usage}. ${batches}. `+
  `פרמטרים: batch=${metrics.batch_size||'—'}, temperature=${generation.temperature??'—'}, `+
  `top-p=${generation.top_p??'ברירת מחדל'}, top-k=${generation.top_k??'ברירת מחדל'}, `+
  `max tokens=${generation.max_output_tokens??'ברירת מחדל'}, timeout=${generation.request_timeout??'—'}s. `+
  `יתרת מכסה אינה מדווחת דרך בקשת התיקון; טוקני הקלט כוללים גם תמונות.`+
  (warning?` ${warning}`:'');
 aiMetricsSummary=stats;
 if(!aiCorrections.length){panel.style.display='block';
  document.getElementById('aiReviewMsg').textContent='לא נמצאו שורות לשינוי. '+stats;
  return;}
 panel.style.display='block';
 for(const c of aiCorrections){
  const row=document.createElement('div');row.className='line';row.style.cssText='padding:8px;border-bottom:1px solid #e3e7ef;flex-wrap:wrap';
  const label=document.createElement('b');label.textContent='שורה '+c.lineNumber+': ';
  const before=document.createElement('span');before.className='muted';before.textContent='זוהה: '+c.original;
  const after=document.createElement('span');after.style.cssText='font-weight:600;color:#145c42';after.textContent='הצעת AI: '+c.text;
  const apply=document.createElement('button');apply.className='btn green';apply.textContent='החל לטיוטה';
  apply.onclick=()=>{applyAICorrection(c);apply.disabled=true;apply.textContent='הוחל בטיוטה';};
  const skip=document.createElement('button');skip.className='btn';skip.textContent='השאר מקור';
  skip.onclick=()=>{c.skipped=true;row.style.display='none';updateAIReviewStatus();};
  row.append(label,before,after,apply,skip);list.appendChild(row);}
 document.getElementById('aiReviewMsg').textContent=
  `${stats}. מפתח/ות: ${keys.length?keys.join(', '):'—'}. `+
  `ההצעות לא נשמרות עד שתלחץ בעצמך על שמירה לאימון.`;
}
function applyAICorrection(c){
 const item=lastLines.lines[c.index];
 const input=document.querySelector('#lines textarea.lt[data-i="'+c.index+'"]');
 if(!item||!input)return;
 item.text=c.text;input.value=c.text;autoGrow(input);c.applied=true;updateAIReviewStatus();}
function applyAllAICorrections(){
 let applied=0;
 for(const c of aiCorrections){if(!c.applied&&!c.skipped){applyAICorrection(c);applied++;}}
 document.getElementById('aiReviewMsg').textContent=
  'הוחלו '+applied+' הצעות בטיוטה. עבור על הטקסט וערוך לפי הצורך לפני שמירה לאימון. '+
  aiMetricsSummary;}
function updateAIReviewStatus(){
 const applied=aiCorrections.filter(c=>c.applied).length;
 const skipped=aiCorrections.filter(c=>c.skipped).length;
 document.getElementById('aiReviewMsg').textContent=
  'הוחלו '+applied+' הצעות בטיוטה; '+skipped+' נדחו. ניתן לערוך את השורות לפני שמירה לאימון. '+
  aiMetricsSummary;}
let aiKeyChecks=Object.create(null);
let aiModelTests=Object.create(null);
function aiKeyStatusText(check){
 if(!check)return 'לא נבדק';
 const labels={
  connected:'✅ מחובר — מכסת יצירה לא נבדקה',
  rejected:'❌ מפתח נדחה/אין הרשאה',
  quota:'⚠️ שירות דיווח על מכסה/קצב',
  unreachable:'⚠️ אין חיבור ל-Google',
  service_error:'⚠️ שגיאת שירות'
 };
 const checked=check.checked_at?new Date(check.checked_at).toLocaleString():'';
 return `${labels[check.status]||'⚠️ סטטוס לא ידוע'}${check.detail?' — '+check.detail:''}`+
  `${checked?' · נבדק '+checked:''} · ${check.elapsed_seconds}s`;
}
async function checkAIKey(id){
 const msg=document.getElementById('aiKeyMsg');
 try{
  const r=await api('/api/ai_keys/check',{id});
  if(r.error)throw Error(r.error);
  aiKeyChecks[r.result.id]=r.result;renderAIKeyList();
 }catch(e){msg.textContent='בדיקת החיבור נכשלה: '+e.message;}
}
async function loadAIModels(id){
 const msg=document.getElementById('aiKeyMsg');
 try{
  const r=await api('/api/ai_keys/models',{id});
  if(r.error)throw Error(r.error);
  aiModelCatalog[id]=r.models;
  delete aiModelTests[id];
  renderAIKeyList();
  msg.textContent=`Google מציג ${r.models.length} מודלים שתומכים ביצירת תוכן עבור ${r.name}. `+
   'הרשימה אינה מבטיחה שהמודל יענה כרגע; שגיאת 503 בדרך כלל מציינת עומס זמני.';
 }catch(e){msg.textContent='טעינת רשימת המודלים נכשלה: '+e.message;}
}
const aiModelTestLabels={
 working:'✅ יצירה הצליחה',
 quota:'⚠️ מכסה/קצב',
 overloaded:'⏳ עומס זמני',
 unavailable:'❌ המודל אינו זמין',
 unsupported:'⚠️ הבקשה אינה נתמכת',
 access_denied:'❌ אין הרשאה',
 unreachable:'⚠️ אין חיבור',
 no_response:'⚠️ לא התקבל טקסט',
 service_error:'⚠️ שגיאת שירות'
};
function recommendedAIModel(id){
 const results=aiModelTests[id]?.results||{};
 const working=Object.values(results).filter(result=>result.status==='working');
 const priority=['gemini-3.1-flash-lite','gemini-3.1-flash-lite-preview',
  'gemini-3.5-flash-lite','gemini-flash-lite-latest','gemini-3.5-flash',
  'gemini-3.6-flash','gemini-3.7-flash','gemini-3-flash-preview'];
 for(const model of priority){const result=working.find(item=>item.id===model);if(result)return result;}
 return working[0]||null;
}
async function testAIModels(id){
 const models=aiModelCatalog[id]||[],msg=document.getElementById('aiKeyMsg');
 if(!models.length){msg.textContent='תחילה טען את רשימת המודלים למפתח הזה.';return;}
 if(!await appConfirm(`תישלח בקשת יצירה קצרה לכל אחד מ-${models.length} המודלים ברשימה. `+
  'הבדיקה פונה ל-Google בפועל ועשויה לצרוך מכסה. להמשיך?'))return;
 const state=aiModelTests[id]={results:Object.create(null),running:true,completed:0,total:models.length};
 renderAIKeyList();
 for(const model of models){
  try{
   const r=await api('/api/ai_keys/test_model',{id,model:model.id});
   if(r.error)throw Error(r.error);
   state.results[model.id]=r.result;
  }catch(e){
   state.results[model.id]={id:model.id,status:'service_error',detail:e.message};
  }
  state.completed++;
  renderAIKeyList();
 }
 state.running=false;
 renderAIKeyList();
 const recommendation=recommendedAIModel(id);
 msg.textContent=recommendation
  ? `הבדיקה הסתיימה. מומלץ כרגע: ${recommendation.id}. אפשר לבחור בו מהרשימה.`
  : 'הבדיקה הסתיימה; אף מודל לא החזיר תשובת טקסט תקינה. בדוק את הסיבות לכל מודל.';
}
async function saveAIKeyModel(id,model){
 const msg=document.getElementById('aiKeyMsg');
 try{
  const r=await api('/api/ai_keys/model',{id,model});
  if(r.error)throw Error(r.error);
  const key=aiKeyList.find(item=>item.id===id);
  if(key)key.model=r.key.model;
  delete aiKeyChecks[id];
  renderAIKeyList();
  msg.textContent=`המודל ${r.key.model} נשמר עבור המפתח ${r.key.name}; בודק גישה...`;
  await checkAIKey(id);
 }catch(e){msg.textContent='שמירת המודל נכשלה: '+e.message;await loadAIKeys();}
}
async function checkAllAIKeys(){
 const btn=document.getElementById('aiKeysCheckAll'),msg=document.getElementById('aiKeyMsg');
 btn.disabled=true;btn.textContent='בודק חיבור...';
 try{
  const r=await api('/api/ai_keys/check',{});
  if(r.error)throw Error(r.error);
  for(const result of r.results)aiKeyChecks[result.id]=result;
  renderAIKeyList();
  const connected=r.results.filter(x=>x.status==='connected').length;
  msg.textContent=`נבדקו ${r.results.length} מפתחות; ${connected} התחברו בהצלחה. בדיקה זו אינה מוודאת מכסת יצירת תוכן.`;
 }catch(e){msg.textContent='רענון בדיקת החיבור נכשל: '+e.message;}
 finally{btn.disabled=false;btn.textContent='🔄 בדוק/רענן חיבור לכל המפתחות';}
}
function renderAIKeyList(){
 const list=document.getElementById('aiKeyList');
 list.replaceChildren();
 renderAIKeySelector();
 const keys=aiKeyList;
 if(!keys.length){list.textContent='לא נשמרו מפתחות עדיין.';return;}
 for(const key of keys){
  const row=document.createElement('div');row.style.cssText='margin-top:8px;display:flex;gap:8px;align-items:center;flex-wrap:wrap';
  const name=document.createElement('input');name.type='text';name.maxLength=60;
  name.value=key.name;name.title='שם המפתח השמור';name.style.width='150px';
  const rename=document.createElement('button');rename.className='mini';
  rename.textContent='שמור שם';rename.onclick=()=>renameAIKey(key.id,name.value);
  const status=document.createElement('span');status.textContent=aiKeyStatusText(aiKeyChecks[key.id]);
  status.title=aiKeyChecks[key.id]?aiKeyChecks[key.id].detail:'';
  const modelSelect=document.createElement('select');
  const models=aiModelCatalog[key.id]||[];
  const selectedAvailable=models.some(model=>model.id===key.model);
  const selectedOption=document.createElement('option');
  selectedOption.value=key.model;
  selectedOption.textContent=selectedAvailable
   ? `${key.model} (מופיע ברשימת Google)`
   : `${key.model} (ברירת מחדל; טרם אומת מול Google)`;
  modelSelect.appendChild(selectedOption);
  for(const model of models){
   if(model.id===key.model)continue;
   const option=document.createElement('option');
   option.value=model.id;
   const tested=aiModelTests[key.id]?.results?.[model.id];
   option.textContent=`${model.display_name} — ${model.id}`+
    (tested?` · ${aiModelTestLabels[tested.status]||tested.status}`:'');
   option.title=model.description||'';
   modelSelect.appendChild(option);
  }
  modelSelect.value=key.model;
  modelSelect.disabled=!models.length;
  modelSelect.title=models.length
   ? 'המודל שבו ישתמש מפתח זה בתיקון AI'
   : 'לחץ על „טען מודלים” כדי לקבל את המודלים הזמינים למפתח';
  modelSelect.onchange=()=>saveAIKeyModel(key.id,modelSelect.value);
  const test=document.createElement('button');test.className='mini';test.textContent='בדוק חיבור';
  test.onclick=()=>checkAIKey(key.id);
  const refreshModels=document.createElement('button');refreshModels.className='mini';
  refreshModels.textContent='טען מודלים';
  refreshModels.onclick=()=>loadAIModels(key.id);
  const testModels=document.createElement('button');testModels.className='mini';
  testModels.textContent=aiModelTests[key.id]?.running?'בודק מודלים...':'בדוק מודלים בפועל';
  testModels.disabled=!models.length||!!aiModelTests[key.id]?.running;
  testModels.onclick=()=>testAIModels(key.id);
  const remove=document.createElement('button');remove.className='mini';remove.textContent='הסר';
  remove.onclick=()=>removeAIKey(key.id);
  row.append(name,rename,modelSelect,refreshModels,testModels,status,test,remove);list.appendChild(row);
  const testState=aiModelTests[key.id];
  if(testState){
   const results=document.createElement('div');
   results.style.cssText='margin:4px 0 10px 24px;padding:8px 12px;background:#f6f8fc;border-radius:8px';
   if(testState.running){
    const progress=document.createElement('div');progress.className='muted';
    progress.textContent=`בודק מודלים: ${testState.completed} מתוך ${testState.total} הושלמו.`;
    results.appendChild(progress);
   }
   for(const model of models){
    const tested=testState.results[model.id];if(!tested)continue;
    const line=document.createElement('div');line.className='muted';
    line.textContent=`${model.display_name} (${model.id}) — `+
     `${aiModelTestLabels[tested.status]||tested.status}: ${tested.detail}`;
    results.appendChild(line);
   }
   const recommendation=recommendedAIModel(key.id);
   if(recommendation){
    const rec=document.createElement('div');rec.style.cssText='margin-top:6px;font-weight:600;color:#145c42';
    rec.textContent=`מומלץ כרגע: ${recommendation.id} — הבדיקה הצליחה.`;
    const choose=document.createElement('button');choose.className='mini';
    choose.textContent='בחר מודל מומלץ';choose.onclick=()=>saveAIKeyModel(key.id,recommendation.id);
    results.append(rec,choose);
   }else if(!testState.running&&testState.completed){
    const none=document.createElement('div');none.className='muted';
    none.textContent='לא נמצא מודל שהחזיר תשובת טקסט תקינה בבדיקה.';
    results.appendChild(none);
   }
   list.appendChild(results);
  }
 }
}
function renderAIKeySelector(){
 const select=document.getElementById('aiKeyChoice');if(!select)return;
 const selected=select.value;
 select.replaceChildren(new Option('בחירה אוטומטית',''));
 for(const key of aiKeyList)select.appendChild(new Option(key.name,key.id));
 select.value=aiKeyList.some(key=>key.id===selected)?selected:'';
}
let aiKeyList=[];
let aiModelCatalog=Object.create(null);
async function loadAIKeys(){try{const r=await api('/api/ai_keys');
 if(r.error)throw Error(r.error);
 aiKeyList=r.keys;
 const currentIds=new Set(aiKeyList.map(key=>key.id));
 for(const id of Object.keys(aiKeyChecks))if(!currentIds.has(id))delete aiKeyChecks[id];
 for(const id of Object.keys(aiModelCatalog))if(!currentIds.has(id))delete aiModelCatalog[id];
 for(const id of Object.keys(aiModelTests))if(!currentIds.has(id))delete aiModelTests[id];
 renderAIKeyList();
 }catch(e){document.getElementById('aiKeyMsg').textContent='שגיאה בטעינת המפתחות: '+e.message;}}
async function addAIKey(){const name=document.getElementById('aiKeyName').value.trim();
 const key=document.getElementById('aiKeyValue').value.trim(),msg=document.getElementById('aiKeyMsg');
 if(!name||!key){msg.textContent='צריך למלא שם ומפתח API.';return;}
 try{const r=await api('/api/ai_keys/add',{name:name,key:key});
 if(r.error)throw Error(r.error);
 document.getElementById('aiKeyName').value=''; document.getElementById('aiKeyValue').value='';
 msg.textContent='המפתח נשמר בהצפנה במחשב. לחץ על „בדוק חיבור” כדי לבדוק אותו מול Google.';
 await loadAIKeys();
 }catch(e){msg.textContent='לא ניתן לשמור את המפתח: '+e.message;}}
async function renameAIKey(id,name){
 const msg=document.getElementById('aiKeyMsg');
 try{const r=await api('/api/ai_keys/rename',{id:id,name:name.trim()});
 if(r.error)throw Error(r.error);
 msg.textContent='שם המפתח עודכן ונשמר לנתוני התוכנה.';
 await loadAIKeys();
 }catch(e){msg.textContent='לא ניתן לשנות את שם המפתח: '+e.message;}}
async function removeAIKey(id){if(!await appConfirm('להסיר את מפתח ה-API הזה?'))return;
 try{const r=await api('/api/ai_keys/remove',{id:id});
 if(r.error)throw Error(r.error);
 await loadAIKeys();}catch(e){await appAlert('לא ניתן להסיר את המפתח: '+e.message,'הסרת מפתח נכשלה');}}

/* ---------- תמלול חי ---------- */
function setUploadLock(on){document.getElementById('drop').classList.toggle('disabled',on);
 document.getElementById('btnGo').disabled=on;}
async function transcribe(){if(!chosenFiles.length){await appAlert('בחר קבצים קודם');return;}
 const btn=document.getElementById('btnGo');setUploadLock(true);btn.textContent='מתמלל...';
 document.getElementById('modelUsed').textContent='מזהה מודל...';
 lastLines.lines=[];const linesDiv=document.getElementById('lines');
 document.getElementById('aiReview').style.display='none';aiCorrections=[];
 linesDiv.innerHTML='';linesDiv.dataset.lastSrc='';
 document.getElementById('resultsCard').style.display='none';
 document.getElementById('scanProg').style.display='block';
 try{
  const files=[];for(const f of chosenFiles)files.push({name:f.name,data:await readAsB64(f)});
  const start=await api('/api/transcribe',{files:files,page:document.getElementById('pageMode').checked,
   writer:document.getElementById('writer').value,prefer:document.getElementById('modelPick').value,
   fix:document.getElementById('fixMode').checked});
  while(true){
   const s=await api('/api/transcribe_status',{job:start.job,from:lastLines.lines.length});
   if(s.models_used&&s.models_used.length){
    const names=s.models_used.map(model=>model==='__base__'?'מודל הבסיס':
     model==='__general__'?'המודל הכללי':
     model==='__typed__'?'מודל הטקסט המוקלד':'המודל האישי של '+model);
    document.getElementById('modelUsed').textContent='בפועל: '+names.join(' + ');
   }
   if(s.lines&&s.lines.length){const off=lastLines.lines.length;
    s.lines.forEach(l=>l.orig=l.text);  /* שמירת המקור ללמידת תיקונים */
    lastLines.lines.push(...s.lines);appendLines('lines',s.lines,off);
    document.getElementById('resultsCard').style.display='block';}
   const pct=Math.round(100*(s.progress||0));
   document.getElementById('scanBar').style.width=pct+'%';
   document.getElementById('scanBar').textContent=pct+'%';
   document.getElementById('scanCount').textContent=
    (s.pages_total>1?('עמוד '+Math.min(s.pages_done+1,s.pages_total)+'/'+s.pages_total+' · '):'')+
    (s.cur||'')+' · '+lastLines.lines.length+' שורות עד כה';
   if(s.done||s.error){
    if(s.error&&(!s.models_used||!s.models_used.length))
     document.getElementById('modelUsed').textContent='לא נבחר מודל';
    if(s.suggs)for(const k in s.suggs)attachSugg('lines',+k,s.suggs[k]);
    if(s.error)await appAlert(s.error,'שגיאת תמלול');break;}
   await new Promise(r=>setTimeout(r,700));}
 }catch(e){document.getElementById('modelUsed').textContent='התמלול נכשל';
  await appAlert('שגיאה: '+e,'התמלול נכשל');}
 document.getElementById('scanProg').style.display='none';
 setUploadLock(false);btn.textContent='🚀 תמלל הכול';
 chosenFiles=[];document.getElementById('fileList').textContent='';
 document.getElementById('file').value='';
 if(!lastLines.lines.length)await appAlert('לא זוהו שורות');}

/* ---------- ייצוא ---------- */
function expMode(){return document.querySelector('input[name=expMode]:checked').value;}
function fullText(){const items=collect('lines').filter(l=>l.text.trim());
 const parts=items.map(l=>l.note?('(הערה: '+l.text.trim()+')'):l.text.trim());
 const m=expMode();
 if(m==='orig')return parts.join('\n');
 const cont=parts.join(' ');
 if(m==='flow')return cont;
 return cont.replace(/\.\s+/g,'.\n');}
function copyText(){navigator.clipboard.writeText(fullText());
 document.getElementById('saveMsg').textContent='הועתק ✓';}
function downloadTxt(){const a=document.createElement('a');
 a.href=URL.createObjectURL(new Blob([fullText()],{type:'text/plain;charset=utf-8'}));
 a.download='transcript.txt';a.click();}
async function downloadDocx(){let paragraphs;
 if(expMode()==='orig'){ /* מצב שורות-מקור: מעביר גם אינדקסי מודגש (𝐁) */
  paragraphs=collect('lines').filter(l=>l.text.trim())
   .map(l=>({text:l.note?('(הערה: '+l.text.trim()+')'):l.text.trim(),
             bold:(!l.note&&l.bold)?l.bold:[]}));
 }else paragraphs=fullText().split('\n');
 const r=await fetch('/api/export_docx',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({paragraphs:paragraphs})});
 const blob=await r.blob();const a=document.createElement('a');
 a.href=URL.createObjectURL(blob);a.download='transcript.docx';a.click();}
async function downloadDocxLayout(){const items=collect('lines')
  .filter(l=>l.text&&l.text.trim())
  .map(l=>({text:l.text,note:l.note,geom:l.geom||null,bold:l.bold||[]}));
 const r=await fetch('/api/export_docx_layout',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({lines:items})});
 const blob=await r.blob();const a=document.createElement('a');
 a.href=URL.createObjectURL(blob);a.download='transcript_layout.docx';a.click();}
async function saveAll(divId,msgId){const w=document.getElementById('writer').value;
 if(!w){await appAlert('הוסף כותב קודם (בלשונית האימון)');return;}
 const r=await api('/api/save',{writer:w,lines:collect(divId)});
 let msg='נשמרו '+r.saved+' שורות ✓';
 if(r.quarantined)msg+=' · '+r.quarantined+' ממתינות לאישור בהסגר (בלשונית האימון)';
 if(r.dups)msg+=' · '+r.dups+' כפילויות דולגו';
 document.getElementById(msgId).textContent=msg;refreshAll(w);}

/* ---------- מודל התיקונים ---------- */
async function deepSuggest(){const w=document.getElementById('writer').value;
 const items=collect('lines');
 if(!items||!items.length){await appAlert('אין תמלול לנתח');return;}
 const msg=document.getElementById('saveMsg');
 msg.textContent='🧠 מנתח את כל המסמך (הקשר מלא + רצף חוצה-שורות)...';
 try{
  const r=await api('/api/suggest_page',{writer:w,
   lines:items.map(l=>({text:l.text||'',src:l.src||'',typed:!!l.typed}))});
  let n=0;for(const k in r.suggs){attachSugg('lines',+k,r.suggs[k]);n++;}
  msg.textContent=n?('נמצאו '+n+' הצעות 💡 — עבור עליהן אחת-אחת או "החל את כל ההצעות"'):
   'הניתוח הסתיים — לא נמצאו הצעות נוספות';
 }catch(e){msg.textContent='שגיאה בניתוח: '+e;}}
function applyAllSuggs(){let n=0;
 document.querySelectorAll('#lines .mini.sugg').forEach(b=>{b.click();n++;});
 document.getElementById('saveMsg').textContent=n?('הוחלו '+n+' הצעות 💡 — אפשר לשמור אותן למודל התיקונים'):'אין הצעות ממתינות';}
async function saveFixes(){const w=document.getElementById('writer').value;
 collect('lines');
 const items=lastLines.lines.filter(l=>l.orig&&l.text&&l.orig!==l.text)
  .map(l=>({orig:l.orig,final:l.text}));
 if(!items.length){await appAlert('אין תיקונים ללמידה — ערוך שורות או החל הצעות קודם');return;}
 const r=await api('/api/save_fixes',{writer:w,scope:document.getElementById('fixScope').value,lines:items});
 document.getElementById('saveMsg').textContent=r.msg;loadFixStats();}
async function loadFixStats(){try{const w=document.getElementById('writer').value;
 const s=await api('/api/fixes_stats?writer='+encodeURIComponent(w||''));
 document.getElementById('fixCntP').textContent=s.personal;
 document.getElementById('fixCntG').textContent=s.general;
 document.getElementById('fixCntA').textContent=s.active!=null?s.active:'—';
 const tbl=document.getElementById('fixTable');
 tbl.innerHTML='<tr><th>שגוי</th><th>נכון</th><th>פעמים</th><th></th></tr>';
 if(!s.top.length){const tr=document.createElement('tr');
  tr.innerHTML='<td colspan="4" class="muted">אין תיקונים חדשים מאז הניקוי האחרון (הקיימים פעילים ברקע)</td>';
  tbl.appendChild(tr);}
 for(const t of s.top){const tr=document.createElement('tr');
  tr.innerHTML='<td>'+t.wrong+'</td><td>'+t.right+'</td><td>'+t.count+'</td>';
  const td=document.createElement('td');
  const ed=document.createElement('button');ed.className='mini';ed.textContent='✏️';
  ed.title='עריכת התיקון — הצורה שתקליד תהפוך לרוב הקובע';
  ed.onclick=async()=>{const nr=await appPrompt('הצורה הנכונה עבור "'+t.wrong+'":',t.right);
   if(!nr||!nr.trim()||nr.trim()===t.right)return;
   const r=await api('/api/fixes_override',{writer:w,wrong:t.wrong,right:nr.trim()});
   document.getElementById('fixClearMsg').textContent=r.msg;loadFixStats();};
  const dl=document.createElement('button');dl.className='mini';dl.textContent='🗑';
  dl.title='הסרת התיקון מהמודל לגמרי (לא יוצע יותר)';
  dl.onclick=async()=>{if(!await appConfirm('להסיר את התיקון "'+t.wrong+' ← '+t.right+'" מהמודל?'))return;
   const r=await api('/api/fixes_remove',{writer:w,wrong:t.wrong});
   document.getElementById('fixClearMsg').textContent=r.msg;loadFixStats();};
  td.appendChild(ed);td.appendChild(dl);tr.appendChild(td);
  tbl.appendChild(tr);}}catch(e){}}
function exportFixes(){const w=document.getElementById('writer').value;
 location.href='/api/export_fixes?writer='+encodeURIComponent(w||'');}
async function clearFixes(){const w=document.getElementById('writer').value;
 const r=await api('/api/fixes_clear',{writer:w});
 document.getElementById('fixClearMsg').textContent=r.msg;loadFixStats();}
async function importFixes(){const b64=await readFile('fixFile');
 if(!b64)return;
 const r=await api('/api/import_fixes',{file:b64});
 document.getElementById('fixMsg').textContent=r.msg;loadFixStats();}

/* ---------- הסגר ---------- */
let quarSeq=0; /* מונה-רצף: שתי קריאות מקבילות (רענון+החלפת כותב) הציגו את
                  הרשימה פעמיים — רק הקריאה האחרונה מציירת */
async function refreshQuar(){const w=document.getElementById('writer').value;
 const seq=++quarSeq;
 const card=document.getElementById('quarCard'),div=document.getElementById('quarLines');
 if(!w){div.innerHTML='';card.style.display='none';return;}
 let items=[];try{items=await api('/api/quarantine?writer='+encodeURIComponent(w));}catch(e){}
 if(seq!==quarSeq)return; /* קריאה חדשה יותר כבר רצה */
 div.innerHTML='';
 if(!items.length){card.style.display='none';return;}
 card.style.display='block';
 for(const it of items){
  const row=document.createElement('div');row.className='line';
  const img=document.createElement('img');img.src='data:image/png;base64,'+it.img;
  img.onclick=()=>img.classList.toggle('zoom');
  const inp=document.createElement('textarea');inp.className='lt';inp.rows=1;inp.value=it.text;
  inp.addEventListener('input',()=>autoGrow(inp));
  const why=document.createElement('span');why.className='noteTag';why.textContent=it.reason;
  why.title=it.reason;why.style.maxWidth='230px';why.style.overflow='hidden';why.style.textOverflow='ellipsis';
  const ok=document.createElement('button');ok.className='btn green';ok.textContent='אשר';
  ok.onclick=async()=>{const r=await api('/api/quarantine_resolve',{writer:w,file:it.file,approve:true,text:inp.value});
   if(!r.ok)await appAlert(r.msg,'אישור שורה נכשל');refreshQuar();refreshAll(w);};
  const no=document.createElement('button');no.className='btn red';no.textContent='מחק';
  no.onclick=async()=>{await api('/api/quarantine_resolve',{writer:w,file:it.file,approve:false});refreshQuar();refreshAll(w);};
  row.appendChild(img);row.appendChild(inp);row.appendChild(why);row.appendChild(ok);row.appendChild(no);
  div.appendChild(row);requestAnimationFrame(()=>autoGrow(inp));}}

/* ---------- אימון ---------- */
async function train(scope,resume){const w=document.getElementById('writer').value;
 scope=scope||document.getElementById('trainScope').value;
 if(scope!=='general'&&!w){await appAlert('בחר כותב');return;}
 if(resume===false&&!await appConfirm('Start a fresh run from the base/general model? The interrupted checkpoint will be kept as a backup.'))return;
 const r=await api('/api/train',{writer:w,scope:scope,resume:resume!==false});
 if(!r.ok)await appAlert(r.msg,'האימון לא התחיל');else document.getElementById('trainIdle').textContent=r.msg;}
async function importGeneral(){const b64=await readFile('generalFile');
 if(!b64)return;
 document.getElementById('generalMsg').textContent='מעדכן...';
 const r=await api('/api/import_general',{file:b64});
 document.getElementById('generalMsg').textContent=r.msg;
 if(r.ok&&await appConfirm('המודל הכללי עודכן בהצלחה!\nהאם תרצה לעדכן את נתוני האימון האישיים שלך במודל הראשי של התוכנה? (מומלץ — כך המודל החדש ילמד גם את כתב היד שלך)')){
  train('general');}}
async function uploadPack(){const w=document.getElementById('writer').value;
 if(!w){await appAlert('בחר כותב');return;}
 if(!await appConfirm('לשלוח את נתוני האימון של "'+w+'" למפתח התוכנה?\nנשלחים רק נתונים שאתה יצרת (תמונות שורות והטקסט שלהן).'))return;
 document.getElementById('uploadMsg').textContent='שולח... (זה יכול לקחת דקה)';
 const r=await api('/api/upload_pack',{writer:w});
 document.getElementById('uploadMsg').textContent=r.msg;}
async function saveConfig(){const r=await api('/api/config',{upload_url:document.getElementById('uploadUrl').value.trim()});
 await appAlert('נשמר','ההגדרות נשמרו');}
async function deleteTrainData(){const w=document.getElementById('writer').value;
 const msg=document.getElementById('cleanMsg');
 if(!w){await appAlert('בחר כותב');return;}
 if(!await appConfirm('למחוק את קובצי האימון של "'+w+'"?\n\nלא מומלץ: הקבצים משמשים גיבוי ויידרשו לאימון מחדש אחרי עדכון מודל עתידי.\nהמודלים המאומנים לא יימחקו והתמלול ימשיך לעבוד כרגיל.'))return;
 if(await appConfirm('לשלוח את הנתונים למפתח לפני המחיקה?\n(מומלץ — כך הם ישפרו את המודל הכללי ולא ילכו לאיבוד)')){
  msg.textContent='שולח למפתח לפני המחיקה... (זה יכול לקחת דקה)';
  const u=await api('/api/upload_pack',{writer:w});
  if(!u.ok&&!await appConfirm('השליחה נכשלה: '+u.msg+'\nלמחוק בכל זאת?')){msg.textContent=u.msg;return;}}
 const r=await api('/api/delete_train_data',{writer:w});
 msg.textContent=r.msg;refreshAll(w);}
async function loadConfig(){try{const c=await api('/api/config');
 if(c.upload_url)document.getElementById('uploadUrl').value=c.upload_url;}catch(e){}}
loadConfig();
async function stopTrain(){const r=await api('/api/train_stop',{});
 if(!r.stopped)await appAlert('לא נמצא אימון רץ');}
function drawCer(cers){const el=document.getElementById('cerChart');
 if(!cers||!cers.length){el.textContent='אין נתונים עדיין';return;}
 const w=460,h=130,max=Math.max(...cers,1),n=cers.length;
 let pts=cers.map((c,i)=>((i/(Math.max(n-1,1)))*w)+','+(h-(c/max)*h*0.85-8)).join(' ');
 el.innerHTML=`<svg width="${w}" height="${h}" style="direction:ltr;max-width:100%">
  <polyline points="${pts}" fill="none" stroke="#1d4e89" stroke-width="2.5"/>
  <text x="4" y="14" font-size="12" fill="#888">${max.toFixed(1)}%</text>
  <text x="${w-105}" y="${h-8}" font-size="13" fill="#17805a" font-weight="bold">אחרון: ${cers[cers.length-1].toFixed(2)}%</text></svg>`;}
let wasRunning=false;
async function pollTrain(){try{
 const s=await api('/api/train_status');
 const live=document.getElementById('trainLive'),idle=document.getElementById('trainIdle');
 if(s.running||s.step){live.style.display='block';idle.style.display='none';
  document.getElementById('trainLabel').textContent=(s.running?'⏳ ':'✅ ')+(s.label||'');
  document.getElementById('trainPhase').textContent='שלב '+(s.phase||'');
  document.getElementById('trainBg').style.display=s.running?'inline-block':'none';
  const pct=s.total?Math.round(100*s.step/s.total):0;
  const bar=document.getElementById('trainBar');bar.style.width=pct+'%';
  bar.textContent=pct+'% ('+s.step+'/'+s.total+')';
  document.getElementById('trainLoss').textContent=s.loss!=null?s.loss:'—';
  document.getElementById('log').textContent=s.log||'';drawCer(s.cers);
  document.getElementById('cerTitle').textContent='גרף דיוק (CER) — '+(s.label||'');}
 else{live.style.display='none';idle.style.display='block';}
 document.getElementById('genBest').textContent=
  s.general_best!=null?('המודל הכללי — השגיאה המיטבית שנשמרה: '+s.general_best.toFixed(2)+'%'):'';
 if(wasRunning&&!s.running)refreshAll();
 wasRunning=!!s.running;
}catch(e){}
 setTimeout(pollTrain,3000);}

/* ---------- כללי ---------- */
async function refreshAll(keep){const ws=await api('/api/writers');
 const sel=document.getElementById('writer');const cur=keep||sel.value;sel.innerHTML='';
 for(const w of ws){const o=document.createElement('option');o.value=w.name;
  o.textContent=w.name+' ('+w.samples+')';sel.appendChild(o);}
 if(!ws.length){const o=document.createElement('option');o.value='';o.textContent='— אין כותבים —';sel.appendChild(o);}
 if(cur)sel.value=cur;
 const w=ws.find(x=>x.name===sel.value);
 document.getElementById('writerInfo').textContent=w?(w.trained?'מודל אישי מאומן ✓':'אין עדיין מודל אישי'):'';
 const tbl=document.getElementById('writersTable');
 tbl.innerHTML='<tr><th>שם</th><th>שורות מתויגות</th><th>מודל אישי</th><th>שגיאה מיטבית</th></tr>';
 for(const x of ws){const tr=document.createElement('tr');
  tr.innerHTML=`<td>${x.name}</td><td>${x.samples}</td><td>${x.trained?'✅':'—'}</td><td>${x.best_cer!=null?x.best_cer.toFixed(2)+'%':'—'}</td>`;
  tbl.appendChild(tr);}
 const ts=await api('/api/templates/list');const tsel=document.getElementById('tmplName');
 tsel.innerHTML='';for(const t of ts){const o=document.createElement('option');o.value=t;o.textContent=t;tsel.appendChild(o);}
 refreshQuar();loadFixStats();refreshExamples();}

let examplesOffset=0,examplesLimit=20,examplesWriter='';
async function refreshExamples(){
 const writer=document.getElementById('writer').value;
 const list=document.getElementById('examplesList');
 const summary=document.getElementById('examplesSummary');
 if(!writer){list.textContent='';summary.textContent='בחר כותב כדי להציג את דוגמאות האימון';return;}
 if(writer!==examplesWriter){examplesWriter=writer;examplesOffset=0;}
 const page=await api('/api/examples?writer='+encodeURIComponent(writer)+
  '&offset='+examplesOffset+'&limit='+examplesLimit);
 if(page.error){summary.textContent=page.error;return;}
 if(page.total>0&&examplesOffset>=page.total){
  examplesOffset=Math.floor((page.total-1)/examplesLimit)*examplesLimit;
  return refreshExamples();
 }
 if(page.total===0)examplesOffset=0;
 list.replaceChildren();
 summary.textContent=page.total+' דוגמאות שמורות';
 const pageNo=Math.floor(examplesOffset/examplesLimit)+1;
 const pageCount=Math.max(1,Math.ceil(page.total/examplesLimit));
 document.getElementById('examplesPage').textContent='עמוד '+pageNo+' מתוך '+pageCount;
 document.getElementById('examplesPrev').disabled=examplesOffset<=0;
 document.getElementById('examplesNext').disabled=examplesOffset+examplesLimit>=page.total;
 for(const item of page.items){
  const row=document.createElement('div');row.className='line';
  const image=document.createElement('img');image.src='data:image/png;base64,'+item.img;
  image.alt='דוגמת אימון '+item.file;image.onclick=()=>image.classList.toggle('zoom');
  const text=document.createElement('textarea');text.className='lt';text.rows=2;
  text.value=item.text;text.setAttribute('aria-label','תיוג דוגמת '+item.file);
  const filename=document.createElement('span');filename.className='muted';
  filename.textContent=item.file;
  const save=document.createElement('button');save.className='btn';save.textContent='שמור עריכה';
  save.onclick=async()=>{
   const result=await api('/api/examples/edit',{writer,file:item.file,text:text.value});
   if(!result.ok){await appAlert(result.msg||result.error||'העריכה נכשלה');return;}
   summary.textContent=result.msg;refreshAll(writer);
  };
  const remove=document.createElement('button');remove.className='btn red';remove.textContent='מחק דוגמה';
  remove.onclick=async()=>{
   if(!await appConfirm('להסיר לצמיתות את דוגמת האימון "'+item.file+'"?'))return;
   const result=await api('/api/examples/delete',{writer,file:item.file});
   if(!result.ok){await appAlert(result.msg||result.error||'המחיקה נכשלה');return;}
   summary.textContent=result.msg;refreshAll(writer);
  };
  row.append(image,text,filename,save,remove);list.appendChild(row);
 }
}
function pageExamples(direction){
 examplesOffset=Math.max(0,examplesOffset+direction*examplesLimit);
 refreshExamples();
}
async function addWriter(){const n=document.getElementById('newWriter').value.trim();if(!n)return;
 await api('/api/writers',{name:n});document.getElementById('newWriter').value='';refreshAll(n);}
async function deleteWriter(){const w=document.getElementById('writer').value;
 const msg=document.getElementById('delWriterMsg');
 if(!w){await appAlert('בחר כותב (למעלה, ליד הכותרת)');return;}
 if(!await appConfirm('⚠️ אזהרה חמורה ⚠️\n\nמחיקת הכותב "'+w+'" תמחק לצמיתות:\n• את כל נתוני האימון שלו (תמונות השורות והתיוגים)\n• את המודל האישי המאומן שלו (כולל גרסאות הארכיון)\n\nזה בלתי הפיך! להמשיך?'))return;
 if(await appConfirm('מומלץ מאוד: להוריד עכשיו גיבוי מלא של הכותב (חבילת אימון + המודל)?\n\nאישור = הורדת הגיבוי · ביטול = דילוג על הגיבוי')){
  location.href='/api/export_writer?name='+encodeURIComponent(w);
  await new Promise(r=>setTimeout(r,1500));
  if(!await appConfirm('הגיבוי החל לרדת (בדוק בתיקיית ההורדות).\n\nלמחוק את "'+w+'" סופית עכשיו?'))return;
 }else if(!await appConfirm('למחוק את "'+w+'" סופית בלי גיבוי?'))return;
 const r=await api('/api/delete_writer',{writer:w});
 msg.textContent=r.msg;if(!r.ok)await appAlert(r.msg,'מחיקת כותב נכשלה');
 refreshAll();}
async function makeTemplates(){const r=await api('/api/templates/make',{count:10});
 document.getElementById('tmplMsg').textContent='נוצרו: '+r.made.join(', ');refreshAll();}
async function makeCustom(){const t=document.getElementById('customText').value;
 if(!t.trim()){await appAlert('הקלד משפטים');return;}
 const r=await api('/api/templates/custom',{text:t});
 document.getElementById('customMsg').textContent='נוצרו: '+r.made.join(', ')+' — הורד ZIP והדפס';refreshAll();}
async function ingestTemplate(){const w=document.getElementById('writer').value;
 if(!w){await appAlert('בחר כותב');return;}
 const b64=await readFile('tmplFile');
 if(!b64)return;
 const r=await api('/api/ingest_template',{img:b64,template:document.getElementById('tmplName').value,writer:w});
 document.getElementById('ingestMsg').textContent=r.msg;refreshAll(w);}
async function parallelMatch(){const b64=await readFile('parFile');
 if(!b64)return;
 const r=await api('/api/parallel',{img:b64,text:document.getElementById('parText').value});
 lastLines.parLines=r.lines||[];document.getElementById('parLines').innerHTML='';
 document.getElementById('parLines').dataset.lastSrc='';
 appendLines('parLines',lastLines.parLines,0);
 document.getElementById('parCard').style.display='block';
 document.getElementById('parMsg').textContent=r.note||'';}
async function exportPack(){const w=document.getElementById('writer').value;
 if(!w){await appAlert('בחר כותב');return;}
 location.href='/api/export_writer?name='+encodeURIComponent(w);}
async function importPack(){const w=document.getElementById('packWriter').value.trim();
 if(!w){await appAlert('תן שם לכותב');return;}
 const b64=await readFile('packFile');
 if(!b64)return;
 const r=await api('/api/import_pack',{zip:b64,writer:w});
 document.getElementById('packMsg').textContent=r.msg;refreshAll(w);}
async function importModel(){const n=document.getElementById('modelName').value.trim();
 if(!n){await appAlert('תן שם');return;}
 const b64=await readFile('modelFile');
 if(!b64)return;
 const r=await api('/api/import_model',{file:b64,name:n});
 document.getElementById('modelMsg').textContent=r.msg;refreshAll();}
refreshAll();pollTrain();
loadAIKeys();loadAISettings();
</script></body></html>"""


# ---------------------------------------------------------------- שרת
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _bytes(self, data, ctype, filename=None):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        # בלי מטמון — שכל פתיחה תקבל את הגרסה העדכנית של הממשק
        self.send_header("Cache-Control", "no-store, must-revalidate")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n)) if n else {}

    def do_GET(self):
        if urlparse(self.path).path == "/":
            self._bytes(HTML.encode(), "text/html; charset=utf-8")
        elif urlparse(self.path).path == "/favicon.ico":
            with open(os.path.join(BUNDLE, "icon.ico"), "rb") as f:
                self._bytes(f.read(), "image/x-icon")
        elif self.path == "/api/writers":
            self._json(list_writers())
        elif urlparse(self.path).path == "/api/examples":
            try:
                q = parse_qs(urlparse(self.path).query)
                writer = records.validate_writer(q.get("writer", [""])[0])
                page = records.list_examples(
                    writer, q.get("offset", [0])[0], q.get("limit", [20])[0])
                items = []
                for item in page["items"]:
                    with Image.open(item["image_path"]) as image:
                        image = image.convert("L")
                        image.thumbnail((420, 120))
                        items.append({"file": item["file"], "text": item["text"],
                                      "img": img_to_b64(image)})
                page["items"] = items
                self._json(page)
            except (ValueError, OSError) as exc:
                self._json({"error": str(exc)}, 400)
        elif self.path == "/api/train_status":
            self._json(train_status())
        elif self.path == "/api/ai_keys":
            self._json({"keys": ai_correction.list_keys()})
        elif self.path == "/api/ai_settings":
            self._json(ai_correction.get_settings())
        elif self.path == "/api/templates/list":
            self._json(list_templates())
        elif self.path == "/api/config":
            self._json(load_config())
        elif self.path.startswith("/api/fixes_stats"):
            q = parse_qs(urlparse(self.path).query)
            self._json(fixes.stats(unquote(q.get("writer", [""])[0]) or None))
        elif self.path.startswith("/api/export_fixes"):
            q = parse_qs(urlparse(self.path).query)
            w = unquote(q.get("writer", [""])[0]) or None
            data = fixes.export_tsv(w).encode("utf-8")
            if not data:
                self._json({"error": "אין עדיין תיקונים שנלמדו"}, 404)
            else:
                self._bytes(data, "text/tab-separated-values; charset=utf-8",
                            "fixes.tsv")
        elif self.path.startswith("/api/quarantine"):
            q = parse_qs(urlparse(self.path).query)
            w = unquote(q.get("writer", [""])[0])
            items = []
            for it in gates.quarantine_list(w):
                p = os.path.join(WRITERS_DIR, w, "quarantine", it["file"])
                try:
                    items.append({**it, "img": img_to_b64(Image.open(p))})
                except Exception:
                    continue
            self._json(items)
        elif self.path == "/api/templates/zip":
            self._bytes(templates_zip_bytes(), "application/zip", "templates.zip")
        elif self.path.startswith("/api/export_writer"):
            q = parse_qs(urlparse(self.path).query)
            data = export_writer_zip(unquote(q.get("name", [""])[0]))
            if data is None:
                self._json({"error": "כותב לא נמצא"}, 404)
            else:
                self._bytes(data, "application/zip", "training_pack.zip")
        elif self.path.startswith("/api/export_model"):
            q = parse_qs(urlparse(self.path).query)
            which = unquote(q.get("which", ["general"])[0])
            path = {"general": os.path.join(MODELS, "general.pt"),
                    "typed": os.path.join(MODELS, "typed.pt"),
                    "base": os.path.join(MODELS, "base_synth.pt")}.get(
                        which, os.path.join(MODELS, f"writer_{which}.pt"))
            if not os.path.exists(path):
                self._json({"error": "המודל המבוקש לא קיים עדיין"}, 404)
            else:
                with open(path, "rb") as f:
                    self._bytes(f.read(), "application/octet-stream", "model.pt")
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        try:
            body = self._body()
            if self.path == "/api/writers":
                name = body.get("name", "").strip().replace("/", "").replace("\\", "")
                if name:
                    os.makedirs(os.path.join(WRITERS_DIR, name, "lines"), exist_ok=True)
                self._json({"ok": True})

            elif self.path == "/api/transcribe":
                job = start_transcribe(body["files"], body.get("page"),
                                       body.get("writer") or None,
                                       body.get("prefer"),
                                       body.get("fix", True))
                self._json({"job": job})

            elif self.path == "/api/transcribe_status":
                job = _jobs.get(body.get("job"))
                if not job:
                    self._json({"error": "משימה לא נמצאה"}, 404)
                    return
                frm = int(body.get("from", 0))
                with _jobs_lock:
                    new_lines = job["lines"][frm:]
                    models_used = list(job["models_used"])
                self._json({"pages_total": job["pages_total"],
                            "pages_done": job["pages_done"], "cur": job["cur"],
                            "progress": job.get("progress", 0),
                            "done": job["done"], "error": job["error"],
                            "model": job["model"], "models_used": models_used,
                            "lines": new_lines,
                            "suggs": job.get("suggs") if job["done"] else None})

            elif self.path == "/api/save":
                saved, quar, dups = save_lines(body["writer"], body["lines"])
                postfix.refresh_vocab()
                self._json({"saved": saved, "quarantined": quar, "dups": dups})

            elif self.path == "/api/examples/edit":
                if training_running():
                    self._json({"ok": False,
                                "msg": "אי אפשר לערוך דוגמאות בזמן אימון פעיל"}, 409)
                    return
                result = records.edit_example(
                    body["writer"], body["file"], body["text"])
                postfix.refresh_vocab()
                self._json({"ok": True, "result": result,
                            "msg": "הדוגמה עודכנה. עדכון המשקולות דורש אימון מחדש."})

            elif self.path == "/api/examples/delete":
                if training_running():
                    self._json({"ok": False,
                                "msg": "אי אפשר למחוק דוגמאות בזמן אימון פעיל"}, 409)
                    return
                result = records.delete_example(body["writer"], body["file"])
                postfix.refresh_vocab()
                self._json({"ok": True, "result": result,
                            "msg": "הדוגמה הוסרה מנתוני האימון. המשקולות שכבר אומנו "
                                   "אינן משתנות עד אימון מחדש או שחזור מודל."})

            elif self.path == "/api/ai_keys/add":
                keys = ai_correction.add_key(body.get("name"), body.get("key"))
                self._json({"ok": True, "keys": keys})

            elif self.path == "/api/ai_keys/remove":
                keys = ai_correction.remove_key(body.get("id", ""))
                self._json({"ok": True, "keys": keys})

            elif self.path == "/api/ai_keys/rename":
                key = ai_correction.rename_key(
                    body.get("id", ""), body.get("name", ""))
                self._json({"ok": True, "key": key})

            elif self.path == "/api/ai_keys/check":
                key_id = body.get("id")
                if key_id:
                    self._json({"result": ai_correction.check_key(key_id)})
                else:
                    self._json({"results": ai_correction.check_all_keys()})

            elif self.path == "/api/ai_keys/models":
                self._json(ai_correction.list_available_models(
                    body.get("id", "")))

            elif self.path == "/api/ai_keys/test_model":
                self._json({"result": ai_correction.test_model(
                    body.get("id", ""), body.get("model", ""))})

            elif self.path == "/api/ai_keys/model":
                key = ai_correction.set_key_model(
                    body.get("id", ""), body.get("model", ""))
                self._json({"ok": True, "key": key})

            elif self.path == "/api/ai_settings":
                settings = ai_correction.save_settings(body)
                self._json({"ok": True, "settings": settings})

            elif self.path == "/api/ai_settings/reset":
                self._json({"ok": True,
                            "settings": ai_correction.reset_settings()})

            elif self.path == "/api/ai_correct":
                self._json(ai_correction.correct_lines(
                    body.get("lines"), batch_size=body.get("batch_size"),
                    key_id=body.get("key_id"), key_name=body.get("key_name"),
                    temperature=body.get("temperature"),
                    top_p=body.get("top_p"), top_k=body.get("top_k"),
                    max_output_tokens=body.get("max_output_tokens"),
                    request_timeout=body.get("request_timeout")))

            elif self.path == "/api/save_fixes":
                # מודל התיקונים: הפקת זוגות (שגוי->נכון) מהעריכות של המשתמש
                pairs = []
                for l in body.get("lines", []):
                    orig = (l.get("orig") or "").strip()
                    final = (l.get("final") or "").strip()
                    if orig and final and orig != final:
                        pairs.extend(fixes.harvest_pairs(orig, final))
                n = fixes.add_pairs(pairs, body.get("writer") or None,
                                    body.get("scope", "both"))
                self._json({"ok": True, "added": n,
                            "msg": f"נלמדו {n} תיקונים ✓" if n else
                                   "לא נמצאו תיקונים חדשים (אין הבדלים ברי-למידה)"})

            elif self.path == "/api/suggest_page":
                # תיקון מורחב: ניתוח כל המסמך במקביל — אוצר המילים של כל
                # העמודים + רצף חוצה-שורות + איטרציית התכנסות (תיקון שהוחל
                # וירטואלית משפר את ההקשר של שכניו בסבב הבא). סף רגיש מעט
                # יותר מהמעבר האוטומטי — המשתמש ממילא מאשר כל הצעה ביד.
                w = body.get("writer") or None
                lines_in = body.get("lines", [])
                orig = [(l.get("text") or "").strip() for l in lines_in]
                srcs = [l.get("src") for l in lines_in]
                cur = list(orig)
                for _round in range(2):
                    counts = postfix.page_word_counts([t for t in cur if t])
                    changed, pt2, pt, prev_src = False, None, None, None
                    for idx, t in enumerate(cur):
                        if srcs[idx] != prev_src:
                            pt2, pt, prev_src = None, None, srcs[idx]
                        if not t:
                            continue
                        if lines_in[idx].get("typed"):
                            continue    # שורת צילום-מסך: ה-LM לא בדומיין שלה
                        ctx = counts - postfix.page_word_counts([t])
                        s_text, ch = postfix.suggest_line(
                            t, min_score=1.05, writer=w, page_words=ctx,
                            prev_tail=pt, prev_tail2=pt2)
                        if ch and s_text != t:
                            cur[idx] = s_text
                            changed = True
                        pt2, pt = postfix.advance_tail(pt2, pt, cur[idx])
                    if not changed:
                        break
                suggs = {i: s for i, (s, o) in enumerate(zip(cur, orig))
                         if s != o}
                self._json({"suggs": suggs})

            elif self.path == "/api/fixes_clear":
                # ניקוי תצוגה בלבד — הלמידה נשארת פעילה במלואה
                n = fixes.mark_seen(body.get("writer") or None)
                self._json({"ok": True,
                            "msg": "הרשימה נוקתה — כל התיקונים ממשיכים "
                                   f"לפעול ברקע ({n} פעילים)"})

            elif self.path == "/api/fixes_remove":
                n = fixes.remove(body["wrong"], body.get("writer") or None)
                self._json({"ok": n > 0,
                            "msg": f'התיקון "{body["wrong"]}" הוסר מהמודל'
                                   if n else "לא נמצא"})

            elif self.path == "/api/fixes_override":
                fixes.override(body["wrong"], body["right"],
                               body.get("writer") or None)
                self._json({"ok": True,
                            "msg": f'עודכן: {body["wrong"]} ← {body["right"]}'})

            elif self.path == "/api/import_fixes":
                text = base64.b64decode(body["file"]).decode("utf-8", "replace")
                n = fixes.import_tsv(text)
                self._json({"ok": n > 0,
                            "msg": f"מוזגו {n} תיקונים למאגר הכללי ✓" if n else
                                   "לא נמצאו תיקונים תקינים בקובץ"})

            elif self.path == "/api/quarantine_resolve":
                ok, msg = gates.quarantine_resolve(body["writer"], body["file"],
                                                   bool(body.get("approve")),
                                                   body.get("text"))
                if ok:
                    postfix.refresh_vocab()
                self._json({"ok": ok, "msg": msg})

            elif self.path == "/api/train":
                ok, msg = start_training(body.get("writer") or "",
                                         body.get("scope", "both"),
                                         bool(body.get("resume", True)))
                self._json({"ok": ok, "msg": msg})

            elif self.path == "/api/import_general":
                tmp = os.path.join(MODELS, "_general_check.pt")
                with open(tmp, "wb") as f:
                    f.write(base64.b64decode(body["file"]))
                try:
                    load_checkpoint(tmp, NUM_CLASSES, CHARSET)
                except Exception as e:
                    os.remove(tmp)
                    self._json({"ok": False, "msg": f"קובץ המודל אינו תואם: {e}"})
                    return
                os.replace(tmp, os.path.join(MODELS, "general.pt"))
                _state["models"].pop("__general__", None)
                self._json({"ok": True,
                            "msg": "המודל הכללי עודכן ✓ (המודל האישי שלך לא השתנה)"})

            elif self.path == "/api/upload_pack":
                ok, msg = upload_pack_to_dev(body["writer"])
                self._json({"ok": ok, "msg": msg})

            elif self.path == "/api/delete_train_data":
                ok, msg = delete_training_data(body["writer"])
                self._json({"ok": ok, "msg": msg})

            elif self.path == "/api/delete_writer":
                ok, msg = delete_writer(body["writer"])
                self._json({"ok": ok, "msg": msg})

            elif self.path == "/api/config":
                self._json(save_config({"upload_url": body.get("upload_url")}))

            elif self.path == "/api/train_stop":
                self._json({"stopped": stop_training()})

            elif self.path == "/api/export_docx":
                paragraphs = [p for p in body.get("paragraphs", [])]
                self._bytes(build_docx(paragraphs),
                            "application/vnd.openxmlformats-officedocument"
                            ".wordprocessingml.document", "transcript.docx")

            elif self.path == "/api/export_docx_layout":
                self._bytes(build_docx_layout(body.get("lines", [])),
                            "application/vnd.openxmlformats-officedocument"
                            ".wordprocessingml.document", "transcript_layout.docx")

            elif self.path == "/api/import_pack":
                added, msg = import_writer_zip(body["zip"], body["writer"].strip())
                self._json({"ok": added > 0, "msg": msg})

            elif self.path == "/api/import_model":
                name, msg = import_model_file(body["file"], body.get("name", ""))
                self._json({"ok": name is not None, "msg": msg})

            elif self.path == "/api/templates/make":
                import make_template
                made = make_template.make_pages(int(body.get("count", 10))) or []
                self._json({"made": made})

            elif self.path == "/api/templates/custom":
                import make_template
                sentences = [clean(s) for s in body.get("text", "").splitlines()]
                self._json({"made": make_template.make_custom_pages(sentences)})

            elif self.path == "/api/ingest_template":
                os.makedirs(UPLOADS_DIR, exist_ok=True)
                tmp = os.path.join(UPLOADS_DIR, "_tmpl_upload.png")
                with open(tmp, "wb") as f:
                    f.write(base64.b64decode(body["img"]))
                n = ingest_template(tmp, body["template"], body["writer"])
                msg = (f"נקלטו {n} שורות בתיוג אוטומטי ✓" if n else
                       "מספר השורות לא תאם לתבנית — עברו למצב 'טקסט מקביל' לתיוג ידני")
                self._json({"ok": bool(n), "msg": msg})

            elif self.path == "/api/parallel":
                os.makedirs(UPLOADS_DIR, exist_ok=True)
                tmp = os.path.join(UPLOADS_DIR, "_par_upload.png")
                Image.open(io.BytesIO(base64.b64decode(body["img"]))).convert("L").save(tmp)
                texts = [clean(t.strip()) for t in body.get("text", "").splitlines() if t.strip()]
                imgs = segment_page(tmp)
                note = ""
                if len(imgs) != len(texts):
                    note = (f"זוהו {len(imgs)} שורות בסריקה מול {len(texts)} שורות טקסט "
                            f"— בדוק את ההצמדה בקפידה")
                lines = [{"img": img_to_b64(im),
                          "text": texts[i] if i < len(texts) else "", "src": None}
                         for i, im in enumerate(imgs)]
                self._json({"lines": lines, "note": note})

            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:
            self._json({"error": str(e)}, 500)


class LocalHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = False


def _create_server():
    """Use the preferred local port, then any OS-assigned free port."""
    for candidate in (*range(PORT, PORT + 10), 0):
        try:
            return LocalHTTPServer(("127.0.0.1", candidate), Handler)
        except OSError:
            continue
    raise OSError("לא ניתן לפתוח שרת מקומי בפורט פנוי")


def _show_startup_error(message):
    if os.name == "nt":
        ctypes.windll.user32.MessageBoxW(None, message, "כתב-יד AI", 0x10)


def _load_window_size():
    try:
        with open(WINDOW_SIZE_PATH, encoding="utf-8") as f:
            size = json.load(f)
        width, height = int(size["width"]), int(size["height"])
        if not (800 <= width <= 10000 and 600 <= height <= 6000):
            raise ValueError("saved window size is outside the allowed range")
        return width, height
    except FileNotFoundError:
        return DEFAULT_WINDOW_SIZE
    except (OSError, ValueError, TypeError, KeyError) as exc:
        import logging
        logging.getLogger(__name__).warning(
            "Could not load saved window size from %s: %s",
            WINDOW_SIZE_PATH, exc)
        return DEFAULT_WINDOW_SIZE


def _save_window_size(width, height):
    os.makedirs(DATA_DIR, exist_ok=True)
    temporary_path = WINDOW_SIZE_PATH + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as f:
        json.dump({"width": int(width), "height": int(height)}, f)
    os.replace(temporary_path, WINDOW_SIZE_PATH)


def _find_app_browser():
    candidates = [
        shutil.which("msedge"),
        shutil.which("msedge.exe"),
        shutil.which("chrome"),
        shutil.which("chrome.exe"),
    ]
    for root in (os.environ.get("PROGRAMFILES"),
                 os.environ.get("PROGRAMFILES(X86)"),
                 os.environ.get("LOCALAPPDATA")):
        if not root:
            continue
        candidates.extend((
            os.path.join(root, "Microsoft", "Edge", "Application", "msedge.exe"),
            os.path.join(root, "Google", "Chrome", "Application", "chrome.exe"),
        ))
    return next((path for path in candidates if path and os.path.isfile(path)), None)


def main():
    logger = logging.getLogger("ktavyadai.startup")
    os.makedirs(WRITERS_DIR, exist_ok=True)
    try:
        server = _create_server()
    except OSError as exc:
        _show_startup_error(f"לא ניתן להפעיל שרת מקומי.\n\n{exc}")
        raise
    url = f"http://127.0.0.1:{server.server_port}"
    logger.info("local HTTP server listening at %s", url)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        try:
            webview = importlib.import_module("webview")
        except ModuleNotFoundError as exc:
            if exc.name != "webview":
                raise
            browser = _find_app_browser()
            if browser is None:
                raise RuntimeError(
                    "לא נמצא Microsoft Edge או Google Chrome לפתיחת חלון "
                    "התוכנה. יש להתקין אחד מהם ולנסות שוב.")
            width, height = _load_window_size()
            logger.info("opening standalone browser app window with %s at %dx%d",
                        browser, width, height)
            subprocess.Popen(
                [browser, f"--app={url}", f"--window-size={width},{height}"])
            threading.Event().wait()
        else:
            width, height = _load_window_size()
            window = webview.create_window(
                f"כתב-יד AI {VERSION}", url, width=width, height=height,
                min_size=(800, 600))
            if window is None:
                raise RuntimeError("לא ניתן ליצור את חלון התוכנה")
            window.events.resized += _save_window_size
            logger.info("opening embedded interface at %dx%d", width, height)
            webview.start(gui="qt", icon=os.path.join(BUNDLE, "icon.ico"))
    except Exception as exc:
        _show_startup_error(f"לא ניתן להפעיל את הממשק המקומי.\n\n{exc}")
        raise
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
