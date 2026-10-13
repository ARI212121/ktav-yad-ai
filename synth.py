# -*- coding: utf-8 -*-
"""מחולל שורות כתב יד סינתטיות בעברית, מבוסס על גופני כתב-יד המותקנים ב-Windows.
משמש לאימון-קדם של מודל הבסיס לפני כוונון על כתב יד אמיתי.
"""
import os
import glob
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from charset import EXTRA, clean, to_visual
from model import IMG_H

# גופנים מועדפים — כתב יד תחילה, אחר כך גופני דפוס לגיוון
PREFERRED_FONTS = [
    "Guttman Yad", "Guttman Yad-Brush", "Guttman Yad-Light", "Guttman Drogolin",
    "Guttman Kav", "Guttman Kav-Light", "Guttman Haim", "Guttman Hatzvi",
    "David", "FrankRuehl", "Miriam", "Narkisim", "Aharoni", "Levenim MT",
    "Guttman Aharoni", "Guttman Miryam", "Guttman Vilna", "Rod", "Hadassah Friedlaender",
    # גופני מסך/מסמכים — לתמיכה בתמלול טקסט מוקלד וצילומי מסך
    "Segoe UI", "Arial", "Calibri", "Tahoma",
]
# משקל גבוה יותר לגופני כתב יד אמיתיים
FONT_WEIGHTS = {"Guttman Yad": 6, "Guttman Yad-Brush": 6, "Guttman Yad-Light": 6,
                "Guttman Drogolin": 4, "Guttman Kav": 3, "Guttman Kav-Light": 3}

_font_cache = {}
_glyph_support_cache = {}
SPECIAL_SYMBOLS = set(EXTRA[3:])


def _discover_fonts():
    """ממפה שמות גופנים לקבצי TTF בתיקיית הגופנים של Windows."""
    found = {}
    fonts_dir = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    for path in glob.glob(os.path.join(fonts_dir, "*.tt[fc]")) + \
                glob.glob(os.path.join(fonts_dir, "*.otf")):
        try:
            name = ImageFont.truetype(path, 20).getname()[0]
        except Exception:
            continue
        found.setdefault(name, path)
    result = []
    for name in PREFERRED_FONTS:
        if name in found:
            result.extend([found[name]] * FONT_WEIGHTS.get(name, 1))
    if not result:
        raise RuntimeError("לא נמצאו גופנים עבריים")
    return result


def get_fonts():
    if "list" not in _font_cache:
        _font_cache["list"] = _discover_fonts()
    return _font_cache["list"]


def _load_font(path, size):
    key = (path, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(path, size)
    return _font_cache[key]


def _supports_symbols(path, symbols):
    font = _load_font(path, 32)
    missing = font.getmask(chr(0x10ffff))
    for symbol in symbols:
        key = (path, symbol)
        if key not in _glyph_support_cache:
            glyph = font.getmask(symbol)
            _glyph_support_cache[key] = not (
                glyph.size == missing.size and bytes(glyph) == bytes(missing))
        if not _glyph_support_cache[key]:
            return False
    return True


def render_line(text, rng=None):
    """מציירת שורת טקסט (בסדר לוגי) כתמונת גווני אפור בגובה IMG_H.
    מחזירה numpy array של float32 בטווח 0..1 (דיו כהה על רקע בהיר)."""
    rng = rng or random
    visual = to_visual(clean(text))
    if not visual.strip():
        visual = "א"
    size = rng.randint(30, 46)
    symbols = set(visual).intersection(SPECIAL_SYMBOLS)
    font_paths = get_fonts()
    candidates = ([path for path in font_paths if _supports_symbols(path, symbols)]
                  if symbols else font_paths)
    if not candidates:
        raise RuntimeError("No installed font supports all symbols in this sample")
    font = _load_font(rng.choice(candidates), size)

    # ציור מילה-מילה עם רעידות מיקום — מדמה חוסר אחידות של כתב יד
    pad = size
    est_w = int(len(visual) * size * 0.8) + pad * 2
    canvas = Image.new("L", (est_w, size * 3), 255)
    draw = ImageDraw.Draw(canvas)
    x = pad
    baseline = size
    ink = rng.randint(0, 90)
    for word in visual.split(" "):
        if word:
            dy = rng.randint(-3, 3)
            draw.text((x, baseline + dy), word, font=font, fill=ink)
            x += int(draw.textlength(word, font=font))
        x += int(size * rng.uniform(0.2, 0.45))  # רוחב רווח אקראי
    # חיתוך לתוכן
    arr = np.array(canvas)
    ys, xs = np.where(arr < 200)
    if len(xs) == 0:
        return None
    m = 6
    img = canvas.crop((max(0, xs.min() - m), max(0, ys.min() - m),
                       min(canvas.width, xs.max() + m), min(canvas.height, ys.max() + m)))

    # הטיה (slant) אקראית — מאפיין מרכזי של כתב יד
    shear = rng.uniform(-0.25, 0.25)
    w, h = img.size
    img = img.transform((w + int(abs(shear) * h), h), Image.AFFINE,
                        (1, shear, -shear * h if shear > 0 else 0, 0, 1, 0),
                        resample=Image.BILINEAR, fillcolor=255)
    return augment(img, rng)


def augment(img, rng=None):
    """אוגמנטציות: טשטוש, רעש, עובי קו, ניגודיות — ואז נרמול לגובה קבוע."""
    rng = rng or random
    if rng.random() < 0.12:
        # הדמיית צילום מסך / טקסט מוקלד קטן: הקטנה חדה וניפוח חזרה —
        # מאבדת פרטים עדינים בדיוק כמו גופן מסך של 16px שמנורמל ל-48
        small_h = rng.randint(14, 24)
        w0, h0 = img.size
        img = img.resize((max(8, int(w0 * small_h / h0)), small_h), Image.BILINEAR)
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    if rng.random() < 0.25:
        img = img.filter(ImageFilter.MinFilter(3))  # קו עבה יותר
    elif rng.random() < 0.2:
        img = img.filter(ImageFilter.MaxFilter(3))  # קו דק יותר
    # נרמול גובה
    w, h = img.size
    new_w = max(16, int(w * IMG_H / h))
    img = img.resize((min(new_w, 2000), IMG_H), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    # רקע לא אחיד + רעש
    if rng.random() < 0.5:
        arr += rng.uniform(-25, 10)
    if rng.random() < 0.6:
        arr += np.random.normal(0, rng.uniform(2, 10), arr.shape)
    arr = np.clip(arr, 0, 255) / 255.0
    return arr


# ---------------------------------------------------- טקסט מוקלד / צילום מסך
SCREEN_FAMILIES = {"Segoe UI", "Arial", "Calibri", "Tahoma", "David",
                   "Levenim MT", "Miriam", "Narkisim", "FrankRuehl"}


def get_screen_fonts():
    """קובצי גופני מסך על כל משקליהם (Regular/Bold/Semibold) — המפה הרגילה
    שומרת קובץ אחד למשפחה ומאבדת את המודגשים, שחיוניים לצילומי מסך."""
    if "screen" not in _font_cache:
        fonts_dir = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
        out = []
        for path in glob.glob(os.path.join(fonts_dir, "*.tt[fc]")):
            try:
                f = ImageFont.truetype(path, 20)
                fam, style = f.getname()
            except Exception:
                continue
            if fam not in SCREEN_FAMILIES or "italic" in style.lower():
                continue
            # חובה גליפים עבריים אמיתיים: גופן בלי עברית (Calibri) מרנדר
            # ריבועי tofu — א' וב' יוצאות זהות, וזה מרעיל את האימון
            try:
                a = np.array(f.getmask("א"), dtype=np.uint8)
                b = np.array(f.getmask("ב"), dtype=np.uint8)
                if a.size == 0 or (a.shape == b.shape and (a == b).all()):
                    continue
            except Exception:
                continue
            out.append(path)
        _font_cache["screen"] = out or get_fonts()
    return _font_cache["screen"]


# אוצר מילים ממשקי/צ'אטי — עברית מודרנית + אנגלית טכנית, לגיוון שורות מסך
_UI_HE = ("אני אתה זה לא כן מה איך למה מתי איפה עכשיו אחר כך תודה בסדר "
          "תוכנה קובץ חלון לחצן כפתור הגדרות שמירה פתח סגור הפעל עצור המשך "
          "בדוק תקין שגיאה הודעה שורה עמוד תמונה טקסט מודל אימון נתונים "
          "רשימה טבלה חיפוש עזרה גרסה עדכון התקנה הורדה העלאה שלח קבל "
          "לחץ גלול בחר סמן מחק הוסף ערוך שנה תגיד ראה תראה נמשיך נתקן "
          "צריך אפשר חייב כדאי מומלץ פשוט מהר לאט טוב נכון שגוי למעלה למטה "
          "ימינה שמאלה בפינה לשונית כרטיס גרף מצב רואה אותו מחדש נסגר פותח").split()
_UI_EN = ("server client error file windows update click open close save "
          "start stop run test model data text line page image button menu "
          "settings options help version install download upload send local "
          "the is on off of and or not yes no ok done ready alive still "
          "python torch app main log status port host user admin new").split()


def sample_screen_text(ts, rng=None):
    """שורת טקסט בסגנון מסך: קורפוס / עברית ממשקית / מעורב אנגלית / אנגלית."""
    rng = rng or random
    r = rng.random()
    if r < 0.5:
        return ts.sample(rng)
    if r < 0.75:   # עברית ממשקית/צ'אטית עם פיסוק אופייני
        n = rng.randint(3, 10)
        ws = [rng.choice(_UI_HE) for _ in range(n)]
        if rng.random() < 0.3:
            ws.insert(rng.randint(1, max(1, len(ws) - 1)), rng.choice(["-", "–"]))
        if rng.random() < 0.25:
            i = rng.randint(0, len(ws) - 1)
            ws[i] = '"' + ws[i] + '"'
        line = " ".join(ws)
        if rng.random() < 0.3:
            line = f"{rng.randint(1, 9)}. " + line
        if rng.random() < 0.35:
            line += rng.choice([":", ".", "?", "!"])
        return line
    if r < 0.9:    # משולב עברית-אנגלית-מספרים (הודעות טכניות)
        ws = []
        for _ in range(rng.randint(3, 9)):
            k = rng.random()
            if k < 0.55:
                ws.append(rng.choice(_UI_HE))
            elif k < 0.85:
                ws.append(rng.choice(_UI_EN))
            else:
                ws.append(str(rng.randint(0, 9999)))
        return " ".join(ws)
    return " ".join(rng.choice(_UI_EN) for _ in range(rng.randint(2, 8)))


def render_screen_line(text, rng=None):
    """שורת טקסט מוקלד כמו בצילום מסך: גופן מסך בגודל אמיתי (11-22px),
    בלי רעידות/הטיית כתב-יד, רקע בגוני בועת-צ'אט, ריכוך anti-aliasing —
    ואז נרמול לגובה המודל (48) כמו בצינור האמיתי."""
    rng = rng or random
    visual = to_visual(clean(text))
    if not visual.strip():
        visual = "א"
    px = rng.randint(11, 22)
    symbols = set(visual).intersection(SPECIAL_SYMBOLS)
    font_paths = get_screen_fonts()
    candidates = ([path for path in font_paths if _supports_symbols(path, symbols)]
                  if symbols else font_paths)
    if not candidates:
        raise RuntimeError("No installed screen font supports all symbols in this sample")
    font = _load_font(rng.choice(candidates), px)
    bg = rng.choice([255, 252, 248, 244, 240, 236])
    ink = rng.randint(0, 70)
    pad = max(3, px // 3)
    est_w = int(len(visual) * px * 0.75) + pad * 4
    canvas = Image.new("L", (est_w, px * 2 + pad * 2), bg)
    draw = ImageDraw.Draw(canvas)
    draw.text((pad * 2, pad), visual, font=font, fill=ink)
    arr = np.array(canvas)
    ys, xs = np.where(arr < bg - 40)
    if len(xs) == 0:
        return None
    m = max(2, px // 5)
    img = canvas.crop((max(0, xs.min() - m), max(0, ys.min() - m),
                       min(canvas.width, xs.max() + m),
                       min(canvas.height, ys.max() + m)))
    if rng.random() < 0.5:   # ריכוך ClearType-י קל
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.7)))
    w, h = img.size
    new_w = max(16, int(w * IMG_H / h))
    img = img.resize((min(new_w, 2000), IMG_H), Image.BILINEAR)
    out = np.array(img, dtype=np.float32)
    if rng.random() < 0.4:
        out += np.random.normal(0, rng.uniform(1, 4), out.shape)
    return np.clip(out, 0, 255) / 255.0


class TextSampler:
    """מקור טקסט לאימון: שורות מהקורפוס + צירופים אקראיים + מספרים."""
    def __init__(self, corpus_path):
        with open(corpus_path, encoding="utf-8") as f:
            self.lines = [clean(l.strip()) for l in f if l.strip()]
        self.words = sorted({w for l in self.lines for w in l.split() if w})

    def sample(self, rng=None):
        rng = rng or random
        if rng.random() < 0.08:
            symbols = [
                f"invoice_{rng.randint(1, 9999)}.pdf",
                f"price: {rng.randint(1, 9999)} \u20aa",
                f"cost: $ {rng.randint(1, 9999)} / {rng.randint(1, 99)}",
                r"path: C:\data\page_1.txt",
                "A|B < C > D {x} [y]",
                f"\u05ea\u05e4\u05e7\u05d9\u05d3: {'_' * rng.randint(8, 24)}",
                f"\u05e9\u05dd \u05de\u05d5\u05e8\u05e9\u05d4: {'_' * rng.randint(8, 24)}",
                "user@example.com #100",
                "EUR \u20ac GBP \u00a3 JPY \u00a5",
            ]
            return rng.choice(symbols)
        r = rng.random()
        if r < 0.55:  # שורה אמיתית מהקורפוס (או חלק ממנה)
            line = rng.choice(self.lines)
            ws = line.split()
            if len(ws) > 8 and rng.random() < 0.5:
                i = rng.randint(0, len(ws) - 4)
                line = " ".join(ws[i:i + rng.randint(3, 8)])
            return line
        elif r < 0.85:  # סלט מילים אקראי, לעיתים עם סוגריים וקו מפריד
            n = rng.randint(2, 8)
            ws = [rng.choice(self.words) for _ in range(n)]
            if len(ws) >= 3 and rng.random() < 0.25:
                i = rng.randint(0, len(ws) - 2)
                op, cl = rng.choice([("[", "]"), ("(", ")")])
                ws[i] = op + ws[i]
                ws[min(len(ws) - 1, i + rng.randint(0, 2))] += cl
            if rng.random() < 0.1:
                ws.insert(rng.randint(1, len(ws) - 1), "–")
            return " ".join(ws)
        elif r < 0.95:  # מספרים, תאריכים, שעות
            kind = rng.random()
            if kind < 0.35:
                return f"{rng.randint(1,31)}/{rng.randint(1,12)}/{rng.randint(2020,2027)}"
            if kind < 0.6:
                return f"{rng.choice(self.words)} {rng.randint(0,23)}:{rng.randint(0,59):02d}"
            return f"{rng.randint(1,99999)} {rng.choice(['שקל','אחוז','פריטים','ק"מ'])}"
        else:  # תווים בודדים / פיסוק
            n = rng.randint(1, 4)
            return " ".join(rng.choice(self.words) + rng.choice(".,!?:") for _ in range(n))


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    fonts = get_fonts()
    print(f"נמצאו {len(set(fonts))} קבצי גופן")
    ts = TextSampler(os.path.join(here, "data", "corpus_he.txt"))
    out = os.path.join(here, "data", "synth_preview")
    os.makedirs(out, exist_ok=True)
    for i in range(12):
        t = ts.sample()
        arr = render_line(t)
        if arr is None:
            continue
        Image.fromarray((arr * 255).astype(np.uint8)).save(os.path.join(out, f"sample_{i:02d}.png"))
        print(f"{i}: {t}")
    print(f"דוגמאות נשמרו ב-{out}")
