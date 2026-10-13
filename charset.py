# -*- coding: utf-8 -*-
"""מפת תווים + טיפול בכיווניות (RTL) עבור מודל תמלול כתב יד עברי."""

HEBREW = "אבגדהוזחטיכךלמםנןסעפףצץקרשת"
DIGITS = "0123456789"
LATIN = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
PUNCT = " .,:;!?\"'()-/%+=*#@&_₪״׳"
# תווים שנוספו מאוחר יותר — חובה להוסיף רק בסוף כדי לא לשבור מודלים קיימים
EXTRA = "[]\u2013\\|<>{}~^$\u20ac\u00a3\u00a5\u00a9\u00ae\u00b0\u2026"
CHARSET = HEBREW + DIGITS + LATIN + PUNCT + EXTRA

BLANK = 0  # CTC blank
char_to_idx = {c: i + 1 for i, c in enumerate(CHARSET)}
idx_to_char = {i + 1: c for i, c in enumerate(CHARSET)}
NUM_CLASSES = len(CHARSET) + 1


# גרסאות טיפוגרפיות שקודם נמחקו בטעות — מנורמלות לתווים שהמודל מכיר.
# לא נוגעים ב-״/׳ העבריים (U+05F4/U+05F3) — הם מחלקות קיימות במודל.
_NORMALIZE = {"“": '"', "”": '"', "‘": "'", "’": "'",
              "־": "-", "—": "–"}


def clean(text):
    """מנרמל גרסאות טיפוגרפיות ומשאיר רק תווים שהמודל מכיר."""
    for s, d in _NORMALIZE.items():
        if s in text:
            text = text.replace(s, d)
    return "".join(c for c in text if c in char_to_idx)


def encode(text):
    return [char_to_idx[c] for c in text if c in char_to_idx]


def decode_ctc(indices):
    out, prev = [], BLANK
    for i in indices:
        if i != BLANK and i != prev:
            out.append(idx_to_char.get(i, ""))
        prev = i
    return "".join(out)


# ---------- כיווניות: המרה בין סדר לוגי (מחרוזת) לסדר חזותי (תמונה) ----------
# המודל קורא את התמונה משמאל לימין, לכן מאמנים אותו על הסדר החזותי,
# וממירים חזרה לסדר לוגי בפלט.

_LTR = set(DIGITS + LATIN)
_WEAK = set(".,:/-%_\\|<>~^$" + chr(8364) + chr(163) + chr(165)
            + chr(39) + chr(34))
_MIRROR = {
    "(": ")", ")": "(", "[": "]", "]": "[",
    "{": "}", "}": "{", "<": ">", ">": "<",
}


def _fix_runs(s):
    """הופך רצפי LTR (ספרות/לטינית) בתוך מחרוזת הפוכה, וממרר סוגריים."""
    out, i, n = [], 0, len(s)
    while i < n:
        if s[i] in _LTR:
            j = i + 1
            while j < n:
                if s[j] in _LTR:
                    j += 1
                elif s[j] in _WEAK and j + 1 < n and s[j + 1] in _LTR:
                    j += 2
                else:
                    break
            out.append(s[i:j][::-1])
            i = j
        else:
            out.append(_MIRROR.get(s[i], s[i]))
            i += 1
    return "".join(out)


def to_visual(logical):
    """סדר לוגי -> סדר חזותי (משמאל לימין כפי שנראה בתמונה)."""
    return _fix_runs(logical[::-1])


def to_logical(visual):
    """סדר חזותי -> סדר לוגי (מחרוזת עברית תקינה)."""
    return _fix_runs(visual)[::-1]


if __name__ == "__main__":
    tests = ["שלום עולם", "שילמתי 250 שקל", "פגישה ב-14:30 (חשוב!)", "קניתי iPhone חדש"]
    for t in tests:
        v = to_visual(t)
        back = to_logical(v)
        status = "OK" if back == t else "FAIL"
        print(f"[{status}] {t!r} -> visual {v!r} -> back {back!r}")
