# -*- coding: utf-8 -*-
"""טעינת נתונים לאימון: שורות סינתטיות שנוצרות תוך כדי ריצה + שורות אמיתיות מתויגות."""
import os
import random
import numpy as np
import torch
from PIL import Image

from charset import encode, clean, to_visual
from model import IMG_H, DOWNSAMPLE_W
import synth

from paths import WRITERS_DIR, SYNTH_CORPUS  # noqa: F401 — נתיבים מרכזיים

MAX_W = 1600


def load_line_image(path):
    """טוענת תמונת שורה אמיתית ומנרמלת לגובה IMG_H, טווח 0..1."""
    img = Image.open(path).convert("L")
    w, h = img.size
    new_w = max(16, int(w * IMG_H / h))
    img = img.resize((min(new_w, MAX_W), IMG_H), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def light_augment(arr, rng):
    """אוגמנטציה קלה לשורות אמיתיות (שלא נהרוס את הסגנון)."""
    a = arr.copy()
    if rng.random() < 0.5:
        a += rng.uniform(-0.08, 0.08)
    if rng.random() < 0.5:
        a += np.random.normal(0, rng.uniform(0.01, 0.04), a.shape).astype(np.float32)
    if rng.random() < 0.3:  # מתיחה אופקית קלה
        h, w = a.shape
        new_w = int(w * rng.uniform(0.9, 1.1))
        img = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
        a = np.array(img.resize((min(new_w, MAX_W), h), Image.BILINEAR),
                     dtype=np.float32) / 255.0
    return np.clip(a, 0, 1)


class RealLines:
    """שורות אמיתיות של כותב: data/writers/<name>/lines/*.png + labels.tsv"""
    def __init__(self, writer):
        self.dir = os.path.join(WRITERS_DIR, writer)
        self.items = []
        labels = os.path.join(self.dir, "labels.tsv")
        if os.path.exists(labels):
            with open(labels, encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) >= 2 and parts[1].strip():
                        p = os.path.join(self.dir, "lines", parts[0])
                        if os.path.exists(p):
                            self.items.append((p, clean(parts[1].strip())))

    def __len__(self):
        return len(self.items)

    def split(self, val_ratio=0.1, seed=42):
        idx = list(range(len(self.items)))
        random.Random(seed).shuffle(idx)
        n_val = max(1, int(len(idx) * val_ratio)) if len(idx) >= 5 else 0
        val = [self.items[i] for i in idx[:n_val]]
        train = [self.items[i] for i in idx[n_val:]]
        return train, val


def make_batch(samples):
    """samples: list of (np_img HxW, text_logical). מרפדת לרוחב אחיד ומקודדת CTC.
    דוגמה שהתמונה שלה צרה מכדי להכיל את התווית (שורה צפופה שנחתכה ברוחב)
    מדולגת — חיתוך התווית, כמו פעם, היה מאמן את המודל על תיוג שקרי."""
    kept = []
    for a, t in samples:
        if a is None:
            continue
        enc = encode(to_visual(t))
        if not enc:
            continue
        if min(a.shape[1], MAX_W) // DOWNSAMPLE_W < len(enc) * 2:
            continue
        kept.append((a, t))
    samples = kept
    if not samples:
        return None
    max_w = min(MAX_W, max(a.shape[1] for a, _ in samples))
    max_w = ((max_w + DOWNSAMPLE_W - 1) // DOWNSAMPLE_W) * DOWNSAMPLE_W
    B = len(samples)
    imgs = np.ones((B, 1, IMG_H, max_w), dtype=np.float32)
    input_lengths, targets, target_lengths = [], [], []
    for i, (a, text) in enumerate(samples):
        w = min(a.shape[1], max_w)
        imgs[i, 0, :, :w] = a[:, :w]
        enc = encode(to_visual(text))
        input_lengths.append(w // DOWNSAMPLE_W)
        targets.extend(enc)
        target_lengths.append(len(enc))
    return (torch.from_numpy(imgs), torch.tensor(targets, dtype=torch.long),
            torch.tensor(input_lengths, dtype=torch.long),
            torch.tensor(target_lengths, dtype=torch.long),
            [t for _, t in samples])


class MixedSampler:
    """מייצר אצוות: תערובת של שורות סינתטיות ושורות אמיתיות ביחס נתון."""
    def __init__(self, real_items=None, real_ratio=0.0, seed=None):
        self.rng = random.Random(seed)
        self.text_sampler = synth.TextSampler(SYNTH_CORPUS)
        self.real_items = real_items or []
        self.real_ratio = real_ratio if self.real_items else 0.0
        self._real_order = []
        self._real_pos = 0

    def _next_real(self):
        if self._real_pos >= len(self._real_order):
            self._real_order = list(range(len(self.real_items)))
            self.rng.shuffle(self._real_order)
            self._real_pos = 0
        index = self._real_order[self._real_pos]
        self._real_pos += 1
        return self.real_items[index]

    def state_dict(self):
        return {
            "rng": self.rng.getstate(),
            "real_order": self._real_order,
            "real_pos": self._real_pos,
        }

    def load_state_dict(self, state):
        self.rng.setstate(state["rng"])
        order = state.get("real_order", [])
        if (not isinstance(order, list)
                or any(not isinstance(index, int) or index < 0
                       or index >= len(self.real_items) for index in order)):
            raise ValueError("checkpoint contains an invalid real-sample order")
        self._real_order = order
        self._real_pos = min(max(0, int(state.get("real_pos", 0))), len(order))

    def sample_one(self):
        if self.real_items and self.rng.random() < self.real_ratio:
            path, text = self._next_real()
            try:
                arr = light_augment(load_line_image(path), self.rng)
                return arr, text
            except Exception:
                pass
        text = self.text_sampler.sample(self.rng)
        return synth.render_line(text, self.rng), text

    def batch(self, batch_size):
        while True:
            b = make_batch([self.sample_one() for _ in range(batch_size)])
            if b is not None:
                return b


class ScreenSampler:
    """אצוות טקסט-מסך סינתטי — למודל הטקסט המוקלד (typed). אין שורות
    אמיתיות: הרינדור עצמו הוא "האמת" (גופני המסך של Windows)."""
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.text_sampler = synth.TextSampler(SYNTH_CORPUS)

    def batch(self, batch_size):
        while True:
            samples = []
            for _ in range(batch_size):
                t = synth.sample_screen_text(self.text_sampler, self.rng)
                samples.append((synth.render_screen_line(t, self.rng), t))
            b = make_batch(samples)
            if b is not None:
                return b


def eval_batch_from_items(items, limit=160):
    """אצוות הערכה משורות אמיתיות (ללא אוגמנטציה). limit גבוה — שער
    הרגרסיה נמדד על כל סט האימות; מדגם 64 היה רועש (שורה אחת ≈ 0.2%)."""
    chosen = items[:limit]
    return make_batch([(load_line_image(p), t) for p, t in chosen])
