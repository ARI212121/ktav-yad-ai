# -*- coding: utf-8 -*-
"""אימון המודל.

שימוש:
  python train.py base                     אימון-קדם על נתונים סינתטיים (מודל בסיס)
  python train.py writer <שם>              כוונון עדין על כתב יד של כותב מסוים
אפשרויות:
  --steps N        מספר צעדי אימון (ברירת מחדל: base=30000, writer=2000)
  --batch N        גודל אצווה (ברירת מחדל 24)
  --lr X           קצב למידה (ברירת מחדל: base=3e-4, writer=1e-4)
  --resume         המשך מהצ'קפוינט האחרון של אותו יעד
"""
import os
import sys
import json
import time
import random
import shutil
import argparse
import numpy as np
import torch
import torch.nn as nn

from charset import CHARSET, NUM_CLASSES, decode_ctc, to_logical, to_visual
from model import CRNN, save_checkpoint, load_checkpoint
from dataset import (MixedSampler, RealLines, ScreenSampler,
                     eval_batch_from_items, make_batch, WRITERS_DIR)
import synth

from paths import MODELS_DIR as MODELS
os.makedirs(MODELS, exist_ok=True)
BASE_CKPT = os.path.join(MODELS, "base_synth.pt")
HISTORY_DIR = os.path.join(MODELS, "history")
STOP_FILE = os.path.join(MODELS, "train.stop")

# אותו קוד משרת את שני מסלולי החומרה שבתרשים: עם GPU (מסלול A) אימון מואץ,
# בלעדיו (מסלול B) הכול על המעבד — אין שתי מערכות נפרדות.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred, truth):
    if not truth:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, truth) / len(truth)


@torch.no_grad()
def evaluate(model, batch):
    model.eval()
    imgs, _, _, _, texts = batch
    logits = model(imgs.to(DEVICE))
    preds = logits.argmax(-1).cpu()
    total = 0.0
    for i, truth in enumerate(texts):
        visual_pred = decode_ctc(preds[i].tolist())
        total += cer(to_logical(visual_pred), truth)
    model.train()
    return total / len(texts)


def make_synth_val(n=48, seed=1234):
    sampler = MixedSampler(seed=seed)
    return sampler.batch(n)


GENERAL_CKPT = os.path.join(MODELS, "general.pt")
TYPED_CKPT = os.path.join(MODELS, "typed.pt")


def make_screen_val(n=64, seed=777):
    """סט אימות קבוע למודל המוקלד — סינתזת מסך בזרע קבוע (בר-השוואה בין ריצות)."""
    return ScreenSampler(seed=seed).batch(n)


# ------------------------- סט אימות קבוע (held-out) — הבסיס לשער הרגרסיה
def _val_split_path(writer):
    return os.path.join(WRITERS_DIR, writer, "val_split.json")


def load_persistent_split(writer, items):
    """חלוקת אימון/אימות קבועה: קובצי האימות נרשמים ב-val_split.json של הכותב
    ונשארים באימות לתמיד — כך אין דליפה בין הריצות, וה-CER בין אימונים בר-השוואה
    (דרישת שער 4 בתרשים). קבצים חדשים מצטרפים לאימון, ורק אם חלק האימות ירד
    מתחת ל-10% מוגרלים אליו קבצים חדשים."""
    by_name = {os.path.basename(p): (p, t) for p, t in items}
    names = sorted(by_name)
    val_names = []
    try:
        with open(_val_split_path(writer), encoding="utf-8") as f:
            val_names = [n for n in json.load(f) if n in by_name]
    except (OSError, ValueError):
        pass
    target = max(1, len(names) // 10) if len(names) >= 5 else 0
    if len(val_names) < target:
        pool = [n for n in names if n not in set(val_names)]
        random.Random(42).shuffle(pool)
        val_names += pool[:target - len(val_names)]
        try:
            with open(_val_split_path(writer), "w", encoding="utf-8") as f:
                json.dump(sorted(val_names), f, ensure_ascii=False, indent=0)
        except OSError:
            pass
    val_set = set(val_names)
    train_items = [by_name[n] for n in names if n not in val_set]
    val_items = [by_name[n] for n in sorted(val_set)]
    return train_items, val_items


def _all_writers_split():
    """איחוד החלוקות הקבועות של כל הכותבים — למודל הכללי."""
    train_all, val_all = [], []
    if os.path.isdir(WRITERS_DIR):
        for name in sorted(os.listdir(WRITERS_DIR)):
            items = RealLines(name).items
            if not items:
                continue
            tr, va = load_persistent_split(name, items)
            train_all.extend(tr)
            val_all.extend(va)
    return train_all, val_all


# ------------------------- גרסאות מודל (history) ושחזור
def _backup_to_history(ckpt_path, log_name, keep=2):
    """לפני דריסת מודל קיים — מגבה אותו ל-models/history (עד keep גרסאות)."""
    if not os.path.exists(ckpt_path):
        return
    os.makedirs(HISTORY_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    shutil.copy2(ckpt_path, os.path.join(HISTORY_DIR, f"{log_name}__{stamp}.pt"))
    olds = sorted(f for f in os.listdir(HISTORY_DIR)
                  if f.startswith(f"{log_name}__") and f.endswith(".pt"))
    for f in olds[:-keep]:
        try:
            os.remove(os.path.join(HISTORY_DIR, f))
        except OSError:
            pass


def _ckpt_for(name):
    if name in ("general", "__general__"):
        return GENERAL_CKPT, "general"
    if name == "base":
        return BASE_CKPT, "base"
    if name in ("typed", "__typed__"):
        return TYPED_CKPT, "typed"
    return os.path.join(MODELS, f"writer_{name}.pt"), name


def rollback(name):
    """משחזר את הגרסה האחרונה של מודל מהארכיון (history)."""
    ckpt_path, log_name = _ckpt_for(name)
    if not os.path.isdir(HISTORY_DIR):
        print("אין תיקיית history — אין גרסאות קודמות")
        return False
    versions = sorted(f for f in os.listdir(HISTORY_DIR)
                      if f.startswith(f"{log_name}__") and f.endswith(".pt"))
    if not versions:
        print(f"אין גרסה קודמת של '{log_name}' בארכיון")
        return False
    src = os.path.join(HISTORY_DIR, versions[-1])
    if os.path.exists(ckpt_path):
        # שומר את הנוכחי בצד לפני השחזור, ליתר ביטחון
        shutil.copy2(ckpt_path, ckpt_path + ".pre_rollback")
    shutil.copy2(src, ckpt_path)
    print(f"שוחזר: {ckpt_path} <- {versions[-1]}")
    return True


def train(target, steps, batch_size, lr, resume=False):
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    is_base = target == "base"
    is_general = target == "__general__"
    is_typed = target == "__typed__"
    if is_base:
        ckpt_path = BASE_CKPT
        sampler = MixedSampler(seed=None)
        val_batch = make_synth_val()
        start_from = ckpt_path if (resume and os.path.exists(ckpt_path)) else None
    elif is_typed:
        # מודל טקסט מוקלד/צילומי מסך: סינתזת מסך בלבד, יוצא ממודל הבסיס.
        # מסלול נפרד לגמרי ממודלי כתב-היד — אין שום סיכון רגרסיה אליהם.
        ckpt_path = TYPED_CKPT
        sampler = ScreenSampler(seed=None)
        val_batch = make_screen_val()
        if resume and os.path.exists(ckpt_path):
            start_from = ckpt_path
        elif os.path.exists(BASE_CKPT):
            start_from = BASE_CKPT
        else:
            start_from = None
    elif is_general:
        ckpt_path = GENERAL_CKPT
        train_items, val_items = _all_writers_split()
        if len(train_items) + len(val_items) < 20:
            print(f"שגיאה: רק {len(train_items) + len(val_items)} שורות בכל הכותבים יחד"
                  " — צריך לפחות 20 למודל כללי")
            sys.exit(1)
        print(f"מודל כללי: {len(train_items)} שורות אימון, {len(val_items)} אימות קבוע")
        sampler = MixedSampler(real_items=train_items, real_ratio=0.7)
        val_batch = eval_batch_from_items(val_items) if val_items else make_synth_val()
        if resume and os.path.exists(ckpt_path):
            start_from = ckpt_path
        elif os.path.exists(BASE_CKPT):
            start_from = BASE_CKPT
        else:
            start_from = None
    else:
        writer_dir = os.path.join(WRITERS_DIR, target)
        ckpt_path = os.path.join(MODELS, f"writer_{target}.pt")
        real = RealLines(target)
        if len(real) < 5:
            print(f"שגיאה: נמצאו רק {len(real)} שורות מתויגות עבור '{target}'. "
                  f"צריך לפחות 5 (מומלץ 100+) בתיקייה {writer_dir}")
            sys.exit(1)
        train_items, val_items = load_persistent_split(target, real.items)
        print(f"כותב '{target}': {len(train_items)} שורות אימון, {len(val_items)} אימות קבוע")
        # 30% סינתטי כדי שהמודל לא ישכח את הידע הכללי
        sampler = MixedSampler(real_items=train_items, real_ratio=0.7)
        val_batch = eval_batch_from_items(val_items) if val_items else make_synth_val()
        if resume and os.path.exists(ckpt_path):
            start_from = ckpt_path
        elif os.path.exists(GENERAL_CKPT):
            start_from = GENERAL_CKPT  # כותב חדש יוצא מהמודל הכללי אם קיים
        elif os.path.exists(BASE_CKPT):
            start_from = BASE_CKPT
        else:
            print("אזהרה: אין מודל בסיס (models/base_synth.pt) — מאמן מאפס. מומלץ קודם: python train.py base")
            start_from = None

    resume_path = ckpt_path + ".last"
    if not resume and os.path.isfile(resume_path):
        archived_resume = resume_path + ".previous"
        os.replace(resume_path, archived_resume)
    resume_latest = bool(resume and os.path.isfile(resume_path))
    if resume_latest:
        start_from = resume_path
    ckpt = {}
    if start_from:
        model, ckpt = load_checkpoint(start_from, NUM_CLASSES, CHARSET)
        print(f"נטען צ'קפוינט: {start_from} (צעד {ckpt.get('step', '?')})")
    else:
        model = CRNN(NUM_CLASSES)
        print("מודל חדש אותחל מאפס")
    model.to(DEVICE)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.1)
    training_state = ckpt.get("training_state", {}) if resume_latest else {}
    if ckpt.get("charset_upgraded_from"):
        training_state = {}
    start_step = int(ckpt.get("step", 0)) if resume_latest and training_state else 0
    target_steps = int(training_state.get("target_steps", steps))
    if training_state:
        opt.load_state_dict(training_state["optimizer"])
        sched.load_state_dict(training_state["scheduler"])
        random.setstate(training_state["python_rng"])
        np.random.set_state(training_state["numpy_rng"])
        torch.set_rng_state(training_state["torch_rng"])
        if torch.cuda.is_available() and training_state.get("cuda_rng"):
            torch.cuda.set_rng_state_all(training_state["cuda_rng"])
        sampler_state = training_state.get("sampler_state")
        if sampler_state is not None and hasattr(sampler, "load_state_dict"):
            sampler.load_state_dict(sampler_state)
        else:
            sampler_rng = training_state.get("sampler_rng")
            if sampler_rng is not None and hasattr(sampler, "rng"):
                sampler.rng.setstate(sampler_rng)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    t0 = time.time()
    loss_acc, n_acc = 0.0, 0
    log_name = "base" if is_base else ("general" if is_general else
                                       ("typed" if is_typed else target))
    log_path = os.path.join(MODELS, f"train_{log_name}.log")
    log_f = open(log_path, "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    def save_resume_checkpoint(step):
        state = {
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "target_steps": target_steps,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "sampler_rng": sampler.rng.getstate() if hasattr(sampler, "rng") else None,
            "sampler_state": (sampler.state_dict()
                              if hasattr(sampler, "state_dict") else None),
        }
        save_checkpoint(resume_path, model, CHARSET, step,
                        {"target_steps": target_steps}, state)

    def save_if_improved(step, score):
        nonlocal best_cer, improved, backed_up
        if step <= 0 and baseline is None:
            return False
        if score > best_cer:
            return False
        best_cer = score
        if not backed_up and os.path.exists(ckpt_path):
            _backup_to_history(ckpt_path, log_name)
            backed_up = True
        save_checkpoint(ckpt_path, model, CHARSET, step, {"val_cer": score})
        improved = True
        log(f"  -> נשמר מודל משופר: {ckpt_path}")
        return True

    if training_state:
        log(f"Resume from step {start_step}/{target_steps}; data reloaded")
    log(f"מתחיל אימון: יעד={target}, צעדים={steps}, אצווה={batch_size}, lr={lr}"
        + (f" | GPU: {torch.cuda.get_device_name(0)}" if DEVICE.type == "cuda" else ""))

    # שער 4 (רגרסיה): מודדים את המודל הנטען על סט האימות הקבוע לפני האימון.
    # גרסה חדשה נשמרת רק אם היא לפחות טובה כמוהו — אחרת הקובץ הקיים נשאר
    # (rollback אוטומטי), ואי אפשר "לקלקל" מודל עובד בריצת אימון גרועה.
    baseline = None
    if start_from:
        if resume_latest and os.path.isfile(ckpt_path):
            baseline_model, _ = load_checkpoint(ckpt_path, NUM_CLASSES, CHARSET)
            baseline_model.to(DEVICE)
            baseline = evaluate(baseline_model, val_batch)
            del baseline_model
        else:
            baseline = evaluate(model, val_batch)
        log(f"CER בסיס לפני אימון: {baseline*100:.2f}% (שער רגרסיה פעיל)")
    best_cer = baseline if baseline is not None else float("inf")
    improved, backed_up = False, False

    if start_step >= target_steps:
        log(f"No steps remain; checkpoint already reached {target_steps}")
        log_f.close()
        return True

    for step in range(start_step + 1, target_steps + 1):
        if os.path.exists(STOP_FILE):
            save_resume_checkpoint(step - 1)
            score = evaluate(model, val_batch)
            log(f"עצירה: CER באימות באותה חלוקה קבועה: {score*100:.2f}%")
            save_if_improved(step - 1, score)
            log(f"Training stopped by request; resume is available at step {step - 1}")
            log_f.close()
            return False
        imgs, targets, in_lens, tgt_lens, _ = sampler.batch(batch_size)
        logits = model(imgs.to(DEVICE))           # B,T,C
        log_probs = logits.log_softmax(-1).permute(1, 0, 2)  # T,B,C
        loss = ctc(log_probs, targets.to(DEVICE), in_lens.to(DEVICE),
                   tgt_lens.to(DEVICE))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        loss_acc += loss.item()
        n_acc += 1

        if step % 50 == 0:
            rate = step / (time.time() - t0)
            log(f"צעד {step}/{target_steps} | loss={loss_acc/n_acc:.4f} | {rate:.2f} צעדים/שנ'")
            loss_acc, n_acc = 0.0, 0

        eval_every = 500 if is_base else (200 if is_general else 100)
        if step % 50 == 0:
            save_resume_checkpoint(step)
        if step % eval_every == 0 or step == target_steps:
            v = evaluate(model, val_batch)
            log(f"  -> CER באימות: {v*100:.2f}%")
            save_resume_checkpoint(step)
            save_if_improved(step, v)

    if improved:
        log(f"האימון הסתיים. CER הטוב ביותר: {best_cer*100:.2f}%")
    elif baseline is not None:
        log(f"האימון הסתיים. שער הרגרסיה: אף גרסה לא עקפה את המודל הקיים "
            f"({baseline*100:.2f}%) — הקובץ הקודם נשאר ללא שינוי")
    else:
        log("האימון הסתיים בלי שיפור מדיד")
    try:
        with open(os.path.join(MODELS, f"{log_name}.meta.json"), "w",
                  encoding="utf-8") as mf:
            json.dump({"target": log_name,
                       "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "steps": target_steps,
                       "train_samples": len(sampler.real_items) or None,
                       "baseline_cer": baseline,
                       "best_cer": None if best_cer == float("inf") else best_cer,
                       "improved": improved,
                       "device": DEVICE.type}, mf, ensure_ascii=False, indent=1)
    except OSError:
        pass
    log_f.close()
    return True


PID_FILE = os.path.join(MODELS, "train.pid")


def _write_pid():
    _clear_stop_request()
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))


def _clear_pid():
    try:
        os.remove(PID_FILE)
    except OSError:
        pass


def _clear_stop_request():
    try:
        os.remove(STOP_FILE)
    except OSError:
        pass


def chain(writer_name, steps_writer=2000, steps_general=2000, resume=True):
    """שרשרת האימון המלאה: קודם המודל האישי של הכותב, ואז המודל הכללי
    (שמתאמן על כל הכותבים יחד). כך כל אימון משפר את שניהם."""
    _write_pid()
    try:
        try:
            completed = train(writer_name, steps_writer, 24, 1e-4, resume=resume)
        except SystemExit:
            completed = not os.path.exists(STOP_FILE)
        if completed and not os.path.exists(STOP_FILE):
            try:
                train("__general__", steps_general, 24, 1e-4, resume=resume)
            except SystemExit:
                pass
    finally:
        _clear_pid()
        _clear_stop_request()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["base", "writer", "general", "typed",
                                     "chain", "rollback"])
    ap.add_argument("name", nargs="?", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fresh", action="store_true",
                    help="start a fresh run and archive the interrupted checkpoint")
    args = ap.parse_args()

    if args.mode == "rollback":
        if not args.name:
            print("צריך שם: python train.py rollback <שם כותב|general|base>")
            sys.exit(1)
        sys.exit(0 if rollback(args.name) else 1)

    if args.mode == "chain":
        if not args.name:
            print("צריך שם כותב: python train.py chain <שם>")
            sys.exit(1)
        chain(args.name, args.steps or 2000, args.steps or 2000, resume=args.resume and not args.fresh)
        sys.exit(0)

    _write_pid()
    try:
        if args.mode == "writer":
            if not args.name:
                print("צריך שם כותב: python train.py writer <שם>")
                sys.exit(1)
            train(args.name, args.steps or 2000, args.batch, args.lr or 1e-4, args.resume)
        elif args.mode == "general":
            train("__general__", args.steps or 4000, args.batch, args.lr or 1e-4, args.resume)
        elif args.mode == "typed":
            train("__typed__", args.steps or 3000, args.batch, args.lr or 2e-4, args.resume)
        else:
            train("base", args.steps or 30000, args.batch, args.lr or 3e-4, args.resume)
    finally:
        _clear_pid()
        _clear_stop_request()
