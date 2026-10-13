# -*- coding: utf-8 -*-
"""Iteratively train Samuel's personal model until this page matches its transcript."""
import argparse
import base64
import json
import os
import subprocess
import sys
import time
import unicodedata
from pathlib import Path


WRITER = "\u05e9\u05de\u05d5\u05d0\u05dc"
REFERENCE_LINES = (
    "\u05d3\u05e2 \u05db\u05d0 (1)",
    "\u05de\u05e9\u05e4\u05d8\u05d9 \u05d4\u05ea\u05e0\u05d0\u05d9\u05dd (1)",
    "\u05dc\u05ea\u05d5\u05e1' - \u05e1\u05dc\u05e7\u05d5 \u05d0\u05dd "
    "\u05e6\u05e8\u05d9\u05da \u05de\u05e9\u05e4\u05d8\u05d9' "
    "\u05d4\u05ea\u05e0\u05d0\u05d9\u05dd.",
    "\u05dc\u05e8\u05e9\"\u05d9 \u05e0\u05d7\u05dc\u05e7\u05d5 "
    "\u05d1\u05ea\u05e0\u05d0\u05d9 \u05db\u05e4\u05d5\u05dc, \u05d4\u05d0\u05dd "
    "\u05de\u05db\u05dc\u05dc \u05d4\u05df \u05d0\u05ea\u05d4 \u05e9\u05d5\u05de\u05e2 "
    "\u05dc\u05d0\u05d5. \u05d5\u05d9\u05e9 \u05e9\u05d9\u05d8\u05d4 "
    "\u05e9\u05dc\u05d9\u05e9\u05d9\u05ea (\u05e8' \u05d9\u05d4\u05d5\u05d3\u05d4 - "
    "\u05db\u05e0\u05e8\u05d0\u05d4) \u05d3\u05e1\u05d1\u05e8 \u05e9\u05dc\u05d0 "
    "\u05d1\u05e2\u05d9' \u05de\u05e9\u05e4\u05d8\u05d9",
    "\u05d4\u05ea\u05e0\u05d0\u05d9\u05dd",
)


def _normalized(text):
    return unicodedata.normalize("NFC", " ".join(text.split())).strip()


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2),
                         encoding="utf-8")
    os.replace(temporary, path)


def _read_labels(path):
    if not path.is_file():
        raise RuntimeError("Samuel has no labels.tsv; add the verified lines first.")
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "\t" not in raw:
            continue
        filename, text = raw.split("\t", 1)
        if filename.strip() and text.strip():
            rows.append((filename, text))
    if len(rows) != len(REFERENCE_LINES):
        raise RuntimeError(
            f"Expected {len(REFERENCE_LINES)} Samuel labels, found {len(rows)}.")
    if [_normalized(text) for _, text in rows] != [
            _normalized(text) for text in REFERENCE_LINES]:
        raise RuntimeError(
            "Samuel's labels differ from the approved transcript; refusing to train.")
    return [filename for filename, _ in rows]


def _run_ocr(cli, data_dir, image, env):
    result = subprocess.run(
        [sys.executable, str(cli), "--data-dir", str(data_dir), "ocr",
         str(image), "--page", "--writer", WRITER, "--model", "personal",
         "--no-fix", "--format", "json", "--timeout", "240"],
        capture_output=True, env=env, timeout=300)
    if result.returncode:
        detail = result.stderr.decode("utf-8", "replace")
        raise RuntimeError("OCR failed: " + detail)
    try:
        payload = json.loads(result.stdout.decode("utf-8"))
        lines = payload["lines"]
        if not isinstance(lines, list) or not all(
                isinstance(item.get("text"), str) for item in lines):
            raise ValueError("OCR lines are not valid text records")
        return [item["text"] for item in lines]
    except (UnicodeDecodeError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"OCR returned invalid JSON: {exc}") from exc


def _assert_no_training(cli, data_dir, env):
    result = subprocess.run(
        [sys.executable, str(cli), "--data-dir", str(data_dir),
         "train", "status", "--json"],
        capture_output=True, env=env, timeout=60)
    if result.returncode:
        raise RuntimeError(
            "Could not check existing training status: "
            + result.stderr.decode("utf-8", "replace"))
    status = json.loads(result.stdout.decode("utf-8"))
    if status.get("running"):
        raise RuntimeError(
            "Another training job is active. Stop it before running this loop.")


def _checkpoint_needs_resume(path):
    if not path.is_file():
        return False
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("training_state", {})
    step = int(checkpoint.get("step", 0))
    target = int(state.get("target_steps", 0))
    return bool(state and 0 < step < target)


def _train_round(train_script, model_dir, writer, steps, batch_size,
                 learning_rate, env, deadline, stop_file, log_path):
    latest = model_dir / f"writer_{writer}.pt.last"
    resume = _checkpoint_needs_resume(latest)
    command = [
        sys.executable, str(train_script), "writer", writer,
        "--steps", str(steps), "--batch", str(batch_size),
        "--lr", str(learning_rate),
        "--resume" if resume else "--fresh",
    ]
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\nround_start resume={resume} steps={steps}\n")
        log.flush()
        process = subprocess.Popen(
            command, cwd=str(train_script.parent), env=env,
            stdout=log, stderr=subprocess.STDOUT)
        requested_stop = False
        try:
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    requested_stop = True
                    stop_file.write_text("stop", encoding="ascii")
                    break
                time.sleep(2)
        except KeyboardInterrupt:
            requested_stop = True
            stop_file.write_text("stop", encoding="ascii")
        return_code = process.wait()
        if return_code:
            raise RuntimeError(
                f"Training process exited with code {return_code}; "
                f"see {log_path}.")
    return resume, requested_stop


def _restore_split(split_path, state):
    encoded = state.get("original_split_b64")
    if encoded is None:
        try:
            split_path.unlink()
        except FileNotFoundError:
            pass
    else:
        split_path.write_bytes(base64.b64decode(encoded, validate=True))


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train only Samuel's personal OCR model until this page "
                    "matches the approved five-line transcript.")
    parser.add_argument("--data-dir", type=Path, default=project / "portable",
                        help="shared data/models directory (default: portable)")
    parser.add_argument(
        "--image", type=Path,
        default=Path(r"D:\DEV\HebrewOcrStudio\temp_extracted_glyphs\cleaned_page.png"))
    parser.add_argument("--max-hours", type=float, default=3.0)
    parser.add_argument("--steps-per-round", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--reset-loop-state", action="store_true",
                        help="start a new loop counter; keeps the model and labels")
    args = parser.parse_args(argv)

    data_dir = args.data_dir.expanduser().resolve()
    image = args.image.expanduser().resolve()
    if not image.is_file():
        parser.error(f"Image not found: {image}")
    if not data_dir.is_dir():
        parser.error(f"Data directory not found: {data_dir}")
    if not 0 < args.max_hours <= 400:
        parser.error("--max-hours must be greater than 0 and at most 400")
    if not 100 <= args.steps_per_round <= 10000:
        parser.error("--steps-per-round must be between 100 and 10000")
    if not 1 <= args.batch_size <= 32:
        parser.error("--batch-size must be between 1 and 32")
    if not 0 < args.learning_rate <= 1e-2:
        parser.error("--learning-rate must be greater than 0 and at most 0.01")

    writer_dir = data_dir / "data" / "writers" / WRITER
    model_dir = data_dir / "models"
    split_path = writer_dir / "val_split.json"
    state_path = model_dir / "samuel_until_exact_state.json"
    log_path = model_dir / "samuel_until_exact.log"
    stop_file = model_dir / "train.stop"
    labels = _read_labels(writer_dir / "labels.tsv")
    if not args.reset_loop_state and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (state.get("writer") != WRITER
                or state.get("image") != str(image)
                or state.get("data_dir") != str(data_dir)):
            parser.error("Existing loop state targets different data; use "
                         "--reset-loop-state only to start a separate run.")
    else:
        original_split = split_path.read_bytes() if split_path.is_file() else None
        state = {
            "writer": WRITER,
            "image": str(image),
            "data_dir": str(data_dir),
            "round": 0,
            "original_split_b64": (base64.b64encode(original_split).decode("ascii")
                                   if original_split is not None else None),
        }
        _write_json(state_path, state)

    env = dict(os.environ, OCRHE_HOME=str(data_dir),
               PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    cli = project / "cli.py"
    train_script = project / "train.py"
    deadline = time.monotonic() + args.max_hours * 3600
    perfect = False
    try:
        initial = _run_ocr(cli, data_dir, image, env)
        print(f"Initial exact lines: {sum(_normalized(a) == _normalized(b) for a, b in zip(initial, REFERENCE_LINES))}/{len(REFERENCE_LINES)}")
        if len(initial) == len(REFERENCE_LINES) and all(
                _normalized(a) == _normalized(b)
                for a, b in zip(initial, REFERENCE_LINES)):
            perfect = True
        while not perfect and time.monotonic() < deadline:
            round_number = int(state.get("round", 0))
            validation_file = labels[round_number % len(labels)]
            _write_json(split_path, [validation_file])
            state["in_progress_round"] = round_number
            _write_json(state_path, state)
            print(f"Training round {round_number + 1}; "
                  f"hold-out sample {round_number % len(labels) + 1}; "
                  f"log: {log_path}")
            _assert_no_training(cli, data_dir, env)
            _, stopped = _train_round(
                train_script, model_dir, WRITER, args.steps_per_round,
                args.batch_size, args.learning_rate, env, deadline,
                stop_file, log_path)
            predicted = _run_ocr(cli, data_dir, image, env)
            matches = [
                _normalized(predicted[i]) == _normalized(REFERENCE_LINES[i])
                for i in range(min(len(predicted), len(REFERENCE_LINES)))
            ]
            perfect = len(predicted) == len(REFERENCE_LINES) and all(matches)
            state["round"] = round_number if stopped else round_number + 1
            if stopped:
                state["in_progress_round"] = round_number
            else:
                state.pop("in_progress_round", None)
            state["last_match_count"] = sum(matches)
            state["complete"] = perfect
            _write_json(state_path, state)
            print(f"Exact lines after round: {sum(matches)}/{len(REFERENCE_LINES)}")
            if perfect:
                print("Exact transcript reached; personal model saved.")
                break
            if stopped or time.monotonic() >= deadline:
                print("Time limit reached; checkpoint is available to resume.")
                break
    finally:
        _restore_split(split_path, state)

    return 0 if perfect else 2


if __name__ == "__main__":
    raise SystemExit(main())
