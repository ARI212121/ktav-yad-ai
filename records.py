# -*- coding: utf-8 -*-
"""CRUD operations for a writer's saved OCR training examples."""
import os
import tempfile

from PIL import Image

from paths import WRITERS_DIR


def validate_writer(writer):
    writer = (writer or "").strip()
    if (not writer or writer in (".", "..")
            or any(char in writer for char in ("/", "\\", "\0"))
            or os.path.basename(writer) != writer):
        raise ValueError("שם הכותב חסר או אינו תקין")
    return writer


def _paths(writer):
    writer = validate_writer(writer)
    root = os.path.join(WRITERS_DIR, writer)
    return root, os.path.join(root, "labels.tsv"), os.path.join(root, "lines")


def _read_labels(path):
    rows = []
    try:
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                parts = line.rstrip("\r\n").split("\t", 1)
                if len(parts) == 2 and parts[0] and parts[1].strip():
                    rows.append((parts[0], parts[1]))
    except FileNotFoundError:
        pass
    return rows


def _write_labels(path, rows):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".labels-", suffix=".tmp",
                                     dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            for filename, text in rows:
                stream.write(f"{filename}\t{text.replace(chr(9), ' ').strip()}\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _rebuild_hash_index(writer, rows, lines_dir):
    import gates

    index = {}
    for filename, text in rows:
        path = _safe_line_path(lines_dir, filename)
        if not os.path.isfile(path):
            continue
        try:
            with Image.open(path) as image:
                index[gates.img_hash(image)] = text
        except (OSError, ValueError):
            continue
    gates.save_hash_index(writer, index)


def _safe_line_path(lines_dir, filename):
    if (not isinstance(filename, str) or not filename
            or filename != os.path.basename(filename)
            or "/" in filename or "\\" in filename):
        raise ValueError("קובץ הדוגמה אינו תקין")
    path = os.path.abspath(os.path.join(lines_dir, filename))
    if os.path.dirname(path) != os.path.abspath(lines_dir):
        raise ValueError("קובץ הדוגמה מחוץ לתיקיית הכותב")
    return path


def list_examples(writer, offset=0, limit=25):
    _, labels_path, lines_dir = _paths(writer)
    offset = max(0, int(offset))
    limit = min(100, max(1, int(limit)))
    all_rows = _read_labels(labels_path)
    rows = []
    for filename, text in all_rows[offset:offset + limit]:
        path = _safe_line_path(lines_dir, filename)
        if os.path.isfile(path):
            rows.append({"file": filename, "text": text, "image_path": path})
    return {"total": len(all_rows), "offset": offset, "limit": limit,
            "items": rows}


def edit_example(writer, filename, text):
    writer = validate_writer(writer)
    _, labels_path, lines_dir = _paths(writer)
    if (not isinstance(filename, str) or not filename
            or filename != os.path.basename(filename)
            or "/" in filename or "\\" in filename):
        raise ValueError("קובץ הדוגמה אינו תקין")
    image_path = _safe_line_path(lines_dir, filename)
    rows = _read_labels(labels_path)
    row_index = next((i for i, (name, _) in enumerate(rows)
                      if name == filename), None)
    if row_index is None or not os.path.isfile(image_path):
        raise FileNotFoundError("דוגמת האימון לא נמצאה")
    from gates import gate1_format
    try:
        with Image.open(image_path) as image:
            normalized, reason = gate1_format(image.convert("L"), text)
    except OSError as exc:
        raise ValueError(f"לא ניתן לקרוא את תמונת הדוגמה: {exc}") from exc
    if reason:
        raise ValueError(reason)
    rows[row_index] = (filename, normalized)
    _write_labels(labels_path, rows)
    _rebuild_hash_index(writer, rows, lines_dir)
    return {"writer": writer, "file": filename, "text": normalized,
            "model_retrained": False}


def delete_example(writer, filename):
    writer = validate_writer(writer)
    _, labels_path, lines_dir = _paths(writer)
    if (not isinstance(filename, str) or not filename
            or filename != os.path.basename(filename)
            or "/" in filename or "\\" in filename):
        raise ValueError("קובץ הדוגמה אינו תקין")
    image_path = _safe_line_path(lines_dir, filename)
    rows = _read_labels(labels_path)
    filtered = [row for row in rows if row[0] != filename]
    if len(filtered) == len(rows):
        raise FileNotFoundError("דוגמת האימון לא נמצאה")
    if os.path.exists(image_path):
        os.remove(image_path)
    _write_labels(labels_path, filtered)
    _rebuild_hash_index(writer, filtered, lines_dir)
    return {"writer": writer, "file": filename, "deleted": True,
            "model_retrained": False}
