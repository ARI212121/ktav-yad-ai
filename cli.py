# -*- coding: utf-8 -*-
"""Cross-platform OCRHE command line interface."""
import argparse
import base64
from contextlib import redirect_stdout
import getpass
from io import StringIO
import json
import os
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

VERSION = "1.0.1"


class CLIError(RuntimeError):
    def __init__(self, message, code=1):
        super().__init__(message)
        self.code = code


def _json(value):
    return json.dumps(value, ensure_ascii=False, default=str)


def _output(value, as_json=False):
    if as_json:
        print(_json(value))
    elif isinstance(value, str):
        print(value)
    elif isinstance(value, list):
        for item in value:
            print("- " + (_json(item) if isinstance(item, (dict, list))
                         else str(item)))
    elif isinstance(value, dict):
        for key, item in value.items():
            print(f"{key}: {item}")
    else:
        print(value)


def _app():
    import app
    return app


def _writer(name):
    if name:
        return name
    app = _app()
    writers = app.list_writers()
    if not sys.stdin.isatty():
        raise CLIError("יש לציין --writer במצב לא אינטראקטיבי", 2)
    if not writers:
        return input("שם כותב חדש: ").strip()
    print("בחר כותב:")
    for index, item in enumerate(writers, 1):
        print(f"  {index}. {item['name']}")
    choice = input("מספר או שם כותב חדש: ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(writers):
        return writers[int(choice) - 1]["name"]
    return choice


def _get_model_preference(model):
    return {"auto": None, "personal": "personal", "general": "general",
            "typed": "typed"}[model]


def command_ocr(args):
    if args.include_images and args.format not in ("json", "jsonl"):
        raise CLIError("--include-images מחייב --format json או jsonl", 2)
    if args.include_images and not args.page:
        raise CLIError("--include-images מחייב --page כדי להפיק שורות ממוספרות", 2)
    app = _app()
    source = Path(args.input).expanduser()
    if not source.exists():
        raise CLIError(f"קובץ או תיקייה לא נמצאו: {source}", 4)
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pdf"}
    files = ([path for path in sorted(source.iterdir())
              if path.is_file() and path.suffix.lower() in extensions]
             if source.is_dir() else [source])
    if not files:
        raise CLIError("לא נמצאו בתיקייה קובצי תמונה או PDF נתמכים", 2)
    encoded = []
    for path in files:
        try:
            encoded.append({"name": path.name,
                            "data": base64.b64encode(path.read_bytes()).decode("ascii")})
        except OSError as exc:
            raise CLIError(f"לא ניתן לקרוא את {path}: {exc}", 4) from exc

    job_id = app.start_transcribe(
        encoded, args.page, args.writer, _get_model_preference(args.model),
        not args.no_fix)
    started = time.monotonic()
    last_count = 0
    while True:
        job = app._jobs.get(job_id)
        if job is None:
            raise CLIError("משימת התמלול אבדה לפני קבלת התוצאה")
        if args.progress and job.get("pages_done") != last_count:
            last_count = job["pages_done"]
            print(f"עמודים: {last_count}/{job.get('pages_total', 0)}",
                  file=sys.stderr)
        if job.get("done"):
            break
        if args.timeout and time.monotonic() - started > args.timeout:
            raise CLIError("תם הזמן שהוגדר לתמלול; המשימה עדיין עשויה להסתיים ברקע")
        time.sleep(0.1)
    if job.get("error"):
        raise CLIError(job["error"], 5 if "מודל" in job["error"] else 1)

    lines = []
    for index, item in enumerate(job["lines"], 1):
        page = item.get("page")
        line = {
            "number": index,
            "text": item.get("text", ""),
            "confidence": item.get("conf"),
            "source": item.get("src"),
            "page": page + 1 if isinstance(page, int) and not isinstance(
                page, bool) else page,
            "model": "__typed__" if item.get("typed") else job.get("model"),
            "suggestion": item.get("sugg"),
        }
        if args.include_images:
            line["img"] = item.get("img", "")
        lines.append(line)
    result = {"schema_version": 1, "input": str(source),
              "model": job.get("model"), "models_used": job.get("models_used", []),
              "lines": lines, "text": "\n".join(item["text"] for item in lines)}
    if args.format == "text":
        rendered = result["text"]
    elif args.format == "jsonl":
        rendered = "\n".join(_json(item) for item in lines)
    elif args.format == "docx":
        rendered = app.build_docx([{"text": item["text"]} for item in lines])
    else:
        rendered = _json(result)
    if args.output:
        try:
            path = Path(args.output).expanduser()
            if isinstance(rendered, bytes):
                path.write_bytes(rendered)
            else:
                path.write_text(rendered, encoding="utf-8")
        except OSError as exc:
            raise CLIError(f"לא ניתן לכתוב את קובץ הפלט: {exc}", 4) from exc
    elif isinstance(rendered, bytes):
        sys.stdout.buffer.write(rendered)
    else:
        print(rendered)


def _parse_line_selection(selection, line_numbers):
    if not selection:
        return list(range(len(line_numbers)))
    available = set(line_numbers)
    highest_number = max(available)
    requested = set()
    for part in selection.split(","):
        part = part.strip()
        if not part:
            raise CLIError("בחירת שורות ריקה אינה תקינה", 2)
        try:
            if "-" in part:
                start_text, end_text = part.split("-", 1)
                start, end = int(start_text), int(end_text)
                if start < 1 or end < start or end > highest_number:
                    raise ValueError
                requested.update(range(start, end + 1))
            else:
                number = int(part)
                if number < 1 or number > highest_number:
                    raise ValueError
                requested.add(number)
        except ValueError as exc:
            raise CLIError(
                f"בחירת שורות לא תקינה: {part}; השתמש למשל ב-1,3-5", 2) from exc
    missing = sorted(requested - available)
    if missing:
        raise CLIError(
            "מספרי שורות מחוץ לטווח: " + ", ".join(map(str, missing)), 2)
    return [index for index, number in enumerate(line_numbers)
            if number in requested]


def command_writers(args):
    if args.action == "delete" and not args.yes:
        raise CLIError(
            "מחיקת כותב מחייבת --yes; הפעולה מוחקת גם נתוני אימון ומודל אישי",
            2)
    if args.action == "purge-data" and not args.yes:
        raise CLIError("מחיקת נתוני האימון מחייבת --yes", 2)
    if args.action == "share" and not args.yes:
        raise CLIError(
            "שליחת חבילת אימון לשרת חיצוני מחייבת --yes; בדוק את התוכן לפני השליחה",
            2)
    app = _app()
    if args.action == "list":
        _output(app.list_writers(), args.json)
    elif args.action == "add":
        name = _writer(args.name).strip()
        if not name:
            raise CLIError("שם הכותב לא יכול להיות ריק", 2)
        app.records.validate_writer(name)
        os.makedirs(os.path.join(app.WRITERS_DIR, name, "lines"), exist_ok=True)
        _output({"ok": True, "writer": name}, args.json)
    elif args.action == "delete":
        ok, message = app.delete_writer(args.name)
        if not ok:
            raise CLIError(message)
        _output({"ok": True, "message": message} if args.json else message,
                args.json)
    elif args.action == "export":
        writer = app.records.validate_writer(args.name)
        data = app.export_writer_zip(writer)
        if data is None:
            raise CLIError("כותב לא נמצא", 4)
        Path(args.output).expanduser().write_bytes(data)
        _output({"exported": writer, "output": args.output}, args.json)
    elif args.action == "import":
        writer = app.records.validate_writer(args.name)
        if app.training_running():
            raise CLIError("אי אפשר לייבא נתונים בזמן אימון פעיל")
        payload = base64.b64encode(
            Path(args.input).expanduser().read_bytes()).decode("ascii")
        added, message = app.import_writer_zip(payload, writer)
        app.postfix.refresh_vocab()
        _output({"added": added, "message": message}, args.json)
    elif args.action == "purge-data":
        writer = app.records.validate_writer(args.name)
        ok, message = app.delete_training_data(writer)
        if not ok:
            raise CLIError(message)
        _output({"deleted": True, "message": message}, args.json)
    elif args.action == "share":
        writer = app.records.validate_writer(args.name)
        ok, message = app.upload_pack_to_dev(writer)
        if not ok:
            raise CLIError(message)
        _output({"uploaded": True, "message": message}, args.json)


def command_examples(args):
    if args.action == "delete" and not args.yes:
        raise CLIError("מחיקת דוגמה מחייבת --yes", 2)
    app = _app()
    writer = _writer(args.writer)
    if not writer:
        raise CLIError("שם הכותב לא יכול להיות ריק", 2)
    records = app.records
    if args.action == "list":
        result = records.list_examples(writer, args.offset, args.limit)
        result["items"] = [{k: v for k, v in item.items() if k != "image_path"}
                           for item in result["items"]]
        _output(result, args.json)
        return
    if args.action == "add":
        if app.training_running():
            raise CLIError("אי אפשר להוסיף דוגמאות בזמן אימון פעיל")
        image_path = Path(args.image).expanduser()
        if not image_path.is_file():
            raise CLIError(f"קובץ התמונה לא נמצא: {image_path}", 4)
        text = args.text
        if args.text_file:
            text = Path(args.text_file).expanduser().read_text(encoding="utf-8").strip()
        if text is None and sys.stdin.isatty():
            text = input("הקלד את הטקסט המתאים לתמונה: ")
        if text is None:
            raise CLIError("יש לציין --text או --text-file במצב לא אינטראקטיבי", 2)
        item = {"img": base64.b64encode(image_path.read_bytes()).decode("ascii"),
                "text": text}
        saved, quarantined, duplicates = app.save_lines(writer, [item])
        app.postfix.refresh_vocab()
        _output({"saved": saved, "quarantined": quarantined,
                 "duplicates": duplicates}, args.json)
        return
    if args.action == "edit":
        if app.training_running():
            raise CLIError("אי אפשר לערוך דוגמאות בזמן אימון פעיל")
        result = records.edit_example(writer, args.file, args.text)
        app.postfix.refresh_vocab()
        _output(result, args.json)
        return
    if args.action == "delete":
        if app.training_running():
            raise CLIError("אי אפשר למחוק דוגמאות בזמן אימון פעיל")
        result = records.delete_example(writer, args.file)
        app.postfix.refresh_vocab()
        _output(result, args.json)


def command_quarantine(args):
    app = _app()
    writer = _writer(args.writer)
    if args.action == "list":
        _output(app.gates.quarantine_list(writer), args.json)
        return
    if args.action == "resolve":
        if app.training_running():
            raise CLIError("אי אפשר לשנות דוגמאות בהסגר בזמן אימון פעיל")
        approve = args.decision == "approve"
        ok, message = app.gates.quarantine_resolve(
            writer, args.file, approve, args.text)
        if not ok:
            raise CLIError(message)
        app.postfix.refresh_vocab()
        _output({"ok": ok, "message": message} if args.json else message,
                args.json)


def command_train(args):
    app = _app()
    if args.action == "status":
        _output(app.train_status(), args.json)
    elif args.action == "stop":
        stopped = app.stop_training()
        if not stopped:
            raise CLIError("לא נמצא אימון פעיל לעצירה")
        _output({"stopping": True} if args.json else "בקשת עצירה נשלחה",
                args.json)
    elif args.action == "start":
        writer = (_writer(args.writer) if args.scope != "general"
                  else args.writer or "")
        ok, message = _start_training(
            app, writer, args.scope, not args.fresh, args.steps, args.batch)
        if not ok:
            raise CLIError(message)
        _output({"started": True, "message": message} if args.json else message,
                args.json)


def _start_training(app, writer, scope, resume, steps=None, batch_size=None):
    if app.training_running():
        return False, "אימון כבר רץ"
    if scope not in ("both", "personal", "general"):
        return False, "היקף אימון לא תקין"
    if scope != "general":
        app.records.validate_writer(writer)
    logs = {"both": [f"train_{writer}.log", "train_general.log"],
            "personal": [f"train_{writer}.log"],
            "general": ["train_general.log"]}[scope]
    for name in logs:
        open(os.path.join(app.MODELS, name), "w", encoding="utf-8").close()
    try:
        os.remove(app.TRAIN_STOP_FILE)
    except FileNotFoundError:
        pass
    flag = "--resume" if resume else "--fresh"
    if getattr(sys, "frozen", False):
        command = [sys.executable, "__train_child__", scope, writer or "-", flag]
        if steps is not None or batch_size is not None:
            command.extend([str(steps or 2000), str(batch_size or 24)])
        creationflags = (subprocess.DETACHED_PROCESS |
                         subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0
        kwargs = {"creationflags": creationflags} if os.name == "nt" else {
            "start_new_session": True}
    else:
        train_script = os.path.join(os.path.dirname(__file__), "train.py")
        mode = ("chain" if scope == "both" else
                "writer" if scope == "personal" else "general")
        command = [sys.executable, train_script, mode]
        if mode in ("chain", "writer"):
            command.append(writer)
        if steps is not None:
            command.extend(["--steps", str(steps)])
        if batch_size is not None:
            command.extend(["--batch", str(batch_size)])
        command.append(flag)
        kwargs = ({"creationflags": subprocess.DETACHED_PROCESS |
                   subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
                  else {"start_new_session": True})
    process = subprocess.Popen(command, cwd=os.path.dirname(__file__),
                               stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, close_fds=True, **kwargs)
    with open(app.PID_FILE, "w", encoding="ascii") as stream:
        stream.write(str(process.pid))
    return True, "האימון הופעל ברקע"


def command_fixes(args):
    if args.action == "delete" and not args.yes:
        raise CLIError("מחיקת תיקון מחייבת --yes", 2)
    app = _app()
    writer_name = getattr(args, "writer", None)
    writer = _writer(writer_name) if writer_name else None
    fixes = app.fixes
    if args.action == "list":
        _output(fixes.stats(writer), args.json)
    elif args.action == "edit":
        fixes.override(args.wrong, args.right, writer)
        _output({"updated": True, "wrong": args.wrong, "right": args.right},
                args.json)
    elif args.action == "delete":
        removed = fixes.remove(args.wrong, writer)
        _output({"removed": removed}, args.json)
    elif args.action == "export":
        text = fixes.export_tsv(writer)
        if not text:
            raise CLIError("אין תיקונים לייצוא")
        Path(args.output).expanduser().write_text(text, encoding="utf-8")
        _output({"exported": True, "output": args.output}, args.json)
    elif args.action == "import":
        text = Path(args.input).expanduser().read_text(encoding="utf-8")
        added = fixes.import_tsv(text)
        _output({"imported": added}, args.json)
    elif args.action == "clear-list":
        count = fixes.mark_seen(writer)
        _output({"active": count, "message": "נוקתה תצוגת התיקונים בלבד"}, args.json)


def command_templates(args):
    app = _app()
    if args.action == "list":
        _output(app.list_templates(), args.json)
    elif args.action == "generate":
        if not args.text_file and not 1 <= args.count <= 1000:
            raise CLIError("כמות העמודים חייבת להיות בין 1 ל־1000", 2)
        import make_template
        messages = StringIO()
        if args.text_file:
            text = Path(args.text_file).expanduser().read_text(encoding="utf-8")
            with redirect_stdout(messages):
                names = make_template.make_custom_pages(
                    [line for line in text.splitlines() if line.strip()],
                    prefix=args.prefix)
        else:
            with redirect_stdout(messages):
                names = make_template.make_pages(args.count)
        _output({"created": names, "message": messages.getvalue().strip()},
                args.json)
    elif args.action == "export":
        Path(args.output).expanduser().write_bytes(app.templates_zip_bytes())
        _output({"exported": True, "output": args.output}, args.json)
    elif args.action == "ingest":
        writer = _writer(args.writer)
        if app.training_running():
            raise CLIError("אי אפשר להוסיף דוגמאות בזמן אימון פעיל")
        if not Path(args.input).expanduser().is_file():
            raise CLIError(f"קובץ התבנית הסרוק לא נמצא: {args.input}", 4)
        messages = StringIO()
        with redirect_stdout(messages):
            count = app.ingest_template(str(Path(args.input).expanduser()),
                                        args.template, writer)
        app.postfix.refresh_vocab()
        _output({"saved": count, "writer": writer,
                 "message": messages.getvalue().strip()}, args.json)


def _read_only_validation_split(writer, items):
    import random
    import train

    by_name = {os.path.basename(path): (path, text) for path, text in items}
    names = sorted(by_name)
    split_path = train._val_split_path(writer)
    try:
        with open(split_path, encoding="utf-8") as stream:
            validation_names = [name for name in json.load(stream)
                                if name in by_name]
    except (OSError, ValueError):
        validation_names = []
    target = max(1, len(names) // 10) if len(names) >= 5 else 0
    if len(validation_names) < target:
        pool = [name for name in names if name not in set(validation_names)]
        random.Random(42).shuffle(pool)
        validation_names.extend(pool[:target - len(validation_names)])
    return [by_name[name] for name in sorted(set(validation_names))]


def command_models(args):
    if args.action in ("delete-history", "rollback") and not args.yes:
        action = "מחיקת גרסת מודל מההיסטוריה" if args.action == "delete-history" else "שחזור מודל"
        raise CLIError(f"{action} מחייב --yes", 2)
    app = _app()
    if args.action == "list":
        models = []
        for name, path in (("base", os.path.join(app.MODELS, "base_synth.pt")),
                           ("general", os.path.join(app.MODELS, "general.pt")),
                           ("typed", os.path.join(app.MODELS, "typed.pt"))):
            models.append({"name": name, "path": path, "available": os.path.isfile(path)})
        for writer in app.list_writers():
            path = os.path.join(app.MODELS, f"writer_{writer['name']}.pt")
            if os.path.isfile(path):
                models.append({"name": "writer:" + writer["name"],
                               "path": path, "available": True})
        _output(models, args.json)
    elif args.action == "export":
        target = args.name
        if target in ("base", "general", "typed"):
            filename = {"base": "base_synth.pt",
                        "general": "general.pt", "typed": "typed.pt"}[target]
        elif target.startswith("writer:"):
            writer = app.records.validate_writer(target[len("writer:"):])
            filename = f"writer_{writer}.pt"
        else:
            raise CLIError("שם מודל: base, general, typed או writer:<שם>", 2)
        source = os.path.join(app.MODELS, filename)
        if not os.path.isfile(source):
            raise CLIError("מודל זה לא נמצא", 4)
        shutil.copy2(source, Path(args.output).expanduser())
        _output({"exported": target, "output": args.output}, args.json)
    elif args.action == "import":
        target = args.name
        if app.training_running():
            raise CLIError("אי אפשר לייבא מודל בזמן אימון פעיל")
        raw = Path(args.input).expanduser().read_bytes()
        if target.startswith("writer:"):
            writer = app.records.validate_writer(target[len("writer:"):])
            name, message = app.import_model_file(
                base64.b64encode(raw).decode("ascii"), writer)
            if name is None:
                raise CLIError(message)
            result = {"imported": name, "message": message}
        elif target in ("base", "general", "typed"):
            filename, cache_key = {
                "base": ("base_synth.pt", "__base__"),
                "general": ("general.pt", "__general__"),
                "typed": ("typed.pt", "__typed__"),
            }[target]
            destination = os.path.join(app.MODELS, filename)
            temporary = destination + ".importing"
            try:
                with open(temporary, "wb") as stream:
                    stream.write(raw)
                app.load_checkpoint(temporary, app.NUM_CLASSES, app.CHARSET)
                if os.path.isfile(destination):
                    import train
                    train._backup_to_history(destination, target)
                os.replace(temporary, destination)
            except Exception:
                if os.path.exists(temporary):
                    os.remove(temporary)
                raise
            app._state["models"].pop(cache_key, None)
            result = {"imported": target}
        else:
            raise CLIError("שם מודל: base, general, typed או writer:<שם>", 2)
        _output(result, args.json)
    elif args.action == "history":
        history = os.path.join(app.MODELS, "history")
        files = []
        if os.path.isdir(history):
            for name in sorted(os.listdir(history)):
                if name.endswith(".pt") and (
                        not args.name or name.startswith(args.name + "__")):
                    path = os.path.join(history, name)
                    files.append({"file": name, "bytes": os.path.getsize(path),
                                  "modified": time.strftime(
                                      "%Y-%m-%dT%H:%M:%S",
                                      time.localtime(os.path.getmtime(path)))})
        _output(files, args.json)
    elif args.action == "delete-history":
        if os.path.basename(args.file) != args.file or not args.file.endswith(".pt"):
            raise CLIError("שם קובץ היסטוריה לא תקין", 2)
        path = os.path.join(app.MODELS, "history", args.file)
        if not os.path.isfile(path):
            raise CLIError("גרסת המודל לא נמצאה", 4)
        os.remove(path)
        _output({"deleted": args.file}, args.json)
    elif args.action == "rollback":
        import train
        messages = StringIO()
        with redirect_stdout(messages):
            rolled_back = train.rollback(args.name)
        if not rolled_back:
            raise CLIError("לא נמצאה גרסת מודל קודמת לשחזור")
        _output({"rolled_back": args.name,
                 "message": messages.getvalue().strip()}, args.json)
    elif args.action == "evaluate":
        import train
        if not 1 <= args.limit <= 5000:
            raise CLIError("limit חייב להיות בין 1 ל־5000", 2)
        name = args.name
        if name == "typed":
            model, model_name = app.get_model(prefer="typed")
            if model_name != "__typed__":
                raise CLIError("המודל המוקלד אינו זמין", 4)
            batch = train.make_screen_val(args.limit)
            source = "synthetic-screen-validation"
        elif name == "base":
            path = os.path.join(app.MODELS, "base_synth.pt")
            if not os.path.isfile(path):
                raise CLIError("מודל הבסיס אינו זמין", 4)
            model, _ = app.load_checkpoint(path, app.NUM_CLASSES, app.CHARSET)
            model.to(app.DEVICE)
            model.eval()
            model_name = "__base__"
            batch = train.make_synth_val(args.limit)
            source = "synthetic-validation"
        elif name == "general":
            model, model_name = app.get_model(prefer="general")
            if model_name != "__general__":
                raise CLIError("המודל הכללי אינו זמין", 4)
            from dataset import RealLines
            validation = []
            for writer in sorted(os.listdir(app.WRITERS_DIR)):
                writer_path = os.path.join(app.WRITERS_DIR, writer)
                if not os.path.isdir(writer_path):
                    continue
                validation.extend(_read_only_validation_split(
                    writer, RealLines(writer).items))
            if not validation:
                raise CLIError("אין דוגמאות אימות למודל הכללי", 4)
            batch = train.eval_batch_from_items(validation, args.limit)
            source = "held-out-writer-lines"
        elif name.startswith("writer:"):
            writer = app.records.validate_writer(name[len("writer:"):])
            from dataset import RealLines
            items = RealLines(writer).items
            validation = _read_only_validation_split(writer, items)
            if not validation:
                raise CLIError("לכותב אין מספיק נתונים בקבוצת אימות", 4)
            batch = train.eval_batch_from_items(validation, args.limit)
            model, model_name = app.get_model(writer)
            if model_name != writer:
                raise CLIError(f"לא נמצא מודל אישי לכותב {writer}", 4)
            source = "held-out-writer-lines"
        else:
            raise CLIError("שם מודל: base, general, typed או writer:<שם>", 2)
        if model is None or batch is None:
            raise CLIError(f"לא ניתן להכין הערכה עבור {name}", 4)
        score = train.evaluate(model, batch)
        _output({"model": model_name, "validation_source": source,
                 "cer": score, "cer_percent": round(score * 100, 3)},
                args.json)


def command_ai(args):
    if args.action == "keys" and args.keys_action == "remove" and not args.yes:
        raise CLIError("הסרת מפתח מחייבת --yes", 2)
    app = _app()
    ai = app.ai_correction
    if args.action == "keys":
        if args.keys_action == "list":
            _output(ai.list_keys(), args.json)
        elif args.keys_action == "add":
            key = os.environ.get(args.key_env) if args.key_env else None
            if key is None and sys.stdin.isatty():
                key = getpass.getpass("Google AI Studio API key (input hidden): ")
            if not key:
                raise CLIError("ספק מפתח דרך משתנה סביבה עם --key-env; אין להעביר מפתח גלוי בארגומנט", 2)
            _output(ai.add_key(args.name, key), args.json)
        elif args.keys_action == "remove":
            _output(ai.remove_key(args.id), args.json)
        elif args.keys_action == "rename":
            _output(ai.rename_key(args.id, args.name), args.json)
    elif args.action == "settings":
        if args.settings_action == "get":
            _output(ai.get_settings(), args.json)
        elif args.settings_action == "reset":
            _output(ai.reset_settings(), args.json)
        elif args.settings_action == "set":
            settings = {
                name: getattr(args, name)
                for name in ("batch_size", "temperature", "top_p", "top_k",
                             "max_output_tokens", "request_timeout")
                if getattr(args, name) is not None
            }
            if args.clear_top_p:
                settings["top_p"] = None
            if args.clear_top_k:
                settings["top_k"] = None
            if args.clear_max_output_tokens:
                settings["max_output_tokens"] = None
            _output(ai.save_settings(settings), args.json)
    elif args.action == "models":
        _output(ai.list_available_models(args.id), args.json)
    elif args.action == "test":
        _output(ai.test_model(args.id, args.model), args.json)
    elif args.action == "check":
        _output(ai.check_key(args.id), args.json)
    elif args.action == "set-model":
        _output(ai.set_key_model(args.id, args.model), args.json)
    elif args.action == "correct":
        try:
            payload = json.loads(
                Path(args.input).expanduser().read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CLIError(f"קלט תיקון AI אינו JSON תקין: {exc}", 2) from exc
        lines = payload.get("lines") if isinstance(payload, dict) else payload
        if not isinstance(lines, list) or not lines:
            raise CLIError(
                "קובץ הקלט חייב להכיל מערך שורות או תוצאת OCR עם שורות", 2)
        normalized = []
        line_numbers = []
        for index, source_line in enumerate(lines):
            if not isinstance(source_line, dict):
                raise CLIError("כל פריט בקלט חייב להיות אובייקט שורה", 2)
            line = dict(source_line)
            number = line.get("number", line.get("line_number", index + 1))
            if (not isinstance(number, int) or isinstance(number, bool)
                    or number < 1 or number in line_numbers):
                raise CLIError("מספרי השורות בקובץ הקלט אינם תקינים", 2)
            line_numbers.append(number)
            if line.get("image"):
                image_path = Path(line["image"]).expanduser()
                try:
                    line["img"] = base64.b64encode(
                        image_path.read_bytes()).decode("ascii")
                except OSError as exc:
                    raise CLIError(
                        f"לא ניתן לקרוא את תמונת שורה {number}: {exc}", 4) from exc
            if isinstance(line.get("img"), str) and line["img"].startswith(
                    "data:image/"):
                try:
                    line["img"] = line["img"].split(",", 1)[1]
                except IndexError as exc:
                    raise CLIError(f"תמונה לא תקינה בשורה {number}", 2) from exc
            normalized.append(line)
        selected_indices = _parse_line_selection(args.lines, line_numbers)
        if not selected_indices:
            raise CLIError("לא נבחרו שורות לשליחה", 2)
        selected = [normalized[index] for index in selected_indices]
        missing_images = [
            line_numbers[index] for index in selected_indices
            if not normalized[index].get("img")]
        if missing_images:
            raise CLIError(
                "חסרות תמונות לשורות "
                + ", ".join(map(str, missing_images))
                + '; הפק OCR עם --format json --include-images', 2)
        result = ai.correct_lines(
            selected, batch_size=args.batch_size, key_id=args.key_id,
            key_name=args.key_name, temperature=args.temperature,
            top_p=args.top_p, top_k=args.top_k,
            max_output_tokens=args.max_output_tokens,
            request_timeout=args.request_timeout)
        corrections = []
        for correction in result["corrections"]:
            selected_index = correction["index"]
            source_index = selected_indices[selected_index]
            corrections.append({
                "index": source_index,
                "line_number": line_numbers[source_index],
                "original": normalized[source_index].get("text", ""),
                "text": correction["text"],
            })
        rendered = _json({
            "schema_version": 1,
            "selected_lines": [line_numbers[index] for index in selected_indices],
            **{**result, "corrections": corrections},
        })
        if args.output:
            try:
                Path(args.output).expanduser().write_text(
                    rendered, encoding="utf-8")
            except OSError as exc:
                raise CLIError(f"לא ניתן לכתוב את קובץ הפלט: {exc}", 4) from exc
        else:
            print(rendered)


def command_config(args):
    app = _app()
    if args.action == "get":
        _output(app.load_config(), args.json)
    elif args.action == "set":
        config = app.save_config({"upload_url": args.upload_url})
        _output(config, args.json)


def command_parallel(args):
    app = _app()
    image_path = Path(args.image).expanduser()
    text_path = Path(args.text_file).expanduser()
    if not image_path.is_file():
        raise CLIError(f"קובץ הסריקה לא נמצא: {image_path}", 4)
    if not text_path.is_file():
        raise CLIError(f"קובץ הטקסט לא נמצא: {text_path}", 4)
    from PIL import Image
    import tempfile
    with Image.open(image_path) as image:
        page_image = image.convert("L")
    texts = [line.strip() for line in
             text_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    with tempfile.TemporaryDirectory() as directory:
        page_path = os.path.join(directory, "page.png")
        page_image.save(page_path)
        crops = app.segment_page(page_path)
    if len(crops) != len(texts):
        raise CLIError(
            f"זוהו {len(crops)} שורות בתמונה אך נמצאו {len(texts)} שורות טקסט; "
            "בדוק את הסריקה והטקסט, והתאם את מספר השורות לפני שמירה")
    lines = [{"index": index, "text": text,
              "img": app.img_to_b64(crop)}
             for index, (crop, text) in enumerate(zip(crops, texts))]
    if args.save:
        writer = _writer(args.writer)
        if app.training_running():
            raise CLIError("אי אפשר להוסיף דוגמאות בזמן אימון פעיל")
        saved, quarantined, duplicates = app.save_lines(writer, [
            {"img": item["img"], "text": item["text"]} for item in lines])
        app.postfix.refresh_vocab()
        result = {"saved": saved, "quarantined": quarantined,
                  "duplicates": duplicates}
    else:
        result = {"schema_version": 1, "writer": args.writer,
                  "lines": lines,
                  "note": "תצוגת טיוטה בלבד. הוסף --save לשמירה אחרי אימות.",
                  "save_hint": "בדוק ותאם את הטקסט מול התמונות לפני שמירה"}
    _output(result, args.json or not args.save)


def _read_training_text(args):
    if args.text is not None:
        text = args.text
    elif args.text_file:
        path = Path(args.text_file).expanduser()
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise CLIError(f"לא ניתן לקרוא את קובץ הטקסט: {exc}", 4) from exc
    elif sys.stdin.isatty():
        print("הקלד שורה אחת לכל שורה בתמונה; שורה ריקה מסיימת את הקלט:")
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if not line.strip():
                break
            lines.append(line)
        text = "\n".join(lines)
    else:
        raise CLIError(
            "יש להעביר --text או --text-file, או להפעיל ממסוף אינטראקטיבי", 2)
    return [line.strip() for line in text.splitlines() if line.strip()]


def _normalize_match_text(text):
    return unicodedata.normalize("NFC", " ".join(text.split())).strip()


def _edit_distance(left, right):
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, 1):
        current = [row]
        for column, right_char in enumerate(right, 1):
            current.append(min(
                previous[column] + 1,
                current[column - 1] + 1,
                previous[column - 1] + (left_char != right_char)))
        previous = current
    return previous[-1]


def _measure_page_accuracy(image_path, writer, expected_lines):
    output = StringIO()
    args = argparse.Namespace(
        include_images=False, format="json", input=str(image_path), page=True,
        writer=writer, model="personal", no_fix=True, progress=False,
        timeout=0, output=None)
    with redirect_stdout(output):
        command_ocr(args)
    try:
        payload = json.loads(output.getvalue())
        predicted_lines = [item["text"] for item in payload["lines"]]
    except (ValueError, KeyError, TypeError) as exc:
        raise CLIError(f"לא ניתן לחשב דיוק מתוצאת OCR: {exc}") from exc
    expected = "\n".join(_normalize_match_text(line)
                         for line in expected_lines)
    predicted = "\n".join(_normalize_match_text(line)
                          for line in predicted_lines)
    errors = _edit_distance(predicted, expected)
    accuracy = max(0.0, 100.0 * (1.0 - errors / max(1, len(expected))))
    return accuracy, predicted_lines


def _wait_for_training(app, deadline, round_number, current_accuracy,
                       max_hours):
    stop_requested = False
    interrupted = False
    try:
        while app.training_running():
            now = time.monotonic()
            if now >= deadline and not stop_requested:
                stop_requested = app.stop_training()
                if not stop_requested and app.training_running():
                    raise CLIError(
                        "הזמן הסתיים אך לא ניתן לעצור את האימון בצורה מסודרת")
                if stop_requested:
                    print("הזמן הוקצב; נשלחה בקשת עצירה ושמירת checkpoint.",
                          file=sys.stderr, flush=True)
            status = app.train_status()
            step, total = status.get("step", 0), status.get("total", 0)
            step_percent = 100.0 * step / total if total else 0.0
            elapsed_percent = min(
                100.0, 100.0 * (max_hours * 3600 - max(0, deadline - now))
                / (max_hours * 3600))
            print(
                f"מחזור {round_number} | אימון {step_percent:.1f}% "
                f"({step}/{total}) | זמן כולל {elapsed_percent:.1f}% "
                f"| דיוק OCR אחרון {current_accuracy:.2f}%",
                file=sys.stderr, flush=True)
            time.sleep(2)
    except KeyboardInterrupt:
        interrupted = True
        stop_requested = app.stop_training()
        if not stop_requested and app.training_running():
            raise CLIError("לא ניתן לעצור את האימון בצורה מסודרת")
        print(
            "נשלחה בקשת עצירה מסודרת; שומר checkpoint..."
            if stop_requested else "האימון כבר הסתיים.",
            file=sys.stderr, flush=True)
        while app.training_running():
            time.sleep(0.5)
    return stop_requested, interrupted


def command_train_until(args):
    app = _app()
    writer = app.records.validate_writer(_writer(args.writer))
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise CLIError(f"קובץ התמונה לא נמצא: {image_path}", 4)
    if app.training_running():
        raise CLIError(
            "כבר פועל אימון אחר. עצור אותו או המתן שיסתיים לפני הפקודה הזו.")
    if not 0 < args.accuracy <= 100:
        raise CLIError("--accuracy חייב להיות גדול מ-0 ועד 100", 2)
    if not 0 < args.max_hours <= 400:
        raise CLIError("--max-hours חייב להיות גדול מ-0 ועד 400", 2)
    if not 100 <= args.steps <= 10000:
        raise CLIError("--steps חייב להיות בין 100 ל-10000", 2)
    if not 1 <= args.batch <= 32:
        raise CLIError("--batch חייב להיות בין 1 ל-32", 2)

    texts = _read_training_text(args)
    if not texts:
        raise CLIError("לא הוזן טקסט לאימון", 2)
    from PIL import Image
    import tempfile
    with Image.open(image_path) as image:
        page_image = image.convert("L")
    with tempfile.TemporaryDirectory() as directory:
        page_path = os.path.join(directory, "training-page.png")
        page_image.save(page_path)
        crops = app.segment_page(page_path)
    if len(crops) != len(texts):
        raise CLIError(
            f"זוהו {len(crops)} שורות בתמונה אך הוזנו {len(texts)} שורות טקסט. "
            "תקן את שורות הטקסט או את הסריקה; לא נשמרו דוגמאות.")

    samples = [{"img": app.img_to_b64(crop), "text": text}
               for crop, text in zip(crops, texts)]
    saved, quarantined, duplicates = app.save_lines(writer, samples)
    app.postfix.refresh_vocab()
    if quarantined or saved + duplicates != len(samples):
        raise CLIError(
            f"הדוגמאות לא כולן אושרו: נשמרו {saved}, כפילויות {duplicates}, "
            f"בהסגר {quarantined}. בדוק את ההסגר לפני אימון.")

    accuracy, _ = _measure_page_accuracy(image_path, writer, texts)
    print(
        f"דיוק התחלתי: {accuracy:.2f}% | יעד: {args.accuracy:.2f}% "
        f"| דוגמאות חדשות: {saved} | כפילויות: {duplicates}",
        file=sys.stderr, flush=True)
    deadline = time.monotonic() + args.max_hours * 3600
    round_number = 0
    interrupted = False
    while accuracy < args.accuracy and time.monotonic() < deadline:
        round_number += 1
        resume = round_number == 1
        ok, message = _start_training(
            app, writer, "personal", resume,
            steps=args.steps, batch_size=args.batch)
        if not ok:
            raise CLIError(message)
        print(
            f"מתחיל מחזור {round_number}; checkpoint "
            f"{'ימשיך אם קיים' if resume else 'יתחיל מסבב נוסף'}; "
            f"דיוק נוכחי {accuracy:.2f}%, יעד {args.accuracy:.2f}%",
            file=sys.stderr, flush=True)
        stopped, interrupted = _wait_for_training(
            app, deadline, round_number, accuracy, args.max_hours)
        accuracy, _ = _measure_page_accuracy(image_path, writer, texts)
        print(
            f"סיום מחזור {round_number}: דיוק OCR {accuracy:.2f}% "
            f"(יעד {args.accuracy:.2f}%)",
            file=sys.stderr, flush=True)
        if accuracy >= args.accuracy or interrupted:
            break
        if stopped or time.monotonic() >= deadline:
            break

    result = {
        "writer": writer,
        "image": str(image_path),
        "target_accuracy": args.accuracy,
        "accuracy": round(accuracy, 2),
        "rounds": round_number,
        "saved_examples": saved,
        "duplicate_examples": duplicates,
        "reached_target": accuracy >= args.accuracy,
        "interrupted": interrupted,
    }
    _output(result, args.json)
    if interrupted:
        raise CLIError("האימון הופסק לבקשת המשתמש; אפשר להפעיל שוב ולהמשיך.", 130)
    if accuracy < args.accuracy:
        raise CLIError(
            f"הזמן הסתיים לפני הגעה ליעד. הדיוק האחרון: {accuracy:.2f}%; "
            "אפשר להריץ שוב עם אותו כותב כדי להמשיך.")


def _internal_train(argv):
    if len(argv) not in (5, 7):
        raise CLIError("פרמטרים לא תקינים למשימת אימון פנימית")
    _, scope, writer, run_flag, *options = argv[1:]
    steps = int(options[0]) if options else 2000
    batch_size = int(options[1]) if options else 24
    import train
    train._write_pid()
    resume = run_flag == "--resume"
    try:
        if scope == "both":
            train.chain(writer, steps, steps, resume=resume)
        elif scope == "personal":
            train.train(writer, steps, batch_size, 1e-4, resume=resume)
        elif scope == "general":
            train.train("__general__", steps, batch_size, 1e-4, resume=resume)
        else:
            raise CLIError("היקף אימון פנימי אינו תקין")
        return 0
    finally:
        train._clear_pid()
        train._clear_stop_request()


def _add_json_flag(parser):
    parser.add_argument("--json", action="store_true", help="פלט JSON יציב")


def _add_ai_settings_flags(parser):
    parser.add_argument("--batch-size", type=int,
                        help="מספר שורות בכל בקשת API (1–16)")
    parser.add_argument("--temperature", type=float,
                        help="יצירתיות המודל, 0–2")
    parser.add_argument("--top-p", type=float, help="סף top-p, בין 0 ל-1")
    parser.add_argument("--top-k", type=int, help="סף top-k, בין 1 ל-1000")
    parser.add_argument("--max-output-tokens", type=int,
                        help="תקרת טוקנים לתשובה, 1–65536")
    parser.add_argument("--request-timeout", "--timeout", dest="request_timeout",
                        type=int, help="פסק זמן לבקשת API בשניות (1–300)")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="OCRHE",
        description="OCRHE — תמלול OCR, עריכת נתוני אימון, תיקונים ומודלים.",
        epilog=("למשימות סוכנים השתמשו בדגל --json; מחיקות ושינויי מודל דורשים "
                "--yes. נתוני ברירת המחדל נמצאים ליד קובץ ההפעלה."),
    )
    parser.add_argument("--version", action="version", version=f"OCRHE {VERSION}")
    parser.add_argument("--data-dir", help="תיקיית data/models משותפת; ברירת המחדל ליד קובץ ההפעלה")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ocr", help="תמלול תמונה, PDF או תיקייה")
    p.add_argument("input")
    p.add_argument("--page", action="store_true", help="פילוח עמוד מלא לשורות")
    p.add_argument("--writer", help="כותב למודל אישי")
    p.add_argument("--model", choices=("auto", "personal", "general", "typed"), default="auto")
    p.add_argument("--format", choices=("text", "json", "jsonl", "docx"), default="text")
    p.add_argument("--include-images", action="store_true",
                   help="כלול תמונות שורה ב-JSON/JSONL, כדי לשלוח שורות נבחרות לתיקון AI")
    p.add_argument("--output", "-o")
    p.add_argument("--no-fix", action="store_true", help="ללא תיקון מילוני והצעות")
    p.add_argument("--progress", action="store_true", help="התקדמות ל-stderr")
    p.add_argument("--timeout", type=float, default=0)
    p.set_defaults(handler=command_ocr)

    p = sub.add_parser("writers", help="ניהול כותבים וחבילות אימון")
    sp = p.add_subparsers(dest="action", required=True)
    q = sp.add_parser("list"); _add_json_flag(q); q.set_defaults(handler=command_writers)
    q = sp.add_parser("add"); q.add_argument("--name"); _add_json_flag(q); q.set_defaults(handler=command_writers)
    q = sp.add_parser("delete"); q.add_argument("--name", required=True); q.add_argument("--yes", action="store_true"); _add_json_flag(q); q.set_defaults(handler=command_writers)
    q = sp.add_parser("export"); q.add_argument("--name", required=True); q.add_argument("--output", "-o", required=True); _add_json_flag(q); q.set_defaults(handler=command_writers)
    q = sp.add_parser("import"); q.add_argument("--name", required=True); q.add_argument("--input", required=True); _add_json_flag(q); q.set_defaults(handler=command_writers)
    q = sp.add_parser("purge-data"); q.add_argument("--name", required=True); q.add_argument("--yes", action="store_true"); _add_json_flag(q); q.set_defaults(handler=command_writers)
    q = sp.add_parser("share"); q.add_argument("--name", required=True); q.add_argument("--yes", action="store_true"); _add_json_flag(q); q.set_defaults(handler=command_writers)

    p = sub.add_parser("examples", help="הוספה, צפייה ועריכת דוגמאות אימון")
    sp = p.add_subparsers(dest="action", required=True)
    q = sp.add_parser("list"); q.add_argument("--writer"); q.add_argument("--offset", type=int, default=0); q.add_argument("--limit", type=int, default=25); _add_json_flag(q); q.set_defaults(handler=command_examples)
    q = sp.add_parser("add"); q.add_argument("--writer"); q.add_argument("--image", required=True); q.add_argument("--text"); q.add_argument("--text-file"); _add_json_flag(q); q.set_defaults(handler=command_examples)
    q = sp.add_parser("edit"); q.add_argument("--writer"); q.add_argument("--file", required=True); q.add_argument("--text", required=True); _add_json_flag(q); q.set_defaults(handler=command_examples)
    q = sp.add_parser("delete"); q.add_argument("--writer"); q.add_argument("--file", required=True); q.add_argument("--yes", action="store_true"); _add_json_flag(q); q.set_defaults(handler=command_examples)

    p = sub.add_parser("quarantine", help="הצגה ואישור/דחייה של דוגמאות חשודות")
    sp = p.add_subparsers(dest="action", required=True)
    q = sp.add_parser("list"); q.add_argument("--writer"); _add_json_flag(q); q.set_defaults(handler=command_quarantine)
    q = sp.add_parser("resolve"); q.add_argument("--writer"); q.add_argument("--file", required=True); q.add_argument("--decision", choices=("approve", "reject"), required=True); q.add_argument("--text"); _add_json_flag(q); q.set_defaults(handler=command_quarantine)

    p = sub.add_parser("train", help="התחלה, עצירה והמשך אימון")
    sp = p.add_subparsers(dest="action", required=True)
    q = sp.add_parser("status"); _add_json_flag(q); q.set_defaults(handler=command_train)
    q = sp.add_parser("stop"); _add_json_flag(q); q.set_defaults(handler=command_train)
    q = sp.add_parser("start"); q.add_argument("--writer"); q.add_argument("--scope", choices=("both", "personal", "general"), default="both"); q.add_argument("--steps", type=int); q.add_argument("--batch", type=int); q.add_argument("--fresh", action="store_true", help="התחלה מחדש במקום המשך"); _add_json_flag(q); q.set_defaults(handler=command_train)

    p = sub.add_parser(
        "learn",
        help="שמור תמלול של תמונה ואמן כותב עד לדיוק OCR שתבחר")
    p.add_argument("image", help="נתיב לתמונה של עמוד כתב היד")
    p.add_argument("--writer", help="שם הכותב; אם הושמט תתבקש לבחור/להקליד")
    text_source = p.add_mutually_exclusive_group()
    text_source.add_argument("--text", help="התמלול; שורות מופרדות בירידת שורה")
    text_source.add_argument("--text-file", help="קובץ UTF-8, שורה לכל שורה בתמונה")
    p.add_argument("--accuracy", type=float, required=True,
                   help="סף דיוק תווים בתמלול ה-OCR (1–100 אחוז)")
    p.add_argument("--max-hours", type=float, default=3,
                   help="משך מרבי (עד 400 שעות; ברירת מחדל 3)")
    p.add_argument("--steps", type=int, default=500,
                   help="צעדי אימון בכל מחזור (ברירת מחדל 500)")
    p.add_argument("--batch", type=int, default=2,
                   help="גודל אצוות אימון (ברירת מחדל 2)")
    _add_json_flag(p)
    p.set_defaults(handler=command_train_until)

    p = sub.add_parser("fixes", help="צפייה, עריכה והסרה של תיקוני מילון שנלמדו")
    sp = p.add_subparsers(dest="action", required=True)
    q = sp.add_parser("list"); q.add_argument("--writer"); _add_json_flag(q); q.set_defaults(handler=command_fixes)
    q = sp.add_parser("edit"); q.add_argument("--writer"); q.add_argument("--wrong", required=True); q.add_argument("--right", required=True); _add_json_flag(q); q.set_defaults(handler=command_fixes)
    q = sp.add_parser("delete"); q.add_argument("--writer"); q.add_argument("--wrong", required=True); q.add_argument("--yes", action="store_true"); _add_json_flag(q); q.set_defaults(handler=command_fixes)
    q = sp.add_parser("export"); q.add_argument("--writer"); q.add_argument("--output", "-o", required=True); _add_json_flag(q); q.set_defaults(handler=command_fixes)
    q = sp.add_parser("import"); q.add_argument("--input", required=True); _add_json_flag(q); q.set_defaults(handler=command_fixes)
    q = sp.add_parser("clear-list"); q.add_argument("--writer"); _add_json_flag(q); q.set_defaults(handler=command_fixes)

    p = sub.add_parser("templates", help="בחירת תבניות ויצירת תבניות איסוף")
    sp = p.add_subparsers(dest="action", required=True)
    q = sp.add_parser("list"); _add_json_flag(q); q.set_defaults(handler=command_templates)
    q = sp.add_parser("generate"); q.add_argument("--count", type=int, default=10); q.add_argument("--text-file", help="טקסט של משפט אחד בכל שורה לתבנית אישית"); q.add_argument("--prefix", default="custom"); _add_json_flag(q); q.set_defaults(handler=command_templates)
    q = sp.add_parser("export"); q.add_argument("--output", "-o", required=True); _add_json_flag(q); q.set_defaults(handler=command_templates)
    q = sp.add_parser("ingest"); q.add_argument("--input", required=True); q.add_argument("--template", required=True); q.add_argument("--writer"); _add_json_flag(q); q.set_defaults(handler=command_templates)

    p = sub.add_parser("models", help="רשימת מודלים ושחזור גרסה")
    sp = p.add_subparsers(dest="action", required=True)
    q = sp.add_parser("list"); _add_json_flag(q); q.set_defaults(handler=command_models)
    q = sp.add_parser("export"); q.add_argument("--name", required=True); q.add_argument("--output", "-o", required=True); _add_json_flag(q); q.set_defaults(handler=command_models)
    q = sp.add_parser("import"); q.add_argument("--name", required=True, help="base, general, typed או writer:<שם>"); q.add_argument("--input", required=True); _add_json_flag(q); q.set_defaults(handler=command_models)
    q = sp.add_parser("history"); q.add_argument("--name"); _add_json_flag(q); q.set_defaults(handler=command_models)
    q = sp.add_parser("delete-history"); q.add_argument("--file", required=True); q.add_argument("--yes", action="store_true"); _add_json_flag(q); q.set_defaults(handler=command_models)
    q = sp.add_parser("rollback"); q.add_argument("name"); q.add_argument("--yes", action="store_true"); _add_json_flag(q); q.set_defaults(handler=command_models)
    q = sp.add_parser("evaluate"); q.add_argument("--name", required=True, help="base, general, typed או writer:<שם>"); q.add_argument("--limit", type=int, default=160); _add_json_flag(q); q.set_defaults(handler=command_models)

    p = sub.add_parser("ai", help="ניהול מפתחות ובדיקת מודלי Google AI Studio")
    sp = p.add_subparsers(dest="action", required=True)
    keys = sp.add_parser("keys").add_subparsers(dest="keys_action", required=True)
    q = keys.add_parser("list"); _add_json_flag(q); q.set_defaults(handler=command_ai)
    q = keys.add_parser("add"); q.add_argument("--name", required=True); q.add_argument("--key-env"); _add_json_flag(q); q.set_defaults(handler=command_ai)
    q = keys.add_parser("remove"); q.add_argument("--id", required=True); q.add_argument("--yes", action="store_true"); _add_json_flag(q); q.set_defaults(handler=command_ai)
    q = keys.add_parser("rename"); q.add_argument("--id", required=True); q.add_argument("--name", required=True); _add_json_flag(q); q.set_defaults(handler=command_ai)
    settings = sp.add_parser("settings").add_subparsers(
        dest="settings_action", required=True)
    q = settings.add_parser("get"); _add_json_flag(q); q.set_defaults(handler=command_ai)
    q = settings.add_parser("set"); _add_ai_settings_flags(q)
    q.add_argument("--clear-top-p", action="store_true")
    q.add_argument("--clear-top-k", action="store_true")
    q.add_argument("--clear-max-output-tokens", action="store_true")
    _add_json_flag(q); q.set_defaults(handler=command_ai)
    q = settings.add_parser("reset"); _add_json_flag(q); q.set_defaults(handler=command_ai)
    q = sp.add_parser("models"); q.add_argument("--id", required=True); _add_json_flag(q); q.set_defaults(handler=command_ai)
    q = sp.add_parser("test"); q.add_argument("--id", required=True); q.add_argument("--model", required=True); _add_json_flag(q); q.set_defaults(handler=command_ai)
    q = sp.add_parser("check"); q.add_argument("--id", required=True); _add_json_flag(q); q.set_defaults(handler=command_ai)
    q = sp.add_parser("set-model"); q.add_argument("--id", required=True); q.add_argument("--model", required=True); _add_json_flag(q); q.set_defaults(handler=command_ai)
    q = sp.add_parser("correct"); q.add_argument("--input", required=True, help="תוצאת OCR עם --include-images או מערך JSON של שורות")
    q.add_argument("--output", "-o"); q.add_argument("--lines", help="מספרי שורות, למשל 1,3-5; ברירת מחדל: כולן")
    selector = q.add_mutually_exclusive_group()
    selector.add_argument("--key-id", help="מזהה של מפתח API שמור")
    selector.add_argument("--key-name", help="שם של מפתח API שמור")
    _add_ai_settings_flags(q); q.set_defaults(handler=command_ai)

    p = sub.add_parser("config", help="צפייה והגדרת תצורת העלאה")
    sp = p.add_subparsers(dest="action", required=True)
    q = sp.add_parser("get"); _add_json_flag(q); q.set_defaults(handler=command_config)
    q = sp.add_parser("set"); q.add_argument("--upload-url", required=True); _add_json_flag(q); q.set_defaults(handler=command_config)

    p = sub.add_parser("parallel", help="הצמד טקסט מוקלד לשורות סרוקות")
    p.add_argument("--image", required=True); p.add_argument("--text-file", required=True)
    p.add_argument("--writer"); p.add_argument("--save", action="store_true",
        help="שמור מיד דרך שערי בטיחות; השתמש רק לאחר בדיקת ההתאמה")
    _add_json_flag(p); p.set_defaults(handler=command_parallel)

    return parser


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    argv = list(sys.argv if argv is None else ["OCRHE", *argv])
    if len(argv) > 1 and argv[1] == "__train_child__":
        try:
            return _internal_train(argv)
        except Exception as exc:
            print(f"OCRHE training failed: {exc}", file=sys.stderr)
            return 1
    parser = build_parser()
    args = parser.parse_args(argv[1:])
    if args.data_dir:
        os.environ["OCRHE_HOME"] = os.path.abspath(os.path.expanduser(args.data_dir))
    try:
        args.handler(args)
        return 0
    except CLIError as exc:
        print(f"OCRHE: {exc}", file=sys.stderr)
        return exc.code
    except Exception as exc:
        print(f"OCRHE: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
