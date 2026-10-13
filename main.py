# -*- coding: utf-8 -*-
"""נקודת הכניסה של התוכנה — זהו הקובץ שנארז ל-EXE.

שני מצבים:
  KtavYadAI.exe               ← פותח את התוכנה (שרת + חלון)
  KtavYadAI.exe --train <שם>  ← מריץ שרשרת אימון (אישי+כללי) ברקע —
                                 כך התוכנה מפעילה אימון עצמאי גם כשהיא ארוזה
"""
import sys
import multiprocessing
import logging
import os
import time
from logging.handlers import RotatingFileHandler


def _process_age_seconds():
    if os.name != "nt":
        return None

    import ctypes
    from ctypes import wintypes

    class FileTime(ctypes.Structure):
        _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetProcessTimes.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(FileTime),
        ctypes.POINTER(FileTime),
        ctypes.POINTER(FileTime),
        ctypes.POINTER(FileTime),
    )
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.GetSystemTimeAsFileTime.argtypes = (ctypes.POINTER(FileTime),)
    created, exited, kernel, user, current = (FileTime() for _ in range(5))
    if not kernel32.GetProcessTimes(
            kernel32.GetCurrentProcess(), ctypes.byref(created),
            ctypes.byref(exited), ctypes.byref(kernel), ctypes.byref(user)):
        raise ctypes.WinError(ctypes.get_last_error())
    kernel32.GetSystemTimeAsFileTime(ctypes.byref(current))
    created_ticks = (created.high << 32) | created.low
    current_ticks = (current.high << 32) | current.low
    return max(0.0, (current_ticks - created_ticks) / 10_000_000)


def _configure_startup_log():
    if getattr(sys, "frozen", False):
        base_dir = os.path.dirname(os.path.abspath(sys.executable))
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    log_path = os.path.join(data_dir, "startup.log")
    logger = logging.getLogger("ktavyadai.startup")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = RotatingFileHandler(
            log_path, maxBytes=2 * 1024 * 1024, backupCount=3,
            encoding="utf-8")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.info("========== application startup ==========")
    logger.info("version=1.0.1 frozen=%s args=%r", getattr(sys, "frozen", False),
                sys.argv[1:])
    logger.info("log_path=%s", log_path)
    try:
        logger.info("process_age_at_python_start_seconds=%.2f",
                    _process_age_seconds())
    except OSError:
        logger.exception("could not measure process start time")
    return logger, log_path


def _show_startup_error(message):
    if os.name == "nt":
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, message, "כתב-יד AI", 0x10)


def _run_startup_diagnostics(app, logger):
    started = time.perf_counter()
    import torch

    logger.info("PyTorch imported: version=%s device=%s",
                torch.__version__, app.DEVICE)
    model, model_name = app.get_model(prefer="general")
    if model is None:
        raise RuntimeError("The bundled general OCR model could not be loaded")
    logger.info("general model loaded: %s", model_name)

    server = app._create_server()
    try:
        import threading
        from urllib.request import urlopen

        threading.Thread(target=server.serve_forever, daemon=True).start()
        base_url = "http://127.0.0.1:%d" % server.server_port
        with urlopen(base_url + "/", timeout=10) as response:
            if response.status != 200 or b"1.0.1" not in response.read():
                raise RuntimeError("The local application page failed its startup check")
        with urlopen(base_url + "/favicon.ico", timeout=10) as response:
            if response.status != 200 or response.read(4) != bytes((0, 0, 1, 0)):
                raise RuntimeError("The bundled application icon failed its startup check")
        logger.info("local HTTP server and bundled icon passed startup checks")
    finally:
        server.shutdown()
        server.server_close()

    logger.info("startup diagnostic passed in %.2f seconds",
                time.perf_counter() - started)


def run():
    logger, log_path = _configure_startup_log()
    started = time.perf_counter()
    try:
        phase = time.perf_counter()
        import paths  # noqa: F401 — מאתחל תיקיות נתונים + זריעה ראשונה
        logger.info("paths/data initialization completed in %.2f seconds",
                    time.perf_counter() - phase)

        if "--train" in sys.argv:
            i = sys.argv.index("--train")
            writer = sys.argv[i + 1] if i + 1 < len(sys.argv) else None
            if writer:
                import train
                resume = "--fresh" not in sys.argv
                if "--personal-only" in sys.argv or "--general-only" in sys.argv:
                    target = ("__general__" if "--general-only" in sys.argv else writer)
                    train._write_pid()
                    try:
                        train.train(target, 2000, 24, 1e-4, resume=resume)
                    except SystemExit:
                        pass
                    finally:
                        train._clear_pid()
                        train._clear_stop_request()
                else:
                    train.chain(writer, resume=resume)
            return 0

        phase = time.perf_counter()
        import app
        logger.info("application/Torch import completed in %.2f seconds",
                    time.perf_counter() - phase)
        if "--diagnose-startup" in sys.argv:
            _run_startup_diagnostics(app, logger)
            return 0

        logger.info("starting application interface")
        app.main()
        logger.info("application interface closed")
        return 0
    except Exception as exc:
        logger.exception("startup failed after %.2f seconds: %s",
                         time.perf_counter() - started, exc)
        if "--train" not in sys.argv and "--diagnose-startup" not in sys.argv:
            _show_startup_error(
                f"התוכנה נכשלה בהפעלה.\n\n{exc}\n\n"
                f"יומן האבחון נשמר כאן:\n{log_path}")
        return 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(run())
