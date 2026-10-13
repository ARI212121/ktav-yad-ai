"""Google AI Studio correction requests and locally encrypted API-key storage."""
import base64
import ctypes
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from ctypes import POINTER, Structure, byref, c_ubyte, c_void_p, c_wchar_p, cast, create_string_buffer, string_at
from ctypes import wintypes

from paths import DATA_DIR

# Default for newly added keys and legacy key records without a saved model.
# AI requests use each key's saved model selection.
DEFAULT_MODEL = "gemini-3.1-flash-lite"
LEGACY_DEFAULT_MODEL = "gemini-2.5-flash"
BATCH_SIZE = 8
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 16
REQUEST_TIMEOUT = 90
MAX_REQUEST_TIMEOUT = 300
CONNECTION_CHECK_TIMEOUT = 15
MODEL_TEST_TIMEOUT = 25
MODEL_LIST_PAGE_SIZE = 1000
DEFAULT_SETTINGS = {
    "batch_size": BATCH_SIZE,
    "temperature": 0.0,
    "top_p": None,
    "top_k": None,
    "max_output_tokens": None,
    "request_timeout": REQUEST_TIMEOUT,
}
SETTING_FIELDS = frozenset(DEFAULT_SETTINGS)
MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,100}$")
AUDIO_MODEL_RE = re.compile(
    r"(?:^|[-_.])(audio|tts|transcribe|transcription|speech|lyria)(?:[-_.]|$)",
    re.IGNORECASE)
_lock = threading.RLock()
_next_key = 0


class AIStudioError(RuntimeError):
    pass


KEYRING_SERVICE = "KtavYadAI Google AI Studio"


class _DataBlob(Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", POINTER(c_ubyte))]


def _keys_path():
    return os.path.join(DATA_DIR, "ai_studio_keys.json")


def _settings_path():
    return os.path.join(DATA_DIR, "ai_studio_settings.json")


def _validate_settings(settings):
    if not isinstance(settings, dict):
        raise AIStudioError("הגדרות תיקון AI חייבות להיות אובייקט")
    unknown = set(settings) - SETTING_FIELDS
    if unknown:
        raise AIStudioError(
            "הגדרות תיקון AI לא מוכרות: " + ", ".join(sorted(unknown)))
    merged = {**DEFAULT_SETTINGS, **settings}

    batch_size = merged["batch_size"]
    if (not isinstance(batch_size, int) or isinstance(batch_size, bool)
            or not MIN_BATCH_SIZE <= batch_size <= MAX_BATCH_SIZE):
        raise AIStudioError(
            f"גודל אצווה חייב להיות מספר שלם בין {MIN_BATCH_SIZE} ל-{MAX_BATCH_SIZE}")
    temperature = merged["temperature"]
    if (not isinstance(temperature, (int, float))
            or isinstance(temperature, bool) or not 0 <= temperature <= 2):
        raise AIStudioError("temperature חייב להיות בין 0 ל-2")
    for name, upper in (("top_p", 1),):
        value = merged[name]
        if value is not None and (
                not isinstance(value, (int, float))
                or isinstance(value, bool) or not 0 <= value <= upper):
            raise AIStudioError(f"{name} חייב להיות בין 0 ל-{upper}")
    for name, lower, upper in (
            ("top_k", 1, 1000),
            ("max_output_tokens", 1, 65536),
            ("request_timeout", 1, MAX_REQUEST_TIMEOUT)):
        value = merged[name]
        if value is not None and (
                not isinstance(value, int) or isinstance(value, bool)
                or not lower <= value <= upper):
            raise AIStudioError(
                f"{name} חייב להיות מספר שלם בין {lower} ל-{upper}")
    return merged


def get_settings():
    path = _settings_path()
    try:
        with open(path, encoding="utf-8") as stream:
            stored = json.load(stream)
    except FileNotFoundError:
        return dict(DEFAULT_SETTINGS)
    except (OSError, ValueError) as exc:
        raise AIStudioError(f"לא ניתן לקרוא את הגדרות תיקון AI: {exc}") from exc
    return _validate_settings(stored)


def save_settings(settings):
    if not isinstance(settings, dict) or not settings:
        raise AIStudioError("יש לציין לפחות הגדרה אחת לעדכון")
    merged = _validate_settings({**get_settings(), **settings})
    path = _settings_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(merged, stream, ensure_ascii=False, indent=1)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return merged


def reset_settings():
    path = _settings_path()
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise AIStudioError(f"לא ניתן לאפס את הגדרות תיקון AI: {exc}") from exc
    return dict(DEFAULT_SETTINGS)


def _crypt(data, protect):
    if os.name != "nt":
        raise AIStudioError("שמירת מפתחות API מוגנת זמינה כרגע ב-Windows בלבד")
    source = create_string_buffer(data)
    source_blob = _DataBlob(len(data), cast(source, POINTER(c_ubyte)))
    result_blob = _DataBlob()
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    if protect:
        function = crypt32.CryptProtectData
        function.argtypes = [
            POINTER(_DataBlob), c_wchar_p, POINTER(_DataBlob), c_void_p,
            c_void_p, wintypes.DWORD, POINTER(_DataBlob)]
        args = (byref(source_blob), "KtavYadAI Google AI Studio key",
                None, None, None, 1, byref(result_blob))
    else:
        function = crypt32.CryptUnprotectData
        function.argtypes = [
            POINTER(_DataBlob), c_void_p, POINTER(_DataBlob), c_void_p,
            c_void_p, wintypes.DWORD, POINTER(_DataBlob)]
        args = (byref(source_blob), None, None, None, None, 1,
                byref(result_blob))
    function.restype = wintypes.BOOL
    if not function(*args):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return string_at(result_blob.pbData, result_blob.cbData)
    finally:
        kernel32.LocalFree.argtypes = [c_void_p]
        kernel32.LocalFree.restype = c_void_p
        kernel32.LocalFree(cast(result_blob.pbData, c_void_p))


def _read_keys():
    try:
        with open(_keys_path(), encoding="utf-8") as f:
            store = json.load(f)
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as exc:
        raise AIStudioError(f"לא ניתן לקרוא את קובץ מפתחות AI Studio: {exc}") from exc
    if not isinstance(store, dict) or not isinstance(store.get("keys"), list):
        raise AIStudioError("קובץ מפתחות AI Studio פגום; יש לבדוק את תיקיית data")
    migrated = False
    for item in store["keys"]:
        if (not isinstance(item, dict)
                or not all(isinstance(item.get(field), str)
                           for field in ("id", "name"))
                or not (isinstance(item.get("encrypted"), str)
                        or isinstance(item.get("secret_ref"), str))
                or ("model" in item and not isinstance(item["model"], str))):
            raise AIStudioError("רשומת מפתח AI Studio פגומה; יש לבדוק את תיקיית data")
        if (item.get("model") in (None, LEGACY_DEFAULT_MODEL)
                or AUDIO_MODEL_RE.search(item["model"])):
            item["model"] = DEFAULT_MODEL
            migrated = True
    if migrated:
        _write_keys(store["keys"])
    return store["keys"]


def _write_keys(keys):
    path = _keys_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as f:
            json.dump({"keys": keys}, f, ensure_ascii=False, indent=1)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def list_keys():
    with _lock:
        return [{"id": item["id"], "name": item["name"],
                 "model": item["model"]}
                for item in _read_keys()]


def rename_key(key_id, name):
    name = (name or "").strip()
    if not name or len(name) > 60:
        raise AIStudioError("יש לתת למפתח שם באורך 1–60 תווים")
    with _lock:
        keys = _read_keys()
        item = next((key for key in keys if key["id"] == key_id), None)
        if item is None:
            raise AIStudioError("מפתח ה-API לא נמצא")
        if any(key["id"] != key_id and
               key["name"].casefold() == name.casefold() for key in keys):
            raise AIStudioError("כבר קיים מפתח בשם הזה")
        item["name"] = name
        _write_keys(keys)
        return {"id": item["id"], "name": item["name"],
                "model": item["model"]}


def add_key(name, api_key):
    name = (name or "").strip()
    api_key = (api_key or "").strip()
    if not name or len(name) > 60:
        raise AIStudioError("יש לתת למפתח שם באורך 1–60 תווים")
    if not api_key or len(api_key) > 1024:
        raise AIStudioError("מפתח API חסר או אינו תקין")
    with _lock:
        keys = _read_keys()
        if any(item["name"].casefold() == name.casefold() for item in keys):
            raise AIStudioError("כבר קיים מפתח בשם הזה")
        key_id = uuid.uuid4().hex
        if os.name == "nt":
            encrypted = base64.b64encode(
                _crypt(api_key.encode("utf-8"), True)).decode("ascii")
            item = {"id": key_id, "name": name, "encrypted": encrypted}
        else:
            try:
                import keyring
                keyring.set_password(KEYRING_SERVICE, key_id, api_key)
                if keyring.get_password(KEYRING_SERVICE, key_id) != api_key:
                    raise AIStudioError(
                        "מנהל הסודות של מערכת ההפעלה לא אישר שמירה בטוחה")
            except AIStudioError:
                raise
            except Exception as exc:
                raise AIStudioError(
                    "לא ניתן לשמור מפתח בצורה מאובטחת במנהל הסודות של מערכת "
                    f"ההפעלה: {exc}") from exc
            item = {"id": key_id, "name": name, "secret_ref": key_id}
        item["model"] = DEFAULT_MODEL
        keys.append(item)
        try:
            _write_keys(keys)
        except OSError as write_error:
            if os.name != "nt":
                try:
                    import keyring
                    keyring.delete_password(KEYRING_SERVICE, key_id)
                except Exception as cleanup_error:
                    raise AIStudioError(
                        "שמירת המפתח נכשלה, וגם לא ניתן לנקות את הערך הזמני "
                        f"ממנהל הסודות: {cleanup_error}") from write_error
            raise
        return [{"id": item["id"], "name": item["name"],
                 "model": item["model"]} for item in keys]


def remove_key(key_id):
    with _lock:
        keys = _read_keys()
        updated = [item for item in keys if item.get("id") != key_id]
        if len(updated) == len(keys):
            raise AIStudioError("מפתח ה-API לא נמצא")
        removed = next(item for item in keys if item.get("id") == key_id)
        _write_keys(updated)
        if os.name != "nt" and removed.get("secret_ref"):
            try:
                import keyring
                keyring.delete_password(KEYRING_SERVICE, removed["secret_ref"])
            except Exception as exc:
                raise AIStudioError(
                    f"המפתח הוסר מהרשימה אך לא ניתן למחוק אותו ממנהל הסודות: {exc}") from exc
        return [{"id": item["id"], "name": item["name"],
                 "model": item["model"]} for item in updated]


def set_key_model(key_id, model):
    model = (model or "").strip()
    if (not MODEL_NAME_RE.fullmatch(model)
            or AUDIO_MODEL_RE.search(model)):
        raise AIStudioError("שם מודל אינו תקין")
    with _lock:
        keys = _read_keys()
        item = next((key for key in keys if key["id"] == key_id), None)
        if item is None:
            raise AIStudioError("מפתח ה-API לא נמצא")
        item["model"] = model
        _write_keys(keys)
        return {"id": item["id"], "name": item["name"], "model": model}


def _decrypt_key(item):
    try:
        if item.get("secret_ref"):
            import keyring
            secret = keyring.get_password(KEYRING_SERVICE, item["secret_ref"])
            if not secret:
                raise AIStudioError(
                    f"מפתח '{item['name']}' חסר במנהל הסודות של מערכת ההפעלה")
            return secret
        return _crypt(base64.b64decode(item["encrypted"]), False).decode("utf-8")
    except AIStudioError:
        raise
    except (ValueError, OSError) as exc:
        raise AIStudioError(
            f"לא ניתן לפענח את מפתח '{item['name']}'. "
            "יש לבדוק את מנהל הסודות ואת חשבון מערכת ההפעלה.") from exc


def _error_payload(exc):
    try:
        payload = json.loads(exc.read().decode("utf-8", "replace"))
    except (ValueError, OSError):
        payload = {}
    error = payload.get("error", {}) if isinstance(payload, dict) else {}
    message = error.get("message", "שגיאה לא מפורטת מהשירות")
    state = str(error.get("status", "")).upper()
    reasons = []
    for detail in error.get("details", []):
        if isinstance(detail, dict):
            reasons.extend(str(detail.get(field, ""))
                           for field in ("reason", "type"))
    reason = " ".join(reasons).upper()
    text = f"{state} {reason} {message}".upper()
    if exc.code == 429 or "RESOURCE_EXHAUSTED" in text or "QUOTA" in text:
        category = "quota"
    elif exc.code == 404:
        category = "model_unavailable"
    elif (exc.code in (400, 401, 403)
          and any(term in text for term in (
              "API_KEY_INVALID", "API KEY NOT VALID", "PERMISSION_DENIED",
              "API_KEY_SERVICE_BLOCKED", "SERVICE_DISABLED"))):
        category = "rejected"
    else:
        category = "service_error"
    return category, _translate_google_error(message, exc.code, text)


def _list_models(api_key):
    models = []
    page_token = None
    seen_tokens = set()
    while True:
        query = {"pageSize": MODEL_LIST_PAGE_SIZE}
        if page_token:
            query["pageToken"] = page_token
        url = ("https://generativelanguage.googleapis.com/v1beta/models?"
               + urllib.parse.urlencode(query))
        request = urllib.request.Request(
            url, headers={"x-goog-api-key": api_key}, method="GET")
        try:
            with urllib.request.urlopen(
                    request, timeout=CONNECTION_CHECK_TIMEOUT) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            category, message = _error_payload(exc)
            if category == "quota":
                raise _RotateKey(f"מכסת השימוש במפתח הסתיימה ({message})") from exc
            if category == "rejected":
                raise _RotateKey(f"מפתח API נדחה על ידי Google ({message})") from exc
            raise AIStudioError(
                f"Google AI Studio לא הצליח להחזיר רשימת מודלים "
                f"({exc.code}): {message}") from exc
        except urllib.error.URLError as exc:
            raise AIStudioError(
                f"לא ניתן להתחבר ל-Google AI Studio: {exc.reason}") from exc
        except (OSError, TimeoutError, ValueError, UnicodeDecodeError) as exc:
            raise AIStudioError(
                f"תשובה לא תקינה בעת טעינת מודלים: {exc}") from exc
        if not isinstance(result, dict) or not isinstance(
                result.get("models", []), list):
            raise AIStudioError("Google החזיר רשימת מודלים בפורמט לא צפוי")
        for item in result.get("models", []):
            if not isinstance(item, dict):
                continue
            full_name = item.get("name")
            methods = item.get("supportedGenerationMethods", [])
            if (not isinstance(full_name, str)
                    or not full_name.startswith("models/")
                    or not isinstance(methods, list)
                    or "generateContent" not in methods):
                continue
            model_id = full_name[len("models/"):]
            if not MODEL_NAME_RE.fullmatch(model_id):
                continue
            if AUDIO_MODEL_RE.search(model_id):
                continue
            models.append({
                "id": model_id,
                "display_name": item.get("displayName") or model_id,
                "description": item.get("description") or "",
                "input_token_limit": item.get("inputTokenLimit"),
                "output_token_limit": item.get("outputTokenLimit"),
            })
        next_token = result.get("nextPageToken")
        if not next_token:
            break
        if not isinstance(next_token, str) or next_token in seen_tokens:
            raise AIStudioError("Google החזיר עימוד מודלים לא תקין")
        seen_tokens.add(next_token)
        page_token = next_token
    return sorted(models, key=lambda item: item["id"].casefold())


def list_available_models(key_id):
    with _lock:
        item = next((key for key in _read_keys() if key["id"] == key_id), None)
    if item is None:
        raise AIStudioError("מפתח ה-API לא נמצא")
    return {
        "id": item["id"],
        "name": item["name"],
        "selected_model": item["model"],
        "models": _list_models(_decrypt_key(item)),
    }


def test_model(key_id, model_id):
    model_id = (model_id or "").strip()
    if not MODEL_NAME_RE.fullmatch(model_id) or AUDIO_MODEL_RE.search(model_id):
        raise AIStudioError("מודל הבדיקה אינו תקין או שאינו מודל טקסט")
    with _lock:
        item = next((key for key in _read_keys() if key["id"] == key_id), None)
    if item is None:
        raise AIStudioError("מפתח ה-API לא נמצא")

    api_key = _decrypt_key(item)
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": "Reply with the single word OK."}],
        }],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 8},
    }
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_id}:generateContent",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "x-goog-api-key": api_key},
        method="POST")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=MODEL_TEST_TIMEOUT) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        category, message = _error_payload(exc)
        if exc.code == 400:
            category = "unsupported"
        elif exc.code == 401 or exc.code == 403:
            category = "access_denied"
        elif exc.code == 404:
            category = "unavailable"
        elif exc.code == 503:
            category = "overloaded"
        return {
            "id": model_id,
            "status": category,
            "detail": message.replace(api_key, "[מפתח מוסתר]"),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
    except urllib.error.URLError as exc:
        return {
            "id": model_id,
            "status": "unreachable",
            "detail": f"לא ניתן להתחבר ל-Google: {exc.reason}",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
    except (OSError, TimeoutError, ValueError, UnicodeDecodeError) as exc:
        return {
            "id": model_id,
            "status": "service_error",
            "detail": f"תשובת Google אינה תקינה: {exc}",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }

    text = ""
    try:
        candidates = result["candidates"]
        parts = candidates[0]["content"]["parts"]
        text = "".join(part.get("text", "") for part in parts
                       if isinstance(part, dict))
    except (KeyError, IndexError, TypeError):
        pass
    status = "working" if text.strip() else "no_response"
    return {
        "id": model_id,
        "status": status,
        "detail": (f"המודל החזיר: {text.strip()}" if text.strip()
                   else "Google קיבל את הבקשה אך לא החזיר טקסט."),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def _probe_key(key_id, name, api_key, model):
    started = time.perf_counter()
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}",
        headers={"x-goog-api-key": api_key},
        method="GET")
    checked_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        with urllib.request.urlopen(
                request, timeout=CONNECTION_CHECK_TIMEOUT) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        category, message = _error_payload(exc)
        status = category
        detail = message
    except urllib.error.URLError as exc:
        status = "unreachable"
        detail = f"לא ניתן להתחבר ל-Google: {exc.reason}"
    except (OSError, TimeoutError) as exc:
        status = "unreachable"
        detail = f"שגיאת תקשורת: {exc}"
    except (ValueError, UnicodeDecodeError):
        status = "service_error"
        detail = "Google החזיר תשובה שאינה JSON תקין."
    else:
        supported = (result.get("supportedGenerationMethods", [])
                     if isinstance(result, dict) else [])
        if (isinstance(result, dict)
                and result.get("name") == f"models/{model}"
                and isinstance(supported, list)
                and "generateContent" in supported):
            status = "connected"
            detail = f"המפתח מורשה למודל {model}; מכסת יצירת התוכן לא נבדקה."
        else:
            status = "service_error"
            detail = "Google החזיר פרטי מודל בפורמט לא צפוי."
    return {
        "id": key_id,
        "name": name,
        "model": model,
        "status": status,
        "detail": detail,
        "checked_at": checked_at,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def check_key(key_id):
    with _lock:
        item = next((key for key in _read_keys() if key["id"] == key_id), None)
    if item is None:
        raise AIStudioError("מפתח ה-API לא נמצא")
    return _probe_key(item["id"], item["name"], _decrypt_key(item),
                      item["model"])


def check_all_keys():
    with _lock:
        keys = list(_read_keys())
    return [_probe_key(item["id"], item["name"], _decrypt_key(item),
                       item["model"])
            for item in keys]


def _api_error_message(payload, status):
    error = payload.get("error", {}) if isinstance(payload, dict) else {}
    message = error.get("message", "שגיאה לא מפורטת מהשירות")
    reasons = []
    for detail in error.get("details", []):
        if isinstance(detail, dict):
            reasons.extend(str(detail.get(field, "")) for field in ("reason", "type"))
    reason = " ".join(reasons).upper()
    state = str(error.get("status", "")).upper()
    error_text = f"{state} {reason} {message}".upper()
    if (status == 429 or "RESOURCE_EXHAUSTED" in error_text
            or "QUOTA" in error_text or "RATE LIMIT" in error_text):
        raise _RotateKey(f"מכסת השימוש במפתח הסתיימה ({message})")
    if (status == 401 or "API_KEY_INVALID" in reason
            or "API_KEY_INVALID" in state or "API KEY NOT VALID" in error_text
            or "PERMISSION_DENIED" in error_text
            or "API_KEY_SERVICE_BLOCKED" in error_text
            or "SERVICE_DISABLED" in error_text):
        raise _RotateKey("מפתח API נדחה על ידי Google")
    if status == 404 or "MODEL" in error_text and (
            "NOT_FOUND" in error_text or "NOT SUPPORTED" in error_text):
        raise _RotateKey(
            "המודל שנבחר אינו זמין למפתח הזה; יש לרענן את רשימת המודלים")
    translated = _translate_google_error(message, status, error_text)
    raise AIStudioError(f"Google AI Studio: {translated}")


class _RotateKey(AIStudioError):
    pass


def _translate_google_error(message, status, error_text=""):
    normalized = f"{status} {error_text} {message}".casefold()
    if (status == 503 or "high demand" in normalized
            or "temporarily unavailable" in normalized
            or "service unavailable" in normalized):
        return (
            "שירות Google AI Studio עמוס כרגע (שגיאה 503). "
            "זו בדרך כלל תקלה זמנית, ולא בהכרח בעיה במפתח או בזמינות המודל. "
            "נסה שוב בעוד כמה דקות.")
    return message


def _request_gemini(api_key, lines, model=DEFAULT_MODEL, *, temperature=0.0,
                    top_p=None, top_k=None, max_output_tokens=None,
                    request_timeout=REQUEST_TIMEOUT):
    parts = [{
        "text": (
            "תקן תמלול OCR בעברית על פי התמונות המצורפות. לכל תמונה מצורפים "
            "מספר שורה והטקסט שזוהה כרגע. החזר את הטקסט המדויק שרואים בתמונה; "
            "קרא עברית מימין לשמאל. סדר המילים והשדות בתשובה חייב להתאים "
            "לסדר החזותי בתמונה, מימין לשמאל; אל תהפוך את סדר השדות לפי "
            "ניסוח שנראה טבעי יותר. שמור כל תווית צמודה לקו או לערך שמופיע "
            "לידה, ואל תחליף ביניהם. לדוגמה, אם מופיע מימין לשמאל "
            "'תפקיד: ____' ואז 'שם מורשה חתימה: ____', החזר "
            "'תפקיד: ____ שם מורשה חתימה: ____' ולא את הסדר ההפוך. "
            "אל תשלים תוכן חסר ואל תנחש מילים שאינן קריאות. שמור על "
            "מספרים, סימני פיסוק, מקפים, סוגריים, קווים תחתונים ושדות ריקים. "
            "אם התמלול הנוכחי נכון או שהתמונה אינה מאפשרת תיקון בטוח, החזר "
            "אותו ללא שינוי. החזר JSON בלבד במבנה "
            '{"corrections":[{"index":0,"text":"..."}]}.'
        )
    }]
    for index, item in enumerate(lines):
        line_number = item.get("line_number", item.get("number", index + 1))
        parts.append({
            "text": f"שורה {line_number}; תמלול נוכחי: {item['text']}"})
        parts.append({"inline_data": {
            "mime_type": "image/png",
            "data": item["img"],
        }})
    generation_config = {
        "temperature": temperature,
        "responseMimeType": "application/json",
        "responseSchema": {
            "type": "OBJECT",
            "properties": {
                "corrections": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "index": {"type": "INTEGER"},
                            "text": {"type": "STRING"},
                        },
                        "required": ["index", "text"],
                    },
                },
            },
            "required": ["corrections"],
        },
    }
    if top_p is not None:
        generation_config["topP"] = top_p
    if top_k is not None:
        generation_config["topK"] = top_k
    if max_output_tokens is not None:
        generation_config["maxOutputTokens"] = max_output_tokens
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": generation_config,
    }
    request_data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent",
        data=request_data,
        headers={"Content-Type": "application/json",
                 "x-goog-api-key": api_key},
        method="POST")
    try:
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=request_timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        elapsed = time.perf_counter() - started
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read().decode("utf-8", "replace"))
        except (ValueError, OSError):
            body = {}
        _api_error_message(body, exc.code)
    except urllib.error.URLError as exc:
        raise AIStudioError(f"לא ניתן להתחבר ל-Google AI Studio: {exc.reason}") from exc
    if not isinstance(result, dict):
        raise AIStudioError("תשובת Google AI Studio אינה בפורמט הצפוי")
    try:
        text = "".join(part["text"] for part in result["candidates"][0]["content"]["parts"]
                       if "text" in part)
        decoded = json.loads(text)
        corrections = decoded["corrections"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise AIStudioError("תשובת Google AI Studio אינה בפורמט הצפוי") from exc
    if not isinstance(corrections, list):
        raise AIStudioError("תשובת Google AI Studio אינה כוללת רשימת תיקונים")
    parsed = {}
    for item in corrections:
        if not isinstance(item, dict):
            raise AIStudioError("Google AI Studio החזיר תיקון לא תקין")
        index, text = item.get("index"), item.get("text")
        if (not isinstance(index, int) or isinstance(index, bool)
                or index < 0 or index >= len(lines)
                or not isinstance(text, str) or not text.strip()
                or index in parsed):
            raise AIStudioError("Google AI Studio החזיר תיקון לא תקין")
        parsed[index] = text.strip()
    usage = result.get("usageMetadata", {})
    if not isinstance(usage, dict):
        usage = {}
    return parsed, {
        "request_bytes": len(request_data),
        "prompt_tokens": usage.get("promptTokenCount"),
        "output_tokens": usage.get("candidatesTokenCount"),
        "thoughts_tokens": usage.get("thoughtsTokenCount"),
        "total_tokens": usage.get("totalTokenCount"),
        "elapsed_seconds": round(elapsed, 3),
    }


def _correct_batch(lines, generation, key_id=None, key_name=None):
    global _next_key
    if key_id and key_name:
        raise AIStudioError("בחר מפתח לפי מזהה או לפי שם, לא את שניהם")
    with _lock:
        keys = list(_read_keys())
        if not keys:
            raise AIStudioError("יש להוסיף מפתח Google AI Studio בהגדרות AI")
        if key_id:
            keys = [item for item in keys if item["id"] == key_id]
            if not keys:
                raise AIStudioError("מפתח ה-API שנבחר לא נמצא")
        elif key_name:
            keys = [item for item in keys
                    if item["name"].casefold() == key_name.casefold()]
            if not keys:
                raise AIStudioError(f'לא נמצא מפתח API בשם "{key_name}"')
        else:
            start = _next_key % len(keys)
            keys = keys[start:] + keys[:start]
            _next_key = (start + 1) % len(keys)
    failures = []
    for item in keys:
        try:
            key = _decrypt_key(item)
            model = item["model"]
            corrections, usage = _request_gemini(
                key, lines, model, temperature=generation["temperature"],
                top_p=generation["top_p"], top_k=generation["top_k"],
                max_output_tokens=generation["max_output_tokens"],
                request_timeout=generation["request_timeout"])
            return item["name"], model, corrections, usage
        except _RotateKey as exc:
            failures.append(f"{item['name']}: {exc}")
    if failures and all("המודל שנבחר אינו זמין" in failure
                        for failure in failures):
        raise AIStudioError(
            "המודל שנבחר אינו זמין למפתחות שנבדקו. פתח את ניהול מפתחות "
            "AI, טען מחדש את רשימת המודלים עבור כל מפתח ובחר מודל זמין. "
            "פרטים: " + " | ".join(failures))
    raise AIStudioError(
        f"אף אחד מ-{len(failures)} מפתחות ה-API לא הצליח להשלים את הבקשה. "
        "הסיבות שדווחו: " + " | ".join(failures))


def correct_lines(lines, batch_size=None, *, key_id=None, key_name=None,
                  temperature=None, top_p=None, top_k=None,
                  max_output_tokens=None, request_timeout=None):
    if not isinstance(lines, list) or not lines:
        raise AIStudioError("אין שורות לשליחה לתיקון AI")
    overrides = {
        name: value for name, value in {
            "batch_size": batch_size,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_output_tokens": max_output_tokens,
            "request_timeout": request_timeout,
        }.items() if value is not None
    }
    generation = _validate_settings({**get_settings(), **overrides})
    batch_size = generation["batch_size"]
    if key_id is not None and not isinstance(key_id, str):
        raise AIStudioError("מזהה מפתח ה-API אינו תקין")
    if key_name is not None and not isinstance(key_name, str):
        raise AIStudioError("שם מפתח ה-API אינו תקין")
    corrections = []
    used_keys = []
    used_models = []
    batches = []
    warning = None
    started = time.perf_counter()
    for item in lines:
        if (not isinstance(item, dict)
                or not isinstance(item.get("text"), str)
                or not isinstance(item.get("img"), str)):
            raise AIStudioError("נתוני שורה חסרים או אינם תקינים")
        try:
            base64.b64decode(item["img"], validate=True)
        except (ValueError, TypeError) as exc:
            raise AIStudioError("תמונת שורה אינה תקינה") from exc
    for offset in range(0, len(lines), batch_size):
        batch = lines[offset:offset + batch_size]
        try:
            used_key, model, parsed, usage = _correct_batch(
                batch, generation, key_id=key_id, key_name=key_name)
        except AIStudioError as exc:
            if not batches:
                raise
            warning = (
                f"עיבוד חלקי: הושלמו {offset} מתוך {len(lines)} שורות. "
                f"הבקשה הבאה נכשלה: {exc}")
            break
        if used_key not in used_keys:
            used_keys.append(used_key)
        if not any(used["key"] == used_key and used["model"] == model
                   for used in used_models):
            used_models.append({"key": used_key, "model": model})
        batches.append({
            "line_count": len(batch),
            "model": model,
            "image_bytes": sum(len(base64.b64decode(item["img"]))
                               for item in batch),
            **usage,
        })
        for local_index, corrected in parsed.items():
            original = batch[local_index]["text"]
            if corrected != original:
                corrections.append({"index": offset + local_index,
                                    "text": corrected})
    token_fields = ("request_bytes", "prompt_tokens", "output_tokens",
                    "thoughts_tokens", "total_tokens")
    totals = {
        field: (sum(batch[field] for batch in batches)
                if all(isinstance(batch.get(field), int) for batch in batches)
                else None)
        for field in token_fields
    }
    return {
        "corrections": corrections,
        "keys_used": used_keys,
        "models_used": used_models,
        "metrics": {
            "line_count": sum(batch["line_count"] for batch in batches),
            "requested_line_count": len(lines),
            "complete": warning is None,
            "batch_size": batch_size,
            "generation": {
                "temperature": generation["temperature"],
                "top_p": generation["top_p"],
                "top_k": generation["top_k"],
                "max_output_tokens": generation["max_output_tokens"],
                "request_timeout": generation["request_timeout"],
            },
            "batch_count": len(batches),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "batches": batches,
            **totals,
        },
        "warning": warning,
    }
