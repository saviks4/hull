import ast
import base64
import csv
import html
import io
import json
import multiprocessing
import os
import re
import zipfile
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
from groq import Groq

st.set_page_config(
    page_title="Groq Assistant Pro — Проверка заданий",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
}
.hero-banner {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.14), rgba(59, 130, 246, 0.14));
    border: 1px solid rgba(128, 128, 128, 0.25);
    border-radius: 14px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 1.65rem;
    font-weight: 700;
    margin: 0 0 0.25rem 0;
}
.hero-sub {
    font-size: 0.92rem;
    opacity: 0.85;
    margin: 0;
}
.pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin-top: 0.65rem;
}
.pill {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    border: 1px solid rgba(128, 128, 128, 0.3);
    background: rgba(128, 128, 128, 0.08);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_ZIP_TOTAL_BYTES = 15 * 1024 * 1024
MAX_ZIP_FILES = 25
MAX_IMAGES = 4
MAX_TOTAL_TEXT_CHARS = 60_000
AUTOTEST_TIMEOUT_SEC = 3.0

CHAT_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]
VISION_MODELS = [
    "qwen/qwen3.8-27b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
]
GUARD_MODELS = [
    "meta-llama/llama-prompt-guard-2-86m",
    "meta-llama/llama-prompt-guard-2-22m",
    "openai/gpt-oss-safeguard-20b",
]
STT_MODELS = [
    "whisper-large-v3-turbo",
    "whisper-large-v3",
]
TTS_MODELS = [
    "canopylabs/orpheus-v1-english",
    "canopylabs/orpheus-arabic-saudi",
]

TEXT_EXTS = {
    ".txt", ".md", ".py", ".cs", ".html", ".css", ".js", ".ts", ".jsx", ".tsx",
    ".csv", ".json", ".xml", ".yaml", ".yml", ".toml", ".sql", ".log", ".ini",
    ".cfg", ".java", ".cpp", ".c", ".h",
}
IMAGE_EXTS = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".webm", ".flac"}

SKIP_ARCHIVE_PARTS = {
    "__macosx", ".git", "node_modules", "__pycache__", ".venv", "venv", ".idea", ".vscode",
}

SECRET_PATTERNS = [
    r"gsk_[A-Za-z0-9_\-]{10,}",
    r"sk-[A-Za-z0-9_\-]{16,}",
    r"ghp_[A-Za-z0-9]{20,}",
    r"AIza[0-9A-Za-z_\-]{20,}",
]

ALLOWED_STDLIB_IMPORTS = {
    "math", "statistics", "collections", "itertools", "functools", "re", "json",
    "datetime", "string", "random", "typing", "dataclasses", "decimal",
    "fractions", "heapq", "bisect", "copy", "enum",
}

BLOCKED_IMPORTS = {
    "os", "sys", "subprocess", "shutil", "socket", "requests", "urllib", "http",
    "pathlib", "importlib", "ctypes", "multiprocessing", "threading", "pickle",
    "builtins", "tempfile", "glob",
}

REVIEW_MODES = {
    "Строгий преподаватель": (
        "Ты строгий и объективный проверяющий преподаватель. Выявляй все логические, "
        "синтаксические, архитектурные и оформительские недочёты. Дай чёткий вердикт и исправленный вариант."
    ),
    "Мягкий ментор": (
        "Ты поддерживающий ментор-разработчик. Сначала отметь сильные стороны работы, "
        "затем бережно объясни ошибки с подсказками и покажи улучшенный вариант."
    ),
    "Ревью кода и баг-хантинг": (
        "Ты senior-ревьюер. Фокусируйся на граничных случаях (edge cases), сложности алгоритма, "
        "читаемости, обработке исключений, безопасности и потенциальных багах."
    ),
    "Балльная оценка (0–100)": (
        "Ты экзаменатор. Оцени работу по 100-балльной шкале с Markdown-таблицей: "
        "Корректность (до 40), Качество кода/логики (до 25), Граничные случаи (до 20), "
        "Оформление (до 15). Укажи итоговый балл и причины снижения."
    ),
    "Подготовка к защите": (
        "Ты принимаешь защиту работы. После краткого разбора ошибок сформируй 5–7 вероятных "
        "вопросов преподавателя и дай краткие тезисы для уверенного ответа."
    ),
}

CRITERIA_PRESETS = {
    "Без шаблона": "",
    "Лабораторная Python": (
        "1. Логика вынесена в функции с понятными именами и type hints.\n"
        "2. Есть обработка некорректного ввода и граничных случаев.\n"
        "3. Нет глобальных переменных и голого except:.\n"
        "4. Присутствуют docstrings или краткие комментарии."
    ),
    "Веб-вёрстка (HTML/CSS/JS)": (
        "1. Семантическая разметка (header, main, section, footer).\n"
        "2. Атрибуты alt у изображений и label у полей формы.\n"
        "3. Адаптивность и отсутствие инлайн-стилей.\n"
        "4. Корректная обработка событий в JS без ошибок."
    ),
    "SQL-запросы": (
        "1. Корректные JOIN-ы без дублирования строк.\n"
        "2. Отсутствие деструктивных операций без WHERE.\n"
        "3. Алиасы таблиц и читаемое форматирование.\n"
        "4. Фильтрация WHERE до GROUP BY, агрегаты в HAVING."
    ),
    "Реферат / отчёт по методичке": (
        "1. Наличие цели работы, хода выполнения и выводов.\n"
        "2. Соответствие структуре и объёму методички.\n"
        "3. Логическая связность и отсутствие противоречий.\n"
        "4. Аккуратное оформление таблиц и ссылок."
    ),
}

BUILTIN_SKILLS = {
    "🧪 Автотесты Python": {
        "desc": "Генерирует assert-тесты и запускает их в изолированной песочнице с таймаутом 3 сек.",
        "prompt": "Обязательно учти результаты автотестов и укажи, какие кейсы упали и почему.",
    },
    "📐 Проверка формул и расчётов": {
        "desc": "Пошагово перепроверяет вычисления, единицы измерения и формулы.",
        "prompt": "Выполни пошаговую арифметическую и алгебраическую проверку всех формул, расчётов и размерностей.",
    },
    "🛡️ Аудит безопасности (Guard + Code)": {
        "desc": "Проверяет код на уязвимости, утечки секретов и опционально прогоняет через Prompt Guard.",
        "prompt": "Дополнительно проверь решение на уязвимости (инъекции, утечки ключей, небезопасные вызовы).",
    },
    "🎓 Вопросы к защите": {
        "desc": "Добавляет список вопросов преподавателя и шпаргалку для защиты.",
        "prompt": "В конце добавь блок «Вопросы к защите»: 5 вероятных вопросов преподавателя и краткие ответы.",
    },
}


def _safe_import(name: str, globals_: Any = None, locals_: Any = None, fromlist: Any = (), level: int = 0) -> Any:
    root = name.split(".")[0]
    if root not in ALLOWED_STDLIB_IMPORTS:
        raise ImportError(f"Импорт модуля '{name}' не разрешён в песочнице автотестов.")
    return __import__(name, globals_, locals_, fromlist, level)


SAFE_BUILTINS = {
    "__import__": _safe_import,
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "filter": filter, "float": float, "int": int,
    "len": len, "list": list, "map": map, "max": max, "min": min,
    "pow": pow, "print": print, "range": range, "reversed": reversed,
    "round": round, "set": set, "sorted": sorted, "str": str, "sum": sum,
    "tuple": tuple, "zip": zip, "isinstance": isinstance, "issubclass": issubclass,
    "type": type, "Exception": Exception, "ValueError": ValueError,
    "TypeError": TypeError, "KeyError": KeyError, "IndexError": IndexError,
    "ZeroDivisionError": ZeroDivisionError, "AssertionError": AssertionError,
}


def get_api_key(user_override: str | None = None) -> str | None:
    if user_override and user_override.strip():
        return user_override.strip()
    try:
        if "GROQ_API_KEY" in st.secrets and st.secrets["GROQ_API_KEY"]:
            return str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        pass
    env_key = os.getenv("GROQ_API_KEY")
    return env_key.strip() if env_key else None


def fetch_available_groq_models(api_key: str) -> list[str]:
    client = Groq(api_key=api_key)
    resp = client.models.list()
    ids = sorted(m.id for m in (resp.data or []) if getattr(m, "id", None))
    return ids


def audit_code_or_data(ext: str, content: str, label: str) -> list[str]:
    notes: list[str] = []
    lines = content.splitlines()

    if len(lines) > 1500:
        notes.append(f"В «{label}» много строк ({len(lines)}) — проверьте структуру и дублирование.")

    for pat in SECRET_PATTERNS:
        if re.search(pat, content):
            notes.append(f"В «{label}» обнаружен фрагмент, похожий на API-ключ или токен!")
            break

    if ext == ".py":
        try:
            tree = ast.parse(content, filename=label)
            funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
            if content.strip() and not funcs and not classes and len(lines) > 25:
                notes.append(f"В Python «{label}» нет функций или классов — проверьте декомпозицию.")
            for fn in funcs:
                if hasattr(fn, "end_lineno") and fn.end_lineno and (fn.end_lineno - fn.lineno) > 60:
                    notes.append(f"Функция `{fn.name}` в «{label}» длиннее 60 строк — стоит разбить на подзадачи.")
        except SyntaxError as exc:
            notes.append(
                f"Синтаксическая ошибка в Python «{label}» (строка {exc.lineno}, кол. {exc.offset}): {exc.msg}."
            )
        if re.search(r"except\s*:", content):
            notes.append(f"В «{label}» найден `except:` без типа исключения.")
        if "input(" in content:
            notes.append(f"В «{label}» используется `input()` — для автотестов лучше выносить логику в функции.")

    elif ext == ".json":
        try:
            parsed_json = json.loads(content)
            if isinstance(parsed_json, (dict, list)) and len(parsed_json) == 0:
                notes.append(f"JSON «{label}» пуст.")
        except json.JSONDecodeError as exc:
            notes.append(f"Ошибка JSON в «{label}» (строка {exc.lineno}): {exc.msg}.")

    elif ext == ".csv":
        try:
            dialect = csv.Sniffer().sniff(content[:4096], delimiters=",;\t|")
            rows = list(csv.reader(io.StringIO(content), dialect))
            lengths = {len(r) for r in rows if r}
            if len(lengths) > 1:
                notes.append(f"В CSV «{label}» разное число столбцов ({sorted(lengths)}).")
        except Exception as exc:
            notes.append(f"Не удалось проверить CSV «{label}»: {exc}")

    elif ext == ".html":
        if "<html" not in content.lower() and "<body" not in content.lower():
            notes.append(f"HTML «{label}» выглядит как фрагмент без `<html>`/`<body>`.")
        if "<img" in content.lower() and "alt=" not in content.lower():
            notes.append(f"В HTML «{label}» есть `<img>` без `alt`.")

    elif ext == ".cs":
        if content.count("{") != content.count("}"):
            notes.append(f"В C# «{label}» не совпадает количество фигурных скобок.")

    elif ext == ".sql":
        upper_sql = content.upper()
        if any(k in upper_sql for k in ["DROP TABLE", "TRUNCATE ", "DELETE FROM"]) and "WHERE" not in upper_sql:
            notes.append(f"В SQL «{label}» деструктивная команда без `WHERE`.")

    return notes


def _parse_single_bytes(
    name: str,
    raw: bytes,
    api_key: str | None = None,
    stt_model: str = "whisper-large-v3-turbo",
) -> dict[str, Any]:
    ext = Path(name).suffix.lower()
    result: dict[str, Any] = {
        "name": name,
        "ext": ext,
        "size": len(raw),
        "text": "",
        "image_url": None,
        "error": None,
        "warnings": [],
    }

    if len(raw) == 0:
        result["warnings"].append(f"Файл «{name}» пустой (0 байт).")
        return result

    if len(raw) > MAX_FILE_BYTES:
        result["error"] = f"Файл «{name}» превышает лимит {MAX_FILE_BYTES // (1024 * 1024)} МБ."
        return result

    if ext in TEXT_EXTS:
        decoded = raw.decode("utf-8", errors="replace")
        result["text"] = decoded
        result["warnings"].extend(audit_code_or_data(ext, decoded, name))
        return result

    if ext == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(raw))
            pages = [f"[Стр. {i + 1}]\n{p.extract_text() or ''}" for i, p in enumerate(reader.pages)]
            text = "\n\n".join(pages).strip()
            if not "".join(p.extract_text() or "" for p in reader.pages).strip():
                result["warnings"].append(f"В PDF «{name}» нет текстового слоя (возможно, скан).")
            result["text"] = text
            return result
        except Exception as exc:
            result["error"] = f"Ошибка чтения PDF «{name}»: {exc}"
            return result

    if ext == ".docx":
        try:
            from docx import Document

            doc = Document(io.BytesIO(raw))
            parts: list[str] = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            for idx, table in enumerate(doc.tables, start=1):
                parts.append(f"[Таблица {idx}]")
                for row in table.rows:
                    cells = [c.text.strip() for c in row.cells if c.text.strip()]
                    if cells:
                        parts.append(" | ".join(cells))
            text = "\n".join(parts).strip()
            if not text:
                result["warnings"].append(f"Документ «{name}» пуст.")
            result["text"] = text
            return result
        except Exception as exc:
            result["error"] = f"Ошибка чтения DOCX «{name}»: {exc}"
            return result

    if ext in IMAGE_EXTS:
        b64 = base64.b64encode(raw).decode("ascii")
        result["image_url"] = f"data:{IMAGE_EXTS[ext]};base64,{b64}"
        result["text"] = f"[Изображение: {name}, {len(raw)} байт]"
        return result

    if ext in AUDIO_EXTS:
        if not api_key:
            result["error"] = f"Для расшифровки аудио «{name}» требуется GROQ_API_KEY."
            return result
        try:
            client = Groq(api_key=api_key)
            tr = client.audio.transcriptions.create(
                file=(Path(name).name, raw),
                model=stt_model,
            )
            transcript = getattr(tr, "text", "") or ""
            result["text"] = f"[Расшифровка аудио {name} ({stt_model})]:\n{transcript}"
            if not transcript.strip():
                result["warnings"].append(f"Аудио «{name}» расшифровано как пустой текст.")
            return result
        except Exception as exc:
            result["error"] = f"Ошибка Whisper для «{name}»: {exc}"
            return result

    result["error"] = f"Формат «{ext or 'без расширения'}» для файла «{name}» не поддерживается."
    return result


def parse_uploaded_files(
    uploaded_files: list[Any] | None,
    api_key: str | None = None,
    stt_model: str = "whisper-large-v3-turbo",
) -> list[dict[str, Any]]:
    if not uploaded_files:
        return []
    parsed: list[dict[str, Any]] = []
    for f in uploaded_files:
        name = f.name
        ext = Path(name).suffix.lower()
        raw = f.getvalue()
        if ext == ".zip":
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                    total_uncompressed = 0
                    count = 0
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        parts_lower = {p.lower() for p in Path(info.filename).parts}
                        if parts_lower & SKIP_ARCHIVE_PARTS:
                            continue
                        if count >= MAX_ZIP_FILES:
                            parsed.append(
                                {
                                    "name": f"{name}/{info.filename}",
                                    "ext": "",
                                    "size": 0,
                                    "text": "",
                                    "image_url": None,
                                    "error": None,
                                    "warnings": [f"Архив «{name}»: достигнут лимит {MAX_ZIP_FILES} файлов."],
                                }
                            )
                            break
                        total_uncompressed += info.file_size
                        if total_uncompressed > MAX_ZIP_TOTAL_BYTES:
                            parsed.append(
                                {
                                    "name": name,
                                    "ext": ".zip",
                                    "size": len(raw),
                                    "text": "",
                                    "image_url": None,
                                    "error": f"Распакованный объём архива «{name}» превышает {MAX_ZIP_TOTAL_BYTES // (1024 * 1024)} МБ.",
                                    "warnings": [],
                                }
                            )
                            break
                        inner_bytes = zf.read(info.filename)
                        inner_res = _parse_single_bytes(f"{name}/{info.filename}", inner_bytes, api_key, stt_model)
                        if inner_res["error"] and "не поддерживается" in inner_res["error"]:
                            inner_res["warnings"].append(inner_res["error"])
                            inner_res["error"] = None
                        parsed.append(inner_res)
                        count += 1
            except zipfile.BadZipFile:
                parsed.append(
                    {
                        "name": name,
                        "ext": ".zip",
                        "size": len(raw),
                        "text": "",
                        "image_url": None,
                        "error": f"Архив «{name}» повреждён или не является ZIP.",
                        "warnings": [],
                    }
                )
        else:
            parsed.append(_parse_single_bytes(name, raw, api_key, stt_model))
    return parsed


def extract_python_candidates(task_text: str, parsed_files: list[dict[str, Any]]) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    for pf in parsed_files:
        if pf.get("ext") == ".py" and pf.get("text"):
            candidates.append((pf["name"], pf["text"]))

    fenced = re.findall(
        r"```(?:python|py)\s*\n(.*?)```",
        task_text or "",
        flags=re.DOTALL | re.IGNORECASE,
    )
    for i, block in enumerate(fenced, start=1):
        candidates.append((f"task_snippet_{i}.py", block))
    return candidates


def has_unsafe_ast(tree: ast.AST) -> str | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_mod = alias.name.split(".")[0]
                if root_mod in BLOCKED_IMPORTS or root_mod not in ALLOWED_STDLIB_IMPORTS:
                    return f"Запрещённый импорт модуля `{alias.name}` в автотестах."
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod in BLOCKED_IMPORTS or mod not in ALLOWED_STDLIB_IMPORTS:
                return f"Запрещённый импорт из модуля `{node.module}` в автотестах."
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"eval", "exec", "compile", "__import__", "open"}:
                return f"Запрещённый вызов `{node.func.id}()` в автотестах."
    return None


def _execute_tests_in_memory(source_code: str, source_name: str, tests_code: str) -> dict[str, Any]:
    stdout_buf = io.StringIO()
    env_ns: dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
    try:
        code_tree = ast.parse(source_code, filename=source_name)
        test_tree = ast.parse(tests_code, filename="generated_tests.py")
        with redirect_stdout(stdout_buf):
            exec(compile(code_tree, source_name, "exec"), env_ns)
            assert_nodes = [n for n in ast.walk(test_tree) if isinstance(n, ast.Assert)]
            if not assert_nodes:
                exec(compile(test_tree, "generated_tests.py", "exec"), env_ns)
                return {
                    "ran": True,
                    "summary": f"Тесты для `{source_name}` выполнены успешно.",
                    "results": [{"name": "Тестовый блок", "passed": True, "detail": "PASS"}],
                    "stdout": stdout_buf.getvalue()[:2000],
                }

            helper_nodes = [n for n in test_tree.body if not isinstance(n, ast.Assert)]
            if helper_nodes:
                helper_mod = ast.Module(body=helper_nodes, type_ignores=[])
                ast.fix_missing_locations(helper_mod)
                exec(compile(helper_mod, "test_helpers.py", "exec"), env_ns)

            results: list[dict[str, Any]] = []
            for idx, stmt in enumerate(assert_nodes, start=1):
                single_mod = ast.Module(body=[stmt], type_ignores=[])
                ast.fix_missing_locations(single_mod)
                expr_hint = ast.unparse(stmt.test) if hasattr(ast, "unparse") else f"Тест #{idx}"
                try:
                    exec(compile(single_mod, f"test_{idx}.py", "exec"), env_ns)
                    results.append({"name": f"Тест #{idx}: {expr_hint}", "passed": True, "detail": "PASS"})
                except AssertionError:
                    results.append(
                        {"name": f"Тест #{idx}: {expr_hint}", "passed": False, "detail": "FAIL (AssertionError)"}
                    )
                except Exception as exc:
                    results.append(
                        {
                            "name": f"Тест #{idx}: {expr_hint}",
                            "passed": False,
                            "detail": f"{type(exc).__name__}: {exc}",
                        }
                    )

            passed_cnt = sum(1 for r in results if r["passed"])
            return {
                "ran": True,
                "summary": f"Автотесты для `{source_name}`: пройдено {passed_cnt} из {len(results)}.",
                "results": results,
                "stdout": stdout_buf.getvalue()[:2000],
            }
    except Exception as exc:
        return {
            "ran": True,
            "summary": f"Ошибка выполнения `{source_name}`: {type(exc).__name__}: {exc}",
            "results": [{"name": "Выполнение кода", "passed": False, "detail": f"{type(exc).__name__}: {exc}"}],
            "stdout": stdout_buf.getvalue()[:2000],
        }


def _sandbox_worker(source_code: str, source_name: str, tests_code: str, queue: Any) -> None:
    queue.put(_execute_tests_in_memory(source_code, source_name, tests_code))


def _run_in_subprocess(source_code: str, source_name: str, tests_code: str) -> dict[str, Any]:
    try:
        ctx = multiprocessing.get_context("fork") if hasattr(os, "fork") else multiprocessing.get_context()
        queue = ctx.Queue()
        proc = ctx.Process(target=_sandbox_worker, args=(source_code, source_name, tests_code, queue))
        proc.start()
        proc.join(timeout=AUTOTEST_TIMEOUT_SEC)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1.0)
            return {
                "ran": True,
                "summary": f"Автотесты для `{source_name}` остановлены по таймауту ({AUTOTEST_TIMEOUT_SEC:.0f} сек) — возможен бесконечный цикл.",
                "results": [{"name": "Таймаут песочницы", "passed": False, "detail": "TimeoutError"}],
                "stdout": "",
            }
        if not queue.empty():
            return queue.get()
    except Exception:
        return _execute_tests_in_memory(source_code, source_name, tests_code)

    return {
        "ran": False,
        "summary": f"Не удалось получить результат автотестов для `{source_name}`.",
        "results": [],
        "stdout": "",
    }


def generate_and_run_tests(
    api_key: str,
    model_name: str,
    task_text: str,
    source_name: str,
    source_code: str,
    custom_tests: str = "",
) -> dict[str, Any]:
    try:
        code_tree = ast.parse(source_code, filename=source_name)
    except SyntaxError as exc:
        return {
            "ran": False,
            "summary": f"Автотесты не запущены: синтаксическая ошибка в `{source_name}` (строка {exc.lineno}).",
            "results": [],
            "tests_code": "",
            "stdout": "",
        }

    unsafe_reason = has_unsafe_ast(code_tree)
    if unsafe_reason:
        return {
            "ran": False,
            "summary": f"Автотесты остановлены: {unsafe_reason}",
            "results": [],
            "tests_code": "",
            "stdout": "",
        }

    if custom_tests.strip():
        tests_code = custom_tests.strip()
    else:
        client = Groq(api_key=api_key)
        test_prompt = (
            "Напиши 5–7 коротких независимых тестов на чистом Python с использованием `assert` "
            "для проверки функций/классов из кода ниже (включая граничные случаи: пустой ввод, 0, отрицательные значения).\n"
            "Допустимы только встроенные функции и стандартные модули (math, re, json, datetime, collections, itertools).\n"
            "Верни ТОЛЬКО код внутри блока ```python ... ``` без лишнего текста.\n\n"
            f"ЗАДАНИЕ:\n{task_text or '(условие не указано)'}\n\n"
            f"КОД ({source_name}):\n```python\n{source_code[:8000]}\n```"
        )
        resp = client.chat.completions.create(
            model=model_name,
            temperature=0.1,
            messages=[
                {"role": "system", "content": "Ты пишешь точные unit-тесты на assert для Python."},
                {"role": "user", "content": test_prompt},
            ],
        )
        raw_tests = resp.choices[0].message.content or ""
        m = re.search(r"```(?:python|py)?\s*\n(.*?)```", raw_tests, flags=re.DOTALL | re.IGNORECASE)
        tests_code = (m.group(1) if m else raw_tests).strip()

    try:
        test_tree = ast.parse(tests_code, filename="generated_tests.py")
    except SyntaxError as exc:
        return {
            "ran": False,
            "summary": f"Тесты содержат синтаксическую ошибку: {exc.msg}.",
            "results": [],
            "tests_code": tests_code,
            "stdout": "",
        }

    test_unsafe = has_unsafe_ast(test_tree)
    if test_unsafe:
        return {
            "ran": False,
            "summary": f"Тесты заблокированы: {test_unsafe}",
            "results": [],
            "tests_code": tests_code,
            "stdout": "",
        }

    res = _run_in_subprocess(source_code, source_name, tests_code)
    res["tests_code"] = tests_code
    return res


def run_guard_check(api_key: str, guard_model: str, text: str) -> tuple[bool, str]:
    if not text.strip():
        return True, "Текст пуст."
    try:
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=guard_model,
            temperature=0.0,
            max_tokens=128,
            messages=[{"role": "user", "content": text[:3500]}],
        )
        verdict = (resp.choices[0].message.content or "").strip()
        lower_v = verdict.lower()
        if any(bad in lower_v for bad in ["unsafe", "jailbreak", "injection", "malicious", "1"]):
            if lower_v == "0" or "benign" in lower_v or "safe" in lower_v and "unsafe" not in lower_v:
                return True, f"{guard_model}: {verdict}"
            return False, f"{guard_model} пометил ввод как рискованный: {verdict}"
        return True, f"{guard_model}: OK ({verdict or 'safe'})"
    except Exception as exc:
        return True, f"Guard-проверка пропущена ({exc})"


def detect_profile(task_text: str, files_text: str, images_count: int) -> str:
    combined = f"{task_text}\n{files_text[:4000]}".lower()
    if images_count > 0:
        return "визуальное задание / скриншот / схема"
    if any(k in combined for k in ["def ", "class ", "import ", "select ", "html", "css", "код", "функция", "баг"]):
        return "программирование / код / разметка / запросы"
    if any(k in combined for k in ["формул", "уравнен", "интеграл", "производн", "вероятност", "расчёт"]):
        return "математика / физика / расчёты"
    if any(k in combined for k in ["таблиц", "csv", "столбц", "метрик", "анализ"]):
        return "данные / таблицы / аналитика"
    return "текстовое / учебное / аналитическое задание"


def validate_submission(
    api_key: str | None,
    task_text: str,
    criteria_text: str,
    uploaded_files: list[Any] | None,
    stt_model: str = "whisper-large-v3-turbo",
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    checks: list[str] = []

    clean_task = (task_text or "").strip()
    clean_criteria = (criteria_text or "").strip()

    if not api_key:
        errors.append("Не найден GROQ_API_KEY (укажите в боковой панели, Secrets или окружении).")
    else:
        checks.append("API-ключ OK")
        if not api_key.startswith("gsk_"):
            warnings.append("Ключ не начинается с 'gsk_' — проверьте корректность GROQ_API_KEY.")

    parsed_files = parse_uploaded_files(uploaded_files, api_key=api_key, stt_model=stt_model)
    if not clean_task and not parsed_files:
        errors.append("Введите текст задания или прикрепите файлы.")
    elif clean_task:
        checks.append(f"Задание: {len(clean_task)} симв.")
        for idx, snip in enumerate(
            re.findall(r"```(?:python|py)\s*\n(.*?)```", clean_task, flags=re.DOTALL | re.IGNORECASE),
            start=1,
        ):
            warnings.extend(audit_code_or_data(".py", snip, f"фрагмент кода #{idx}"))
        for pat in SECRET_PATTERNS:
            if re.search(pat, clean_task):
                warnings.append("В тексте задания обнаружен фрагмент, похожий на API-ключ!")
                break

    if clean_criteria:
        checks.append(f"Критерии: {len(clean_criteria)} симв.")
    else:
        warnings.append("Методичка/критерии не заданы — проверка пройдёт по общим стандартам.")

    text_blocks: list[str] = []
    images: list[dict[str, str]] = []
    total_chars = 0
    valid_count = 0

    for pf in parsed_files:
        if pf["error"]:
            errors.append(pf["error"])
            continue
        warnings.extend(pf["warnings"])
        valid_count += 1

        if pf["image_url"]:
            if len(images) < MAX_IMAGES:
                images.append({"name": pf["name"], "url": pf["image_url"]})
            else:
                warnings.append(f"Картинка «{pf['name']}» пропущена (лимит {MAX_IMAGES}).")

        if pf["text"]:
            rem = MAX_TOTAL_TEXT_CHARS - total_chars
            if rem <= 0:
                warnings.append(f"Достигнут лимит {MAX_TOTAL_TEXT_CHARS} симв. — файл «{pf['name']}» пропущен.")
                continue
            clipped = pf["text"][:rem]
            if len(pf["text"]) > rem:
                warnings.append(f"Файл «{pf['name']}» обрезан до {rem} симв. из-за лимита контекста.")
            total_chars += len(clipped)
            text_blocks.append(f"=== ФАЙЛ: {pf['name']} ({pf['size']} Б) ===\n{clipped}")

    if valid_count > 0:
        checks.append(f"Файлов: {valid_count} (картинок: {len(images)})")

    files_text = "\n\n".join(text_blocks)
    profile = detect_profile(clean_task, files_text, len(images))
    py_candidates = extract_python_candidates(clean_task, parsed_files)
    if py_candidates:
        checks.append(f"Python-модулей: {len(py_candidates)}")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "profile": profile,
        "task_text": clean_task,
        "criteria_text": clean_criteria,
        "files_text": files_text,
        "total_chars": total_chars,
        "images": images,
        "py_candidates": py_candidates,
        "parsed_files": parsed_files,
    }


def sanitize_output(answer: str) -> tuple[str, list[str]]:
    notes: list[str] = []
    text = (answer or "").strip()
    if not text:
        return "", ["Модель вернула пустой ответ."]

    for pat in SECRET_PATTERNS:
        rx = re.compile(pat)
        if rx.search(text):
            text = rx.sub("[СКРЫТЫЙ_СЕКРЕТ]", text)
            notes.append("Обнаружен фрагмент, похожий на ключ/токен — он скрыт в выводе.")

    return text, notes


def run_evaluation(
    api_key: str,
    model_name: str,
    vision_model_name: str,
    temperature: float,
    mode_name: str,
    language_pref: str,
    active_skills_prompts: list[str],
    profile: str,
    task_text: str,
    criteria_text: str,
    files_text: str,
    images: list[dict[str, str]],
    autotest_reports: list[dict[str, Any]],
) -> str:
    client = Groq(api_key=api_key)
    mode_instruction = REVIEW_MODES.get(mode_name, REVIEW_MODES["Строгий преподаватель"])
    lang_line = (
        "Отвечай на языке задания (по умолчанию на русском)."
        if language_pref == "Авто"
        else f"Отвечай строго на языке: {language_pref}."
    )
    skills_block = "\n".join(f"- {s}" for s in active_skills_prompts if s.strip())

    system_prompt = (
        f"{mode_instruction}\n{lang_line}\n"
        "Обязательная структура ответа:\n"
        "1. **Вердикт и оценка:** статус (зачтено / есть замечания / требуется доработка) и краткий итог.\n"
        "2. **Ошибки и слабые места:** конкретные недочёты по логике, коду, расчётам или содержанию.\n"
        "3. **Проверка оформления по методичке/критериям:** что выполнено, а что нарушено или отсутствует.\n"
        "4. **Итоги автотестов:** учти результаты запуска тестов (если они выполнялись).\n"
        "5. **Готовое решение / исправленный вариант:** полный рабочий вариант.\n"
        "6. **Уточнения (если нужны):** только критичные вопросы."
    )
    if skills_block:
        system_prompt += f"\n\nАКТИВНЫЕ СКИЛЛЫ (выполнить обязательно):\n{skills_block}"

    parts = [
        f"ПРОФИЛЬ: {profile}",
        f"РЕЖИМ ПРОВЕРКИ: {mode_name}",
        f"ЗАДАНИЕ:\n{task_text or '(текст не введён, см. файлы)'}",
        f"МЕТОДИЧКА / КРИТЕРИИ ОФОРМЛЕНИЯ:\n{criteria_text or '(не заданы — проверь по общим требованиям к аккуратности и полноте)'}",
    ]
    if files_text:
        parts.append(f"ФАЙЛЫ:\n{files_text}")
    if autotest_reports:
        test_lines: list[str] = []
        for rep in autotest_reports:
            if rep.get("summary"):
                test_lines.append(rep["summary"])
                for r in rep.get("results", []):
                    test_lines.append(f"- {r['name']}: {r['detail']}")
                if rep.get("stdout"):
                    test_lines.append(f"Stdout:\n{rep['stdout'][:600]}")
        if test_lines:
            parts.append("РЕЗУЛЬТАТЫ АВТОТЕСТОВ:\n" + "\n".join(test_lines))

    payload = "\n\n".join(parts)

    if images:
        user_content: Any = [{"type": "text", "text": payload}]
        for img in images:
            user_content.append({"type": "image_url", "image_url": {"url": img["url"]}})
        chosen_model = vision_model_name
    else:
        user_content = payload
        chosen_model = model_name

    response = client.chat.completions.create(
        model=chosen_model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    return response.choices[0].message.content or ""


def synthesize_tts(api_key: str, tts_model: str, text: str) -> tuple[bytes | None, str | None]:
    clean = re.sub(r"[#*`>\[\]()_]", " ", text or "").strip()
    snippet = clean[:900]
    if not snippet:
        return None, "Нет текста для озвучки."
    try:
        client = Groq(api_key=api_key)
        voice = "tara" if "english" in tts_model else "fahad"
        resp = client.audio.speech.create(
            model=tts_model,
            voice=voice,
            input=snippet,
            response_format="wav",
        )
        audio_bytes = resp.read() if hasattr(resp, "read") else bytes(resp)
        return audio_bytes, None
    except Exception as exc:
        return None, f"Озвучка недоступна для {tts_model}: {exc}"


def run_followup_chat(
    api_key: str,
    model_name: str,
    active_report: dict[str, Any],
    chat_history: list[dict[str, str]],
    user_question: str,
) -> str:
    client = Groq(api_key=api_key)
    val = active_report["val"]
    context_summary = (
        f"Ты помогаешь пользователю разобраться с проверкой его задания.\n"
        f"ЗАДАНИЕ:\n{val.get('task_text', '')[:6000]}\n\n"
        f"ФАЙЛЫ (фрагмент):\n{val.get('files_text', '')[:8000]}\n\n"
        f"ИТОГ ПРОВЕРКИ:\n{active_report.get('answer', '')[:8000]}"
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": context_summary}]
    for msg in chat_history[-10:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_question})

    resp = client.chat.completions.create(
        model=model_name,
        temperature=0.25,
        messages=messages,
    )
    return resp.choices[0].message.content or ""


def build_report_markdown(
    title: str,
    mode_name: str,
    val: dict[str, Any],
    autotest_reports: list[dict[str, Any]],
    answer: str,
) -> str:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# Отчёт проверки: {title}",
        f"- Дата: {now_str}",
        f"- Режим: {mode_name}",
        f"- Профиль: {val.get('profile', '-')}",
        "",
        "## Валидатор",
    ]
    for c in val.get("checks", []):
        lines.append(f"- OK: {c}")
    for w in val.get("warnings", []):
        lines.append(f"- Предупреждение: {w}")
    for e in val.get("errors", []):
        lines.append(f"- Ошибка: {e}")

    if autotest_reports:
        lines.extend(["", "## Автопроверка тестов"])
        for rep in autotest_reports:
            if rep.get("summary"):
                lines.append(f"### {rep['summary']}")
                for r in rep.get("results", []):
                    lines.append(f"- [{'PASS' if r['passed'] else 'FAIL'}] {r['name']} — {r['detail']}")

    lines.extend(["", "## Разбор и решение", answer])
    return "\n".join(lines)


def build_report_html(title: str, mode_name: str, report_md: str) -> str:
    escaped_title = html.escape(title)
    escaped_body = html.escape(report_md)
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Отчёт — {escaped_title}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #f6f8fa; color: #1f2328; margin: 0; padding: 32px 16px; line-height: 1.6;
    }}
    .card {{
      max-width: 880px; margin: 0 auto; background: #fff; border: 1px solid #d0d7de;
      border-radius: 12px; padding: 32px; box-shadow: 0 4px 18px rgba(0,0,0,0.05);
    }}
    .meta {{ color: #57606a; font-size: 14px; margin-bottom: 20px; }}
    pre {{ white-space: pre-wrap; word-break: break-word; font-family: inherit; margin: 0; }}
    @media print {{
      body {{ background: #fff; padding: 0; }}
      .card {{ border: none; box-shadow: none; padding: 0; }}
    }}
  </style>
</head>
<body>
  <div class="card">
    <div class="meta">Режим: {html.escape(mode_name)} · Сохраните в PDF через Ctrl+P / Cmd+P</div>
    <pre>{escaped_body}</pre>
  </div>
</body>
</html>"""


# --- Инициализация session_state ---
if "last_report" not in st.session_state:
    st.session_state.last_report = None
if "history" not in st.session_state:
    st.session_state.history = []
if "followup_chat" not in st.session_state:
    st.session_state.followup_chat = []
if "criteria_text_state" not in st.session_state:
    st.session_state.criteria_text_state = ""
if "task_text_state" not in st.session_state:
    st.session_state.task_text_state = ""
if "custom_skills" not in st.session_state:
    st.session_state.custom_skills = []
if "live_models" not in st.session_state:
    st.session_state.live_models = []
if "pre_validation" not in st.session_state:
    st.session_state.pre_validation = None

# --- Боковая панель ---
with st.sidebar:
    st.header("⚙️ Центр управления")
    user_api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_... (или из Secrets)",
    )
    active_key = get_api_key(user_api_key)
    if active_key:
        st.caption("Статус ключа: подключён ✅")
    else:
        st.warning("Добавьте GROQ_API_KEY выше или в Secrets.")

    review_mode = st.selectbox("Режим проверки", list(REVIEW_MODES.keys()), index=0)

    chat_choices = list(dict.fromkeys(CHAT_MODELS + [m for m in st.session_state.live_models if "whisper" not in m and "orpheus" not in m and "guard" not in m]))
    selected_model = st.selectbox("ИИ-модель (текст и код)", chat_choices, index=0)

    if st.button("🔄 Обновить модели с Groq API", use_container_width=True):
        if not active_key:
            st.error("Сначала укажите API-ключ.")
        else:
            try:
                st.session_state.live_models = fetch_available_groq_models(active_key)
                st.success(f"Загружено моделей: {len(st.session_state.live_models)}")
            except Exception as exc:
                st.error(f"Не удалось получить список: {exc}")

    with st.expander("🎛️ Модели и параметры (Vision / Audio / Guard)"):
        selected_vision = st.selectbox("Vision-модель (картинки)", VISION_MODELS, index=0)
        selected_stt = st.selectbox("Whisper STT (аудио)", STT_MODELS, index=0)
        selected_guard = st.selectbox("Guard-модель (валидатор)", GUARD_MODELS, index=0)
        enable_guard = st.checkbox("Проверять ввод через Guard-модель", value=False)
        selected_tts = st.selectbox("Orpheus TTS (озвучка)", TTS_MODELS, index=0)
        enable_tts = st.checkbox("Озвучивать итог проверки (TTS)", value=False)
        language_pref = st.selectbox("Язык ответа", ["Авто", "Русский", "Українська", "English"], index=0)
        temperature = st.number_input("Креативность (temperature)", min_value=0.0, max_value=0.7, value=0.2, step=0.05)

    st.divider()
    st.subheader("🧩 Скиллы ассистента")
    selected_skills = st.multiselect(
        "Активные навыки",
        list(BUILTIN_SKILLS.keys()),
        default=["🧪 Автотесты Python"],
    )
    with st.expander("➕ Добавить свой скилл"):
        new_skill_name = st.text_input("Название скилла", placeholder="Например: Проверка PEP8")
        new_skill_prompt = st.text_area("Инструкция скилла", height=70, placeholder="Что ассистент должен делать...")
        if st.button("Сохранить скилл", use_container_width=True):
            if new_skill_name.strip() and new_skill_prompt.strip():
                st.session_state.custom_skills.append(
                    {"name": new_skill_name.strip(), "prompt": new_skill_prompt.strip()}
                )
                st.success(f"Скилл «{new_skill_name.strip()}» добавлен!")
            else:
                st.warning("Заполните название и инструкцию.")

    if st.session_state.custom_skills:
        active_custom = st.multiselect(
            "Мои скиллы",
            [s["name"] for s in st.session_state.custom_skills],
            default=[s["name"] for s in st.session_state.custom_skills],
        )
    else:
        active_custom = []

    st.divider()
    preset_choice = st.selectbox("Шаблон критериев", list(CRITERIA_PRESETS.keys()), index=0)
    c_p1, c_p2 = st.columns(2)
    if c_p1.button("Применить", use_container_width=True):
        st.session_state.criteria_text_state = CRITERIA_PRESETS[preset_choice]
    if c_p2.button("Пример задачи", use_container_width=True):
        st.session_state.task_text_state = (
            "Напиши функцию `is_palindrome(s: str) -> bool`, которая игнорирует пробелы и регистр.\n"
            "```python\ndef is_palindrome(s):\n    return s == s[::-1]\n```"
        )
        st.session_state.criteria_text_state = CRITERIA_PRESETS["Лабораторная Python"]

    if st.session_state.history:
        st.divider()
        st.subheader(f"🕘 История ({len(st.session_state.history)})")
        for idx, item in enumerate(st.session_state.history):
            if st.button(f"{idx + 1}. {item['title'][:28]}", key=f"hist_{idx}", use_container_width=True):
                st.session_state.last_report = item["full_report"]
                st.session_state.followup_chat = []
        if st.button("Очистить историю", use_container_width=True):
            st.session_state.history = []

# --- Главная шапка ---
st.markdown(
    f"""
<div class="hero-banner">
  <div class="hero-title">✅ Groq Assistant Pro — Проверка заданий</div>
  <p class="hero-sub">Интеллектуальная проверка кода, лабораторных, документов, аудио и скриншотов с валидатором и песочницей.</p>
  <div class="pill-row">
    <span class="pill">🤖 {html.escape(selected_model)}</span>
    <span class="pill">🎯 {html.escape(review_mode)}</span>
    <span class="pill">🧩 Скиллов: {len(selected_skills) + len(active_custom)}</span>
    <span class="pill">🔑 {"Ключ активен" if active_key else "Нужен API-ключ"}</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# --- Форма ввода ---
with st.form("main_check_form", clear_on_submit=False):
    col_left, col_right = st.columns([3, 2])
    with col_left:
        task_input = st.text_area(
            "Задание или решение на проверку",
            value=st.session_state.task_text_state,
            height=210,
            placeholder="Вставьте условие задания, код, решение или вопрос...",
        )
    with col_right:
        criteria_input = st.text_area(
            "Методичка / требования к оформлению",
            value=st.session_state.criteria_text_state,
            height=210,
            placeholder="Выберите шаблон слева или вставьте требования преподавателя...",
        )

    uploaded_files = st.file_uploader(
        "Файлы, ZIP-архив или аудио (код, PDF, DOCX, CSV, JSON, SQL, скриншоты, MP3/WAV)",
        accept_multiple_files=True,
        type=[
            "txt", "md", "py", "cs", "html", "css", "js", "ts", "jsx", "tsx",
            "csv", "json", "xml", "yaml", "yml", "toml", "sql", "java", "cpp", "c", "h",
            "pdf", "docx", "zip", "png", "jpg", "jpeg", "webp", "mp3", "wav", "m4a", "ogg",
        ],
    )

    c_opt1, c_opt2 = st.columns([2, 1])
    with c_opt1:
        custom_tests_input = st.text_input(
            "Свои assert-тесты (необязательно)",
            placeholder="Например: assert is_palindrome('А роза упала на лапу Азора') is True",
        )
    with c_opt2:
        st.write("")
        st.write("")
        run_autotests = "🧪 Автотесты Python" in selected_skills

    btn_col1, btn_col2 = st.columns([1, 2])
    with btn_col1:
        validate_only = st.form_submit_button("🛡️ Только валидатор", use_container_width=True)
    with btn_col2:
        submitted = st.form_submit_button("🚀 Проверить задание", type="primary", use_container_width=True)

if validate_only:
    st.session_state.task_text_state = task_input
    st.session_state.criteria_text_state = criteria_input
    pre_val = validate_submission(
        api_key=active_key,
        task_text=task_input,
        criteria_text=criteria_input,
        uploaded_files=uploaded_files,
        stt_model=selected_stt,
    )
    if enable_guard and active_key and pre_val["task_text"]:
        g_ok, g_msg = run_guard_check(active_key, selected_guard, pre_val["task_text"])
        if g_ok:
            pre_val["checks"].append(g_msg)
        else:
            pre_val["warnings"].append(g_msg)
    st.session_state.pre_validation = pre_val

if st.session_state.pre_validation and not submitted:
    pv = st.session_state.pre_validation
    with st.container(border=True):
        st.subheader("🛡️ Результат предварительного валидатора")
        m1, m2, m3 = st.columns(3)
        m1.metric("Статус", "Готово ✅" if pv["ok"] else "Есть ошибки ❌")
        m2.metric("Предупреждения", len(pv["warnings"]))
        m3.metric("Объём текста", f"{pv['total_chars']:,} симв.")
        if pv["checks"]:
            st.success(" | ".join(pv["checks"]))
        for err in pv["errors"]:
            st.error(err)
        for warn in pv["warnings"]:
            st.warning(warn)

if submitted:
    st.session_state.pre_validation = None
    st.session_state.task_text_state = task_input
    st.session_state.criteria_text_state = criteria_input

    val = validate_submission(
        api_key=active_key,
        task_text=task_input,
        criteria_text=criteria_input,
        uploaded_files=uploaded_files,
        stt_model=selected_stt,
    )

    if enable_guard and active_key and val["task_text"]:
        g_ok, g_msg = run_guard_check(active_key, selected_guard, val["task_text"])
        if g_ok:
            val["checks"].append(g_msg)
        else:
            val["warnings"].append(g_msg)

    autotest_reports: list[dict[str, Any]] = []
    answer = ""
    post_notes: list[str] = []
    tts_audio: bytes | None = None

    active_skill_prompts = [BUILTIN_SKILLS[k]["prompt"] for k in selected_skills if k in BUILTIN_SKILLS]
    active_skill_prompts.extend(
        s["prompt"] for s in st.session_state.custom_skills if s["name"] in active_custom
    )

    if val["ok"] and active_key:
        with st.status("Выполняю проверку...", expanded=False) as status:
            if run_autotests and val["py_candidates"]:
                st.write(f"Запуск автотестов ({len(val['py_candidates'])} мод.)...")
                for cand_name, cand_code in val["py_candidates"][:5]:
                    rep = generate_and_run_tests(
                        api_key=active_key,
                        model_name=selected_model,
                        task_text=val["task_text"],
                        source_name=cand_name,
                        source_code=cand_code,
                        custom_tests=custom_tests_input,
                    )
                    autotest_reports.append(rep)
            elif run_autotests:
                autotest_reports.append(
                    {"ran": False, "summary": "Python-код не найден — автотесты не запускались.", "results": []}
                )

            st.write(f"Анализ через {selected_vision if val['images'] else selected_model}...")
            try:
                raw_answer = run_evaluation(
                    api_key=active_key,
                    model_name=selected_model,
                    vision_model_name=selected_vision,
                    temperature=float(temperature),
                    mode_name=review_mode,
                    language_pref=language_pref,
                    active_skills_prompts=active_skill_prompts,
                    profile=val["profile"],
                    task_text=val["task_text"],
                    criteria_text=val["criteria_text"],
                    files_text=val["files_text"],
                    images=val["images"],
                    autotest_reports=autotest_reports,
                )
                answer, post_notes = sanitize_output(raw_answer)
                if enable_tts and answer:
                    st.write(f"Озвучка ({selected_tts})...")
                    tts_audio, tts_err = synthesize_tts(active_key, selected_tts, answer)
                    if tts_err:
                        post_notes.append(tts_err)
                status.update(label="Проверка завершена ✅", state="complete")
            except Exception as exc:
                val["errors"].append(f"Ошибка Groq API: {exc}")
                val["ok"] = False
                status.update(label="Ошибка при проверке ❌", state="error")

    preview_title = (
        val["task_text"][:60]
        if val["task_text"]
        else (", ".join(f.name for f in (uploaded_files or []))[:60] or "Задание")
    )
    report_md = build_report_markdown(preview_title, review_mode, val, autotest_reports, answer) if answer else ""
    report_html = build_report_html(preview_title, review_mode, report_md) if answer else ""

    report_obj = {
        "title": preview_title,
        "mode": review_mode,
        "val": val,
        "autotests": autotest_reports,
        "notes": post_notes,
        "answer": answer,
        "tts_audio": tts_audio,
        "report_md": report_md,
        "report_html": report_html,
    }
    st.session_state.last_report = report_obj
    st.session_state.followup_chat = []

    if answer:
        st.session_state.history.insert(0, {"title": preview_title, "full_report": report_obj})
        st.session_state.history = st.session_state.history[:12]

active = st.session_state.last_report
if active:
    val = active["val"]
    tab_review, tab_tests, tab_files, tab_export = st.tabs(
        ["📋 Разбор и решение", "🧪 Автотесты", "📁 Файлы и валидатор", "⬇️ Экспорт"]
    )

    with tab_review:
        if val["ok"]:
            st.success("Валидация пройдена: " + " | ".join(val["checks"]))
        else:
            for err in val["errors"]:
                st.error(err)
        for warn in val["warnings"]:
            st.warning(warn)
        for note in active.get("notes", []):
            st.warning(note)

        if active.get("tts_audio"):
            st.audio(active["tts_audio"], format="audio/wav")

        if active.get("answer"):
            st.markdown(active["answer"])

            st.divider()
            st.subheader("💬 Уточнить или доработать с ассистентом")
            quick_cols = st.columns(4)
            quick_prompts = [
                ("Объясни проще", "Объясни найденные ошибки максимально простыми словами и на коротком примере."),
                ("Вопросы к защите", "Задай 5 каверзных вопросов, которые преподаватель может задать на защите, и дай краткие ответы."),
                ("Тесты pytest", "Напиши полноценный файл test_solution.py с тестами для pytest."),
                ("Оптимизируй код", "Покажи более быстрый и чистый вариант кода и объясни, что улучшилось."),
            ]
            queued_prompt = None
            for idx, (label, prompt_txt) in enumerate(quick_prompts):
                if quick_cols[idx].button(label, key=f"qp_{idx}", use_container_width=True):
                    queued_prompt = prompt_txt

            for msg in st.session_state.followup_chat:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            chat_input = st.chat_input("Задайте уточняющий вопрос по проверке...")
            final_q = queued_prompt or chat_input
            if final_q:
                if not active_key:
                    st.error("Для диалога укажите Groq API Key в боковой панели.")
                else:
                    st.session_state.followup_chat.append({"role": "user", "content": final_q})
                    with st.spinner("Готовлю ответ..."):
                        reply = run_followup_chat(
                            api_key=active_key,
                            model_name=selected_model,
                            active_report=active,
                            chat_history=st.session_state.followup_chat[:-1],
                            user_question=final_q,
                        )
                        clean_reply, _ = sanitize_output(reply)
                    st.session_state.followup_chat.append({"role": "assistant", "content": clean_reply})
                    st.rerun()

    with tab_tests:
        autos = active.get("autotests", [])
        if not autos:
            st.info("Автотесты не запускались.")
        for t_idx, auto in enumerate(autos, start=1):
            if auto.get("summary"):
                st.info(auto["summary"])
            for r in auto.get("results", []):
                icon = "✅" if r["passed"] else "❌"
                st.write(f"{icon} **{r['name']}** — `{r['detail']}`")
            if auto.get("stdout"):
                with st.expander("Вывод print() (stdout)"):
                    st.code(auto["stdout"])
            if auto.get("tests_code"):
                with st.expander(f"Код сгенерированных тестов #{t_idx}"):
                    st.code(auto["tests_code"], language="python")

    with tab_files:
        used = val.get("total_chars", 0)
        ratio = min(used / MAX_TOTAL_TEXT_CHARS, 1.0)
        st.write(f"**Заполненность текстового контекста:** {used:,} / {MAX_TOTAL_TEXT_CHARS:,} симв.")
        st.progress(ratio)
        for pf in val.get("parsed_files", []):
            with st.expander(f"{pf['name']} ({pf['size']} Б)"):
                if pf.get("error"):
                    st.error(pf["error"])
                for w in pf.get("warnings", []):
                    st.warning(w)
                if pf.get("image_url"):
                    st.image(pf["image_url"], caption=pf["name"])
                elif pf.get("text"):
                    st.code(pf["text"][:4000])

    with tab_export:
        if active.get("answer"):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.download_button(
                    "Скачать .md",
                    data=active["report_md"].encode("utf-8"),
                    file_name="report_check.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with c2:
                st.download_button(
                    "Скачать .html (для PDF)",
                    data=active["report_html"].encode("utf-8"),
                    file_name="report_check.html",
                    mime="text/html",
                    use_container_width=True,
                )
            with c3:
                st.download_button(
                    "Скачать .txt",
                    data=active["report_md"].encode("utf-8"),
                    file_name="report_check.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with c4:
                json_payload = json.dumps(
                    {
                        "title": active["title"],
                        "mode": active.get("mode"),
                        "profile": val.get("profile"),
                        "checks": val.get("checks"),
                        "warnings": val.get("warnings"),
                        "autotests": active.get("autotests"),
                        "answer": active.get("answer"),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                st.download_button(
                    "Скачать .json",
                    data=json_payload.encode("utf-8"),
                    file_name="report_check.json",
                    mime="application/json",
                    use_container_width=True,
                )
        else:
            st.info("Отчёт пока не сформирован.")
