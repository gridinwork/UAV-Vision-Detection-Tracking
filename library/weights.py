"""Check catalog links and store detector files in a local SQLite library.

Only https links from ``model/catalog.json`` are requested. An HTML page is
rejected. A checkpoint is marked usable for UAV mode only when its own class
names contain a UAV class. COCO checkpoints are stored, if a link is ever
added, but they are not selected as the UAV detector.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from detection.class_catalog import allowed_indices, class_names, classify_weights
from utils.paths import MODEL_DIR


CATALOG_PATH = MODEL_DIR / "catalog.json"
LIBRARY_DIR = MODEL_DIR / "library"
DB_PATH = LIBRARY_DIR / "weights.sqlite"
FILES_DIR = LIBRARY_DIR / "files"
MAX_BYTES = 2 * 1024 * 1024 * 1024
_ID = re.compile(r"[A-Za-z0-9._-]{1,80}")


def download_all(
    catalog_path: Path | None = None,
    db_path: Path | None = None,
    files_dir: Path | None = None,
    opener=None,
    read_names=None,
) -> list[dict]:
    """Check every catalog link and download the files that pass."""
    catalog = load_catalog(catalog_path or CATALOG_PATH)
    database = db_path or DB_PATH
    folder = files_dir or FILES_DIR
    folder.mkdir(parents=True, exist_ok=True)
    database.parent.mkdir(parents=True, exist_ok=True)
    names_reader = _default_names if read_names is None else read_names
    connection = _connect(database)
    try:
        with connection:
            for entry in catalog:
                record = _fetch_entry(entry, folder, opener, names_reader)
                _save(connection, record)
            order = [str(item.get("id") or "") for item in catalog]
            rows = {row["id"]: row for row in list_records(connection)}
    finally:
        connection.close()
    return [rows[item] for item in order if item in rows]


def load_catalog(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("catalog must be a list")
    return data


def list_records(connection: sqlite3.Connection) -> list[dict]:
    cursor = connection.execute(
        "SELECT id, title, role, url, filename, local_path, sha256, size_bytes, content_type, "
        "http_status, class_names, kept_classes, kind, warning, license, status, detail, usable, checked_at "
        "FROM weights ORDER BY id"
    )
    columns = [item[0] for item in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS weights (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT '',
            role TEXT NOT NULL DEFAULT '',
            url TEXT NOT NULL DEFAULT '',
            filename TEXT NOT NULL DEFAULT '',
            local_path TEXT NOT NULL DEFAULT '',
            sha256 TEXT NOT NULL DEFAULT '',
            size_bytes INTEGER NOT NULL DEFAULT 0,
            content_type TEXT NOT NULL DEFAULT '',
            http_status INTEGER NOT NULL DEFAULT 0,
            class_names TEXT NOT NULL DEFAULT '',
            kept_classes TEXT NOT NULL DEFAULT '',
            kind TEXT NOT NULL DEFAULT '',
            warning TEXT NOT NULL DEFAULT '',
            license TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT '',
            detail TEXT NOT NULL DEFAULT '',
            usable INTEGER NOT NULL DEFAULT 0,
            checked_at TEXT NOT NULL DEFAULT ''
        )
        """
    )
    return connection


def _fetch_entry(entry: dict, folder: Path, opener, read_names) -> dict:
    record = _blank(entry)
    entry_id = str(entry.get("id") or "")
    filename = Path(str(entry.get("filename") or "")).name
    url = str(entry.get("url") or "").strip()
    if not _ID.fullmatch(entry_id) or not filename or filename in {".", ".."}:
        record["status"] = "failed"
        record["detail"] = "bad catalog entry"
        return record
    record["id"] = entry_id
    record["filename"] = filename
    destination = folder / entry_id / filename
    if not url:
        record["status"] = "not_public"
        record["detail"] = str(entry.get("note") or "NOT_PUBLIC")
        return record
    if not url.lower().startswith("https://"):
        record["status"] = "failed"
        record["detail"] = "only https links are allowed"
        return record
    try:
        status, content_type = _probe(url, opener)
    except Exception as exc:
        record["status"] = "failed"
        record["detail"] = str(exc)
        return record
    record["http_status"] = status
    record["content_type"] = content_type
    expected = str(entry.get("sha256") or "").strip().lower()
    if destination.is_file() and _matches(destination, expected):
        record["status"] = "already"
        record["local_path"] = str(destination)
        record["sha256"] = _sha256(destination)
        record["size_bytes"] = destination.stat().st_size
        _describe(record, destination, read_names)
        return record
    try:
        payload, status, content_type = _download(url, opener)
    except Exception as exc:
        record["status"] = "failed"
        record["detail"] = str(exc)
        return record
    digest = hashlib.sha256(payload).hexdigest()
    if expected and digest != expected:
        record["status"] = "failed"
        record["detail"] = "sha256 does not match the catalog"
        record["sha256"] = digest
        return record
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    record["status"] = "downloaded"
    record["local_path"] = str(destination)
    record["sha256"] = digest
    record["size_bytes"] = len(payload)
    record["http_status"] = status
    record["content_type"] = content_type
    _describe(record, destination, read_names)
    return record


def _describe(record: dict, path: Path, read_names) -> None:
    names: list[str] = []
    if path.suffix.lower() in {".pt", ".onnx"} and read_names is not None:
        try:
            names = [str(item) for item in read_names(path)]
        except Exception as exc:
            record["detail"] = f"file saved, class names were not read: {exc}"
            record["usable"] = 0
            return
    kind, warning = classify_weights(names, path.name)
    kept = sorted(set(allowed_indices({index: name for index, name in enumerate(names)}).values()))
    record["class_names"] = json.dumps(names, ensure_ascii=False)
    record["kept_classes"] = json.dumps(kept, ensure_ascii=False)
    record["kind"] = kind
    record["warning"] = warning
    usable = record["role"] == "uav" and kind == "uav_custom" and path.suffix.lower() in {".pt", ".onnx"}
    record["usable"] = 1 if usable else 0
    if record["status"] in {"downloaded", "already"} and not record["detail"]:
        shown = ", ".join(names) if names else "no class names"
        record["detail"] = shown
    if warning == "coco_not_uav":
        record["usable"] = 0
        record["detail"] = "COCO weights are not a UAV detector. " + record["detail"]


def _probe(url: str, opener) -> tuple[int, str]:
    try:
        response = _open(url, opener, {"Range": "bytes=0-63"})
    except urllib.error.HTTPError as exc:
        if exc.code not in {200, 206}:
            raise RuntimeError(f"link returned HTTP {exc.code}") from exc
        response = exc
    with response:
        status = int(getattr(response, "status", 0) or getattr(response, "code", 0) or 0)
        if status not in {200, 206}:
            raise RuntimeError(f"link returned HTTP {status}")
        final = str(response.geturl())
        if not final.lower().startswith("https://"):
            raise RuntimeError("link redirected away from https")
        content_type = _header(response, "Content-Type")
        sample = response.read(64)
        if _looks_like_html(sample, content_type):
            raise RuntimeError("link returned a web page, not a weight file")
        return status, content_type


def _download(url: str, opener) -> tuple[bytes, int, str]:
    with _open(url, opener, None) as response:
        status = int(getattr(response, "status", 0) or 0)
        if status != 200:
            raise RuntimeError(f"download returned HTTP {status}")
        final = str(response.geturl())
        if not final.lower().startswith("https://"):
            raise RuntimeError("download redirected away from https")
        content_type = _header(response, "Content-Type")
        length = _header(response, "Content-Length")
        if length.isdigit() and int(length) > MAX_BYTES:
            raise RuntimeError("file is larger than the library limit")
        payload = _read_limited(response)
        if _looks_like_html(payload[:64], content_type):
            raise RuntimeError("download returned a web page, not a weight file")
        return payload, status, content_type


def _open(url: str, opener, extra: dict | None):
    headers = {"User-Agent": "UAV-Vision-Detection-Tracking"}
    if extra:
        headers.update(extra)
    request = urllib.request.Request(url, headers=headers, method="GET")
    open_fn = opener or urllib.request.urlopen
    return open_fn(request, timeout=120)


def _read_limited(response) -> bytes:
    chunks = []
    total = 0
    while True:
        block = response.read(1024 * 1024)
        if not block:
            break
        total += len(block)
        if total > MAX_BYTES:
            raise RuntimeError("file is larger than the library limit")
        chunks.append(block)
    if not chunks:
        raise RuntimeError("link returned an empty file")
    return b"".join(chunks)


def _looks_like_html(sample: bytes, content_type: str) -> bool:
    kind = content_type.split(";", 1)[0].strip().lower()
    if kind in {"text/html", "application/xhtml+xml"}:
        return True
    text = sample.lstrip()[:32].lower()
    return text.startswith((b"<!doctype", b"<html", b"<head", b"<body"))


def _header(response, name: str) -> str:
    headers = getattr(response, "headers", None)
    if headers is None:
        return ""
    value = headers.get(name)
    return "" if value is None else str(value)


def _matches(path: Path, expected: str) -> bool:
    if not expected:
        return path.stat().st_size > 0
    return _sha256(path) == expected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _default_names(path: Path) -> list[str]:
    from ultralytics import YOLO

    model = YOLO(str(path))
    return class_names(getattr(model, "names", {}) or {})


def _blank(entry: dict) -> dict:
    return {
        "id": str(entry.get("id") or ""),
        "title": str(entry.get("title") or ""),
        "role": str(entry.get("role") or ""),
        "url": str(entry.get("url") or "").strip(),
        "filename": Path(str(entry.get("filename") or "")).name,
        "local_path": "",
        "sha256": "",
        "size_bytes": 0,
        "content_type": "",
        "http_status": 0,
        "class_names": "[]",
        "kept_classes": "[]",
        "kind": "",
        "warning": "",
        "license": str(entry.get("license") or ""),
        "status": "",
        "detail": "",
        "usable": 0,
        "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _save(connection: sqlite3.Connection, record: dict) -> None:
    connection.execute(
        """
        INSERT INTO weights (
            id, title, role, url, filename, local_path, sha256, size_bytes, content_type,
            http_status, class_names, kept_classes, kind, warning, license, status, detail, usable, checked_at
        ) VALUES (
            :id, :title, :role, :url, :filename, :local_path, :sha256, :size_bytes, :content_type,
            :http_status, :class_names, :kept_classes, :kind, :warning, :license, :status, :detail, :usable, :checked_at
        )
        ON CONFLICT(id) DO UPDATE SET
            title = excluded.title,
            role = excluded.role,
            url = excluded.url,
            filename = excluded.filename,
            local_path = excluded.local_path,
            sha256 = excluded.sha256,
            size_bytes = excluded.size_bytes,
            content_type = excluded.content_type,
            http_status = excluded.http_status,
            class_names = excluded.class_names,
            kept_classes = excluded.kept_classes,
            kind = excluded.kind,
            warning = excluded.warning,
            license = excluded.license,
            status = excluded.status,
            detail = excluded.detail,
            usable = excluded.usable,
            checked_at = excluded.checked_at
        """,
        record,
    )
