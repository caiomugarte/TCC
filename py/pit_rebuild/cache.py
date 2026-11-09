"""Hash-checked download cache of the rebuild (SRC-01..SRC-04, CLI-02).

Every request the tool makes goes through Cache.get or Cache.head_size. A cached
file is reused only while its sha256 matches manifest.json. A file that differs
from its entry is an integrity failure (exit 4). A failed download leaves the
files already cached untouched. Offline, the cache answers from its files and
makes no request.
"""

from datetime import datetime, timezone
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import tempfile
import urllib.request
import zipfile

from .errors import IntegrityError, SourceError

MANIFEST = "manifest.json"
TIMEOUT_SECONDS = 60
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64)"


class Cache:
    """Files of one cache directory, each verified against manifest.json."""

    def __init__(self, root: Path, offline: bool, opener=urllib.request.urlopen):
        self.root = Path(root)
        self.offline = offline
        self._opener = opener
        self._manifest = self._read_manifest()
        self.used: list[dict] = []

    def get(self, url: str, name: str, required: bool = True) -> Path | None:
        """Return the local path of url, downloading it when it is not cached.

        Raises IntegrityError when a cached file differs from its manifest entry,
        and SourceError when a required file cannot be had. An optional file that
        cannot be had returns None.
        """
        path = self.root / name
        entry = self._manifest["files"].get(name)
        if entry is not None and path.is_file():
            if _sha256_of(path) != entry["sha256"]:
                raise IntegrityError(name, "cached file differs from its manifest entry (sha256)")
            return self._use(entry, path)
        if self.offline:
            if required:
                raise SourceError(url, "not in the cache and --offline is set")
            return None
        try:
            data = self._download(url, name)
        except SourceError:
            if required:
                raise
            return None
        return self._store(url, name, data)

    def verify(self) -> None:
        """Re-hash every cached file the manifest names; raise IntegrityError on a difference."""
        for name, entry in sorted(self._manifest["files"].items()):
            path = self.root / name
            if path.is_file() and _sha256_of(path) != entry["sha256"]:
                raise IntegrityError(name, "cached file differs from its manifest entry (sha256)")

    def head_size(self, url: str) -> int | None:
        """Return the Content-Length of url, from the manifest or from one HEAD request."""
        recorded = self._manifest["sizes"].get(url)
        if recorded is not None or self.offline:
            return recorded
        try:
            with self._opener(_request(url, "HEAD"), timeout=TIMEOUT_SECONDS) as response:
                length = response.headers.get("Content-Length")
        except (OSError, http.client.HTTPException):
            return None
        if length is None or not length.isdigit():
            return None
        self._manifest["sizes"][url] = int(length)
        self._write_manifest()
        return int(length)

    def _download(self, url: str, name: str) -> bytes:
        try:
            with self._opener(_request(url), timeout=TIMEOUT_SECONDS) as response:
                data = response.read()
        except (OSError, http.client.HTTPException) as exc:
            raise SourceError(url, f"download failed: {exc}") from exc
        if name.lower().endswith(".zip"):
            try:
                zipfile.ZipFile(io.BytesIO(data)).close()
            except (zipfile.BadZipFile, OSError) as exc:
                raise SourceError(url, "response is not a zip file") from exc
        return data

    def _store(self, url: str, name: str, data: bytes) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / name
        _write_atomically(path, data)
        entry = {
            "bytes": len(data),
            "name": name,
            "retrieved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sha256": hashlib.sha256(data).hexdigest(),
            "url": url,
        }
        self._manifest["files"][name] = entry
        self._write_manifest()
        return self._use(entry, path)

    def _use(self, entry: dict, path: Path) -> Path:
        if all(item["name"] != entry["name"] for item in self.used):
            self.used.append(dict(entry))
        return path

    def _read_manifest(self) -> dict:
        path = self.root / MANIFEST
        if not path.is_file():
            return {"files": {}, "sizes": {}}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise IntegrityError(MANIFEST, f"unreadable cache manifest: {exc}") from exc

    def _write_manifest(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self._manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
        _write_atomically(self.root / MANIFEST, text.encode("utf-8"))


def _request(url: str, method: str = "GET") -> urllib.request.Request:
    return urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT})


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomically(path: Path, data: bytes) -> None:
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise
