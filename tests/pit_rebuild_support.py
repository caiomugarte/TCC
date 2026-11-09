"""Helpers shared by the rebuild tests: a fake urlopen and zip builders. No network.

The module is not a test module (its name does not start with test_), so the
test runner does not collect it. Tests import it as pit_rebuild_support.
"""

import io
from collections.abc import Mapping
import urllib.error
import urllib.request
import zipfile


class FakeResponse:
    """What urlopen returns: a body to read and response headers."""

    def __init__(self, body: bytes = b"", headers: Mapping[str, str] | None = None):
        self._body = body
        self.headers = dict(headers or {})

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class FakeOpener:
    """Stands in for urllib.request.urlopen.

    Every request is recorded as (method, url). A route maps a URL to a
    FakeResponse, or to an exception instance that is raised. An unrouted URL
    answers 404, as a server would.
    """

    def __init__(self, routes: Mapping[str, object] | None = None):
        self.routes = dict(routes or {})
        self.requests: list[tuple[str, str]] = []

    def __call__(self, request, timeout=None):
        if isinstance(request, urllib.request.Request):
            url, method = request.full_url, request.get_method()
        else:
            url, method = str(request), "GET"
        self.requests.append((method, url))
        answer = self.routes.get(url)
        if answer is None:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, io.BytesIO(b""))
        if isinstance(answer, BaseException):
            raise answer
        return answer


def make_zip(members: Mapping[str, bytes | str]) -> bytes:
    """Return zip bytes holding members. Text members are encoded as latin-1, as the CVM files are."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            data = content.encode("latin-1") if isinstance(content, str) else content
            archive.writestr(name, data)
    return buffer.getvalue()
