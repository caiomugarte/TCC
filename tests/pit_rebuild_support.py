"""Helpers shared by the rebuild tests: a fake urlopen and zip builders. No network.

The module is not a test module (its name does not start with test_), so the
test runner does not collect it. Tests import it as pit_rebuild_support.
"""

import base64
import io
import json
from collections.abc import Mapping
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import unquote
import sys
import urllib.error
import urllib.request
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "py") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "py"))

UNIVERSE = PROJECT_ROOT / "tests" / "fixtures" / "pit_rebuild" / "universe"
DFP_YEARS = range(2010, 2027)
ITR_YEARS = range(2011, 2027)
FCA_YEARS = range(2010, 2027)
COTAHIST_YEARS = range(2011, 2027)
PROBE_YEARS = range(2005, 2013)


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


def universe_routes(dates, universe: Path = UNIVERSE) -> dict[str, FakeResponse]:
    """The responses a fixture universe serves for every URL the rebuild can request (no network).

    universe is a directory laid out like tests/fixtures/pit_rebuild/universe; the default is the shared one.
    """
    from pit_rebuild import sources

    period2 = max(dates) + timedelta(days=1)
    routes: dict[str, FakeResponse] = {}

    def serve(url: str, body: bytes) -> None:
        routes[url] = FakeResponse(body, {"Content-Length": str(len(body))})

    for year in DFP_YEARS:
        serve(sources.dfp_url(year), _zip_folder(universe / "dfp" / str(year)))
    for year in ITR_YEARS:
        serve(sources.itr_url(year), make_zip({}))
    for year in FCA_YEARS:
        serve(sources.fca_url(year), _zip_folder(universe / "fca" / str(year)))
    for year in COTAHIST_YEARS:
        serve(sources.cotahist_url(year), _cotahist_zip(year, universe))
    serve(sources.CAD_URL, (universe / "cad_cia_aberta.csv").read_bytes())
    serve(sources.DFP_LISTING, _listing("DFP", DFP_YEARS))
    serve(sources.ITR_LISTING, _listing("ITR", ITR_YEARS))
    serve(sources.FCA_LISTING, _listing("FCA", FCA_YEARS))
    trading: dict[str, str] = {}
    for path in sorted((universe / "b3").glob("company_*.json")):
        root = path.stem[len("company_"):]
        serve(sources.b3_company_url(root), path.read_bytes())
        trading[root] = json.loads(path.read_bytes())[0]["tradingName"].strip()
    for path in sorted((universe / "b3" / "cash").glob("*_p*.json")):
        root, page = path.stem.rsplit("_p", 1)
        serve(sources.b3_cash_url(trading[root], int(page)), path.read_bytes())
    for path in sorted((universe / "yahoo").glob("*.json")):
        serve(sources.yahoo_url(path.stem, period2), path.read_bytes())
    return routes


def populate_cache(root: Path, dates, universe: Path = UNIVERSE) -> FakeOpener:
    """Fill root with everything the universe serves, through the real Cache (no network).

    The planned files, the optional B3 and Yahoo files, the listings, the Content-Length of
    every yearly file and the COTAHIST probes are stored with their manifest entries, as a
    fetch would store them. Names come from the CLI, so a report finds the files it asks for.
    """
    from pit_rebuild import cache as cache_module
    from pit_rebuild import sources

    fake = FakeOpener(universe_routes(dates, universe))
    store = cache_module.Cache(root, offline=False, opener=fake)
    for url, name in sources.plan_files(dates):
        store.get(url, name)
    for url in fake.routes:
        store.get(url, _name_of(url, dates), required=False)
    for url in fake.routes:
        if _is_yearly(url):
            store.head_size(url)
    for year in PROBE_YEARS:
        store.probe(sources.cotahist_url(year))
    return fake


def _zip_folder(folder: Path) -> bytes:
    """A zip of the files in folder, bytes as they are on disk (the CVM members are latin-1)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        if folder.is_dir():
            for path in sorted(folder.iterdir()):
                if path.is_file():
                    archive.writestr(path.name, path.read_bytes())
    return buffer.getvalue()


def _cotahist_zip(year: int, universe: Path = UNIVERSE) -> bytes:
    text = universe / "cotahist" / f"{year}.txt"
    if not text.is_file():
        return make_zip({})
    return make_zip({f"COTAHIST_A{year}.TXT": text.read_bytes()})


def _listing(source: str, years) -> bytes:
    links = "\n".join(
        f'<a href="{source.lower()}_cia_aberta_{year}.zip">{source.lower()}_cia_aberta_{year}.zip</a>' for year in years
    )
    return f"<html><body><pre>\n{links}\n</pre></body></html>\n".encode("utf-8")


def _is_yearly(url: str) -> bool:
    from pit_rebuild import sources

    return url == sources.CAD_URL or url.startswith(sources.CVM_BASE) or url.startswith(sources.COTAHIST_BASE)


def _name_of(url: str, dates) -> str:
    from pit_rebuild import cli, sources

    if url == sources.DFP_LISTING:
        return cli.listing_name("DFP")
    if url == sources.ITR_LISTING:
        return cli.listing_name("ITR")
    if url == sources.FCA_LISTING:
        return cli.listing_name("FCA")
    if url.startswith(f"{sources.B3_BASE}/GetListedSupplementCompany/"):
        return cli.b3_company_name(_payload(url)["issuingCompany"])
    if url.startswith(f"{sources.B3_BASE}/GetListedCashDividends/"):
        payload = _payload(url)
        return cli.b3_cash_name(payload["tradingName"], payload["pageNumber"])
    if url.startswith(sources.YAHOO_BASE):
        ticker = url[len(sources.YAHOO_BASE) + 1 :].split(".SA?", 1)[0]
        return cli.yahoo_name(ticker, max(dates) + timedelta(days=1))
    return url.rsplit("/", 1)[1]


def _payload(url: str) -> dict:
    encoded = unquote(url.rsplit("/", 1)[1])
    return json.loads(base64.b64decode(encoded))
