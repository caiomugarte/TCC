#!/usr/bin/env python3
"""Gera, valida, checa e publica o snapshot do mês de uma vez (issue #85, spec #82).

Rode da raiz do repositório:

    python3 scripts/publish_snapshot.py --date 2026-12-15 --registry snapshots

Etapas, em ordem, paradas na primeira falha:

    series     séries das classes (py/fetch_allocation_snapshot.py, sem as cestas legadas caio_*)
    selecao    CSVs de ações (CVM/B3) e FIIs (Status Invest) (scripts/create_selector_snapshot.py)
    build      manifesto draft (py/snapshot_registry.py build_manifest)
    validate   contrato do manifesto (snapshot_manifest.validate_manifest), draft -> validated
    sanidade   checagens de sanidade com os limites de SanityLimits
    publish    validated -> published; só aqui o manifesto passa a ser escolhido pela API

O manifesto se chama mensal-<--date> e os dados vão para <registry>/data/mensal-<--date>/{series,selecao}/.

Data de corte e idempotência. A data de corte não é --date: é a última data comum às cinco séries das classes
(build_manifest), em geral o pregão anterior a --date. O comando trata "já publicado" assim:

    1. antes de buscar qualquer coisa: se há manifesto publicado com manifest_id mensal-<--date> ou com
       cutoff_date igual a --date, nada é refeito e esse manifesto é devolvido;
    2. depois de buscar, quando a data de corte efetiva é conhecida: se há manifesto publicado com essa mesma data
       de corte (ex.: --date num sábado depois de já publicar na sexta), o novo não é gravado, os dados buscados
       nesta chamada são apagados e o manifesto já publicado é devolvido.

Uma tentativa que falhou deixa mensal-<--date>.json em draft ou validated (não selecionável) e os dados em
data/mensal-<--date>/ para diagnóstico; a próxima chamada com a mesma data apaga esses restos e refaz tudo. Um
manifesto retired com esse id não é apagado: o comando sai com 3.

Cache dos brutos. O Cache do pit_rebuild nunca baixa de novo um arquivo que já tem, então um cache reaproveitado de
um mês para outro deixa COTAHIST, B3 e CVM parados na data da primeira coleta e o universo encolhe sem erro. Por
isso cada --date tem um cache próprio, <--cache-root>/<--date>/, semeado só com os arquivos anuais congelados (DFP,
ITR, FCA, COTAHIST e informes de FII de anos até ano(--date) - 2) do cache de ciclo anterior mais recente em
--cache-root ou de --seed-cache; o resto é baixado de novo. Se <--cache-root>/<--date>/ já existe, é reaproveitado
como está (retomada da mesma data).

Exit codes:

    0  manifesto publicado, ou já existia um publicado com a mesma data de corte (nada refeito)
    1  arquivos não puderam ser lidos ou gravados (registro ou cache)
    2  argumentos inválidos
    3  registro inexistente, ou mensal-<--date> já existe como retired
    4  etapa series falhou
    5  etapa selecao falhou
    6  etapa build ou validate falhou
    7  uma checagem de sanidade reprovou
    8  etapa publish falhou
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT / "py", PROJECT_ROOT / "api", PROJECT_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from snapshot_manifest import SnapshotManifest, SnapshotRegistry, validate_manifest  # noqa: E402
from snapshot_registry import (  # noqa: E402
    build_manifest,
    rewrite_envelope,
    transition_state,
    write_manifest_atomic,
)

MANIFEST_PREFIX = "mensal-"
SERIES_YEARS = 10

EXIT_OK = 0
EXIT_IO = 1
EXIT_USAGE = 2
EXIT_REGISTRY = 3
EXIT_SERIES = 4
EXIT_SELECTION = 5
EXIT_MANIFEST = 6
EXIT_SANITY = 7
EXIT_PUBLISH = 8

STAGE_EXIT = {
    "registro": EXIT_REGISTRY,
    "cache": EXIT_IO,
    "series": EXIT_SERIES,
    "selecao": EXIT_SELECTION,
    "build": EXIT_MANIFEST,
    "validate": EXIT_MANIFEST,
    "sanidade": EXIT_SANITY,
    "publish": EXIT_PUBLISH,
}


@dataclass(frozen=True)
class SanityLimits:
    """Limites das checagens de sanidade antes do publish."""

    min_scored_stocks: int = 40
    min_eligible_fiis: int = 8
    max_universe_drop: float = 0.20
    max_cutoff_lag_days: int = 5


DEFAULT_LIMITS = SanityLimits()


class PublishSnapshotError(RuntimeError):
    """Uma etapa ou checagem falhou; nenhum manifesto novo foi publicado."""

    def __init__(self, stage: str, message: str, *, check: str | None = None, exit_code: int | None = None):
        super().__init__(f"etapa {stage}{f' ({check})' if check else ''}: {message}")
        self.stage = stage
        self.check = check
        self.exit_code = STAGE_EXIT.get(stage, EXIT_IO) if exit_code is None else exit_code


@dataclass(frozen=True)
class PublishResult:
    """O manifesto publicado; ``reused`` diz se ele já existia (nada refeito)."""

    manifest: SnapshotManifest
    manifest_path: Path
    reused: bool
    checks: tuple = ()


def manifest_id_for(day: date) -> str:
    return f"{MANIFEST_PREFIX}{day.isoformat()}"


def _run_stage(stage: str, action: Callable[[], object]):
    try:
        return action()
    except PublishSnapshotError:
        raise
    except Exception as exc:  # noqa: BLE001 - toda falha da etapa vira erro com o nome da etapa
        raise PublishSnapshotError(stage, f"{type(exc).__name__}: {exc}") from exc


def publish_snapshot(
    day: date,
    *,
    registry: Path,
    cache_root: Path,
    seed_cache: Path | None = None,
    limits: SanityLimits = DEFAULT_LIMITS,
    now: Callable[[], datetime] | None = None,
    fetch_series: Callable[[Path, date, date], object] | None = None,
    create_selectors: Callable[[Path, date, Path], object] | None = None,
) -> PublishResult:
    """Gera, valida, checa e publica o snapshot de ``day``; devolve o manifesto publicado.

    Levanta ``PublishSnapshotError`` (com ``stage``, ``check`` e ``exit_code``) quando algo falha; nesse caso o
    snapshot publicado anterior continua sendo o escolhido. ``fetch_series`` e ``create_selectors`` trocam a rede
    nos testes.
    """

    registry = Path(registry)
    clock = now or (lambda: datetime.now(timezone.utc))
    fetch_series = fetch_series or _fetch_series
    create_selectors = create_selectors or _create_selectors
    manifest_id = manifest_id_for(day)
    target = registry / f"{manifest_id}.json"
    data_dir = registry / "data" / manifest_id
    series_dir, selection_dir = data_dir / "series", data_dir / "selecao"

    if not registry.is_dir():
        raise PublishSnapshotError("registro", f"diretório do registro não existe: {registry}")
    existing = _published(registry, lambda manifest: manifest.manifest_id == manifest_id or manifest.cutoff_date == day)
    if existing is not None:
        return PublishResult(existing, Path(existing.manifest_path), reused=True)
    _clear_failed_attempt(target, data_dir)

    start = _years_before(day, SERIES_YEARS)
    _run_stage("series", lambda: fetch_series(series_dir, start, day))
    cache_dir = _run_stage("cache", lambda: prepare_cycle_cache(Path(cache_root), day, seed_cache=seed_cache))
    _run_stage("selecao", lambda: create_selectors(selection_dir, day, cache_dir))

    created_at = clock().isoformat()
    payload = _run_stage("build", lambda: build_manifest(
        registry_root=registry,
        manifest_id=manifest_id,
        created_at=created_at,
        snapshot_dir=series_dir,
        stock_universe=selection_dir / "stocks.csv",
        fii_universe=selection_dir / "fiis.csv",
        stock_provenance=selection_dir / "stocks.provenance.json",
        fii_provenance=selection_dir / "fiis.provenance.json",
    ))
    same_cutoff = _published(registry, lambda manifest: manifest.cutoff_date.isoformat() == payload["cutoff_date"])
    if same_cutoff is not None:
        shutil.rmtree(data_dir)
        return PublishResult(same_cutoff, Path(same_cutoff.manifest_path), reused=True)
    _run_stage("build", lambda: write_manifest_atomic(target, payload))

    def validate() -> None:
        validate_manifest(SnapshotManifest.load(target))
        rewrite_envelope(target, lambda raw: transition_state(raw, "validate", at=clock().isoformat()))

    _run_stage("validate", validate)
    checks = _run_stage("sanidade", lambda: sanity_checks(SnapshotManifest.load(target), day, registry, limits))
    failed = [check for check in checks if not check.passed]
    if failed:
        raise PublishSnapshotError(
            "sanidade", "; ".join(f"{check.name}: {check.detail}" for check in failed), check=failed[0].name,
        )
    _run_stage("publish", lambda: rewrite_envelope(
        target, lambda raw: transition_state(raw, "publish", at=clock().isoformat())))
    return PublishResult(SnapshotManifest.load(target), target, reused=False, checks=tuple(checks))


@dataclass(frozen=True)
class SanityCheck:
    name: str
    passed: bool
    detail: str


def sanity_checks(manifest: SnapshotManifest, day: date, registry: Path, limits: SanityLimits) -> list[SanityCheck]:
    """Todas as checagens de sanidade do manifesto validado de ``day`` (passando ou não)."""
    stock_csv = Path(manifest.selector_source_for("stock").resolved_path())
    fii_csv = Path(manifest.selector_source_for("fii").resolved_path())
    previous = _published(registry, lambda item: item.manifest_id != manifest.manifest_id)
    return [
        _scored_stocks_check(stock_csv, limits),
        _eligible_fiis_check(fii_csv, limits),
        _index_coverage_check(manifest, day, limits),
        SanityCheck(
            "defasagem_corte",
            (day - manifest.cutoff_date).days <= limits.max_cutoff_lag_days,
            f"corte {manifest.cutoff_date.isoformat()} para --date {day.isoformat()}; "
            f"máximo {limits.max_cutoff_lag_days} dias",
        ),
        _universe_drop_check("queda_universo_acoes", "stock", manifest, previous, limits),
        _universe_drop_check("queda_universo_fiis", "fii", manifest, previous, limits),
    ]


def _scored_stocks_check(stock_csv: Path, limits: SanityLimits) -> SanityCheck:
    """Ações com score em cada perfil-âncora (filtros e pesos de STOCK_RULES), pelo caminho da seleção Premium:
    elegibilidade, filtro de robustez e build_scores (que descarta SCORE NaN). O perfil da conta interpola entre as
    âncoras, então o menor dos três limita todos."""
    import contextlib
    import io

    from app.services.premium_policy import STOCK_RULES
    from core.preprocessing import apply_robustness_filter, load_raw_data, preprocess_profile
    from core.scoring import build_scores

    raw = load_raw_data(stock_csv)
    counts = {}
    for band, rules in STOCK_RULES.items():
        with contextlib.redirect_stdout(io.StringIO()):
            clean = preprocess_profile(raw.copy(), filters=rules["filters"])
            if not clean.empty:
                clean = apply_robustness_filter(clean)
            counts[band] = 0 if clean.empty else len(build_scores(clean, factor_weights=rules["factor_weights"]))
    return _minimum_check("acoes_com_score", counts, limits.min_scored_stocks)


def _eligible_fiis_check(fii_csv: Path, limits: SanityLimits) -> SanityCheck:
    """FIIs elegíveis (apply_fii_eligibility) com os filtros de cada perfil-âncora de FII_RULES."""
    from app.services.premium_policy import FII_RULES
    from fii_selection import FiiSelectionError, apply_fii_eligibility, load_fii_data

    raw = load_fii_data(fii_csv)
    counts = {}
    for band, rules in FII_RULES.items():
        try:
            counts[band] = len(apply_fii_eligibility(raw.copy(), rules["filters"]))
        except FiiSelectionError:  # nenhum FII passou
            counts[band] = 0
    return _minimum_check("fiis_elegiveis", counts, limits.min_eligible_fiis)


INDEX_SERIES = {"brazilian_stocks": "IBOV", "fiis": "IFIX", "fixed_income": "CDI"}


def _index_coverage_check(manifest: SnapshotManifest, day: date, limits: SanityLimits) -> SanityCheck:
    """IBOV, IFIX e CDI com dado até a data de corte e no máximo ``max_cutoff_lag_days`` antes de --date.

    O corte é a interseção das séries, então cada uma cobre o corte por construção; o que esta checagem pega é a
    série que parou antes e arrastou o corte para trás, com o nome dela."""
    from allocation_data import read_levels_csv

    floor = day.toordinal() - limits.max_cutoff_lag_days
    last = {}
    for key, label in INDEX_SERIES.items():
        levels = read_levels_csv(Path(manifest.source_for(key).resolved_path()))
        last[label] = max(max(series) for series in levels.values())
    late = [label for label, value in last.items() if value < manifest.cutoff_date or value.toordinal() < floor]
    listed = ", ".join(f"{label} até {value.isoformat()}" for label, value in last.items())
    return SanityCheck(
        "cobertura_indices", not late,
        f"{listed}; corte {manifest.cutoff_date.isoformat()}, --date {day.isoformat()}, "
        f"máximo {limits.max_cutoff_lag_days} dias" + (f"; atrasadas: {', '.join(late)}" if late else ""),
    )


def _universe_drop_check(
    name: str, role: str, manifest: SnapshotManifest, previous: SnapshotManifest | None, limits: SanityLimits,
) -> SanityCheck:
    """Linhas do CSV de seleção contra as do último snapshot publicado; queda acima de ``max_universe_drop`` reprova."""
    current = _csv_rows(Path(manifest.selector_source_for(role).resolved_path()))
    if previous is None:
        return SanityCheck(name, True, f"{current} linhas; sem snapshot publicado anterior")
    before = _csv_rows(Path(previous.selector_source_for(role).resolved_path()))
    drop = 0.0 if before == 0 else (before - current) / before
    return SanityCheck(
        name, drop <= limits.max_universe_drop,
        f"{current} linhas contra {before} em {previous.manifest_id} (queda {drop:.1%}); "
        f"máximo {limits.max_universe_drop:.0%}",
    )


def _csv_rows(path: Path) -> int:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _minimum_check(name: str, counts: dict, minimum: int) -> SanityCheck:
    band, count = min(counts.items(), key=lambda item: item[1])
    listed = ", ".join(f"{key} {value}" for key, value in counts.items())
    return SanityCheck(name, count >= minimum, f"{listed}; mínimo {minimum} (menor: {band})")


def _clear_failed_attempt(target: Path, data_dir: Path) -> None:
    """Apaga o manifesto draft/validated e os dados de uma tentativa anterior que falhou com o mesmo id."""
    if target.exists():
        state = _run_stage("registro", lambda: SnapshotManifest.load(target).state)
        if state not in ("draft", "validated"):
            raise PublishSnapshotError(
                "registro", f"{target.name} já existe como {state or 'legado'}; escolha outra --date ou outro registro",
            )
        target.unlink()
    if data_dir.exists():
        shutil.rmtree(data_dir)


def _published(registry: Path, predicate: Callable[[SnapshotManifest], bool]) -> SnapshotManifest | None:
    """O publicado mais recente (corte, criação, id) que satisfaz ``predicate``, ou None."""
    manifests = _run_stage("registro", lambda: SnapshotRegistry(registry).manifests())
    matches = [manifest for manifest in manifests if manifest.state == "published" and predicate(manifest)]
    if not matches:
        return None
    return max(matches, key=lambda item: (item.cutoff_date, item.created_at, item.manifest_id))


FROZEN_YEARLY = re.compile(
    r"^(?:(?:dfp|itr|fca)_cia_aberta_(?P<cvm>\d{4})\.zip|COTAHIST_A(?P<cotahist>\d{4})\.ZIP"
    r"|inf_mensal_fii_(?P<fii>\d{4})\.zip)$"
)
FROZEN_AGE_YEARS = 2
CACHE_MANIFEST = "manifest.json"


def prepare_cycle_cache(cache_root: Path, day: date, *, seed_cache: Path | None = None) -> Path:
    """O cache de brutos de ``day``: ``<cache_root>/<day>``, criado na primeira vez e reaproveitado depois.

    O Cache do pit_rebuild nunca baixa de novo um arquivo que já tem. Um cache novo por data força COTAHIST, B3
    (classificação, eventos, proventos), cadastro, listagens e os anuais recentes da CVM a serem baixados de novo;
    só os anuais de anos até ano(day) - 2 (que não mudam mais) vêm por hard link do ciclo anterior mais recente em
    ``cache_root`` ou, sem ele, de ``seed_cache``, com as entradas do manifest.json deles (hash e retrieved_at
    originais). Os probes (HEAD dos COTAHIST antigos) também vêm. O diretório é montado ao lado e renomeado no fim,
    então um semeio interrompido não deixa um cache pela metade que seria reaproveitado.
    """
    from pit_rebuild import safety

    cache_root = safety.assert_writable_dir(Path(cache_root), "--cache-root")
    target = cache_root / day.isoformat()
    if target.is_dir():
        return target
    cache_root.mkdir(parents=True, exist_ok=True)
    source = _seed_source(cache_root, day) or (Path(seed_cache) if seed_cache is not None else None)
    manifest = {"files": {}, "sizes": {}, "probes": {}}
    staging = Path(tempfile.mkdtemp(dir=cache_root, prefix=f".{day.isoformat()}."))
    try:
        if source is not None:
            seed = json.loads((source / CACHE_MANIFEST).read_text(encoding="utf-8"))
            frozen = {name: entry for name, entry in seed.get("files", {}).items()
                      if _frozen(name, day) and (source / name).is_file()}
            for name in sorted(frozen):
                _link_or_copy(source / name, staging / name)
            urls = {entry.get("url") for entry in frozen.values()}
            manifest["files"] = frozen
            manifest["sizes"] = {url: size for url, size in seed.get("sizes", {}).items() if url in urls}
            manifest["probes"] = dict(seed.get("probes", {}))
        text = json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
        (staging / CACHE_MANIFEST).write_text(text, encoding="utf-8")
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def _seed_source(cache_root: Path, day: date) -> Path | None:
    """O cache de ciclo mais recente anterior a ``day`` em ``cache_root``."""
    earlier = []
    for path in cache_root.iterdir():
        try:
            other = date.fromisoformat(path.name)
        except ValueError:
            continue
        if other < day and (path / CACHE_MANIFEST).is_file():
            earlier.append((other, path))
    return max(earlier)[1] if earlier else None


def _frozen(name: str, day: date) -> bool:
    match = FROZEN_YEARLY.match(name)
    if match is None:
        return False
    year = int(next(value for value in match.groupdict().values() if value))
    return year <= day.year - FROZEN_AGE_YEARS


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _years_before(day: date, years: int) -> date:
    try:
        return day.replace(year=day.year - years)
    except ValueError:  # 29/02
        return day.replace(year=day.year - years, day=28)


def _fetch_series(snapshot_dir: Path, start: date, end: date) -> None:
    from fetch_allocation_snapshot import fetch_snapshot

    fetch_snapshot(snapshot_dir, None, start, end)


def _create_selectors(out_dir: Path, day: date, cache_dir: Path) -> None:
    raise NotImplementedError
