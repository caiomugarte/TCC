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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
from pathlib import Path
import shutil
import sys

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
    cache_dir = Path(cache_root) / day.isoformat()
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
    return [_scored_stocks_check(stock_csv, limits)]


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
