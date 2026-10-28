#!/usr/bin/env python3
"""_frozen_fitness.py - the frozen recomputation owner (D-e, VC-11).

This module is the single place that turns a row's recorded tickers into a
recomputed fitness under the frozen fitness semantics: `sel["SCORE"].sum()` with
pandas skipna behavior minus `lambda * hhi * n`, over ranked frames rebuilt from
the bundle's own `data/processed/` through the frozen `profiles.build_scores`.
`n * score_mean` is structurally impossible here - no code path in this module
computes a mean-times-n recomputation; a NaN SCORE row therefore recomputes to
the skipna sum and can never be confused with a mean shortcut.

It imports ONLY the bundle's own frozen `code/ga.py` and `code/profiles.py` -
never live `py/` - via `importlib.util.spec_from_file_location` under unique
module names, with `sys.dont_write_bytecode = True` set before the import and
cwd inside a scratch root for the duration of the import (AD-013): importing
frozen code must not write `__pycache__` under the bundle, and nothing here
writes under `code/`, `data/` or `reference/`. Every method after `load` is
pure computation over the already-imported modules.

`evidence_for_row` re-derives one artifact row's evidence fields (D-e
`RowEvidence`): reconstruction against the frozen universe, snapshot match,
the hhi pair, every recorded `score_*` pair, and the reported/recomputed
fitness pair. A row whose chromosome does not reconstruct as exactly `n`
distinct frozen rows is `inexact` with `recomputed = None` - `not_verifiable`,
never a silent pass (VC-10, VC-11).
"""

import sys

sys.dont_write_bytecode = True  # before any bundle import (AD-013)

import importlib.util
import itertools
import os
from dataclasses import dataclass, field
from pathlib import Path

_LOAD_TOKEN = itertools.count()


class FrozenError(Exception):
    """The frozen snapshot cannot answer the question - fail loudly, never guess."""


@dataclass
class RowEvidence:
    """One artifact row's re-derived evidence (D-e). `None` recomputed = not verifiable."""

    row_reconstruction: str  # exact | inexact
    data_snapshot_match: bool
    hhi_reported: float | None
    hhi_recomputed: float | None
    score_fields: dict = field(default_factory=dict)  # name -> (reported, recomputed)
    reported: float | None = None
    recomputed: float | None = None


def _load_code_module(name, path):
    """One frozen source file under a unique module name (registered, never aliased)."""
    path = Path(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FrozenError(f"cannot load frozen module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:  # a broken frozen file must not linger half-imported
        sys.modules.pop(name, None)
        raise
    return module


class FrozenBundle:
    """One bundle's frozen code, ranked frames and recomputation semantics."""

    def __init__(self, bundle, ga, profiles, data_dir):
        self.bundle = Path(bundle)
        self._ga = ga
        self._profiles = profiles
        self._data_dir = Path(data_dir)
        self._ranked = {}

    @classmethod
    def load(cls, bundle, scratch_cwd):
        """Import the bundle's frozen `code/` under a scratch cwd and return the owner.

        `scratch_cwd` is the only cwd the frozen imports may see; the caller keeps
        owning its own cwd afterwards. `sys.dont_write_bytecode` is set first so the
        import writes no `__pycache__` under the bundle (AD-013).
        """
        bundle = Path(bundle).resolve()
        scratch_cwd = Path(scratch_cwd).resolve()
        code_dir = bundle / "code"
        for required in (code_dir / "ga.py", code_dir / "profiles.py",
                         bundle / "data" / "processed"):
            if not required.exists():
                raise FrozenError(f"bundle is missing {required}")
        scratch_cwd.mkdir(parents=True, exist_ok=True)
        sys.dont_write_bytecode = True  # before any bundle import (AD-013)
        token = next(_LOAD_TOKEN)
        saved_cwd = os.getcwd()
        os.chdir(scratch_cwd)  # bundle imports run with cwd inside a scratch root
        try:
            ga = _load_code_module(f"_frozen_fitness_ga_{token}", code_dir / "ga.py")
            profiles = _load_code_module(
                f"_frozen_fitness_profiles_{token}", code_dir / "profiles.py"
            )
        finally:
            os.chdir(saved_cwd)
        return cls(bundle, ga, profiles, bundle / "data" / "processed")

    # ---- frozen frames ------------------------------------------------- #

    def ranked(self, profile):
        """The frozen ranked frame, rebuilt from the bundle's own data/processed/."""
        if profile not in self._ranked:
            try:
                frame = self._profiles.load_profile_data(profile, self._data_dir)
                frame = self._profiles.build_scores(frame, profile)
            except Exception as exc:
                raise FrozenError(f"cannot rank {profile} from {self._data_dir}: {exc}") from exc
            if not frame["TICKER"].is_unique:
                raise FrozenError(
                    f"ranked frame for {profile} has duplicate TICKER rows: "
                    f"chromosome reconstruction would be ambiguous"
                )
            self._ranked[profile] = frame
        return self._ranked[profile]

    def universe(self, profile):
        """The frozen ticker universe one chromosome must reconstruct inside."""
        return set(self.ranked(profile)["TICKER"])

    def n_assets(self, profile):
        return int(self._ga.PERFIL_CONFIG[profile]["n_assets"])

    def lam(self, profile):
        return float(self._ga.PERFIL_CONFIG[profile]["lambda"])

    # ---- recomputation semantics (VC-11) ------------------------------ #

    def _selection(self, profile, tickers):
        """(frame, sel, n, lam) for one ticker set - the row's own rows, nothing else."""
        frame = self.ranked(profile)
        sel = frame[frame["TICKER"].isin(set(tickers))]
        return frame, sel, self.n_assets(profile), self.lam(profile)

    def fitness_of(self, profile, tickers):
        """Frozen semantics over the row's own tickers: ga.fitness = the skipna SCORE
        sum minus lambda*hhi*n (the frozen function's own formula, -inf when the
        selection is not exactly n rows)."""
        frame, _, n, lam = self._selection(profile, tickers)
        mask = frame["TICKER"].isin(set(tickers)).to_numpy()
        return float(self._ga.fitness(frame, mask, lam, n))

    def hhi_of(self, profile, tickers):
        """Sector HHI of the row's own tickers, via the frozen hhi_sector."""
        _, sel, _, _ = self._selection(profile, tickers)
        return float(self._ga.hhi_sector(sel))

    def score_fields_of(self, profile, tickers):
        """mean/median/std of the selected rows' SCORE (the writer's own cells)."""
        _, sel, _, _ = self._selection(profile, tickers)
        scores = sel["SCORE"]
        return {
            "score_mean": float(scores.mean()),
            "score_median": float(scores.median()),
            "score_std": float(scores.std()),
        }

    # ---- evidence re-derivation (D-e, VC-24) --------------------------- #

    def evidence_for_row(self, profile, tickers, recorded):
        """Re-derive one row's evidence fields from its own tickers and recorded cells.

        `recorded` maps the row's recorded cells (at least `fitness`; optionally
        `hhi` and any `score_*`). `recomputed` is `None` unless the chromosome
        reconstructs exactly inside the frozen snapshot - a live-only or duplicate
        ticker yields `inexact` and `recomputed = None`, i.e. not_verifiable,
        never a silent pass (VC-10, VC-11).
        """
        recorded = dict(recorded)
        values = [str(t) for t in tickers]
        universe = self.universe(profile)
        distinct = set(values)
        snapshot = len(distinct) == len(set(values) & universe)
        if len(values) == self.n_assets(profile) and len(distinct) == len(values) and snapshot:
            reconstruction = "exact"
        else:
            reconstruction = "inexact"
        verifiable = reconstruction == "exact" and snapshot

        def _float(key):
            try:
                return float(recorded[key]) if key in recorded and recorded[key] is not None else None
            except (TypeError, ValueError):
                return None

        reported = _float("fitness")
        hhi_reported = _float("hhi")
        recomputed = self.fitness_of(profile, values) if verifiable else None
        hhi_recomputed = self.hhi_of(profile, values) if verifiable else None
        score_names = sorted(k for k in recorded if k.startswith("score_"))
        computed = self.score_fields_of(profile, values) if verifiable else {}
        score_fields = {
            name: (_float(name), computed.get(name)) if name in computed else (_float(name), None)
            for name in score_names
        }
        return RowEvidence(
            row_reconstruction=reconstruction,
            data_snapshot_match=snapshot,
            hhi_reported=hhi_reported,
            hhi_recomputed=hhi_recomputed,
            score_fields=score_fields,
            reported=reported,
            recomputed=recomputed,
        )
