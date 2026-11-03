"""Offline tests for scripts/convergence_analysis.py (issue #9).

Hand-built lots only; no GA runs. The fixture-GA tests live in
tests/test_multirun_convergence.py.
"""

from collections import Counter
import contextlib
import copy
from fractions import Fraction
import io
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import convergence_analysis as ca  # noqa: E402
import multirun_convergence as mc  # noqa: E402

PROTOCOL_PATH = PROJECT_ROOT / "docs" / "validation" / "multirun-convergence-protocol.json"
PREMIUM_RECORD = {
    "suitability_score": 0.582,
    "raw_score": 0.952,
    "dimensions": {"apetite": 0.952, "capacidade": 0.582, "liquidez": 1.0, "conhecimento": 0.5625},
    "generic_profile": "moderado",
    "restrictions": ["nenhuma"],
}
STOCK_UNIVERSE = ["POMO3", "POMO4", "VALE3", "PETR4", "ITUB4", "BBAS3", "WEGE3", "ABEV3"]
FII_UNIVERSE = ["KNCR11", "KNRI11", "MXRF11", "XPML11", "HGLG11", "BTLG11", "VISC11", "RECR11"]
SCORES = {t: 0.9 - 0.07 * i for i, t in enumerate(STOCK_UNIVERSE)}
SCORES.update({t: 0.85 - 0.06 * i for i, t in enumerate(FII_UNIVERSE)})


def small_protocol(**changes):
    """The committed protocol scaled down: 3 seeds, 6 runs, n 2, window 5."""

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol.update(
        {
            "seeds": [1, 2, 3],
            "profiles": {"p": {"record": dict(PREMIUM_RECORD), "expected_n_assets": {"stocks": 2, "fiis": 2}}},
            "production_run": {"id": "x", "profile": "p", "seed": 1, "stocks": ["POMO4", "VALE3"], "fiis": ["KNCR11", "KNRI11"]},
            "ga_override": {"adaptive_mode": False, "run_count": 6},
            "prefixes": [2, 4, 6],
            "curve_step": 2,
            "rank_window_max": 5,
        }
    )
    protocol["production_stop_rule"]["min_runs"] = 2
    protocol["production_stop_rule"]["check_every"] = {"stocks": 2, "fiis": 1}
    protocol["decision"]["run_count"]["candidates"] = [2, 4, 6]
    protocol["decision"]["target"]["prefix"] = 6
    for key, value in changes.items():
        protocol[key] = value
    return protocol


def selector_lot(selector, runs, n, fitness=None, scores=SCORES):
    picked = sorted({t for run in runs for t in run})
    candidates = {t: {"score": scores[t], "sector": "S" + t[0]} for t in picked}
    fitness = fitness or [1.0 + 0.1 * (i % 3) for i in range(len(runs))]
    records = [
        {"run_id": i, "seed": 1000 + i, "tickers": sorted(run), "fitness": fitness[i], "hhi": 0.5, "generations_run": 5}
        for i, run in enumerate(runs)
    ]
    selected = ca.production_consensus(records, candidates, selector, n)
    return {
        "n_assets": n,
        "selection_config": {"n_assets": n},
        "run_count": len(runs),
        "selected_tickers": selected,
        "candidates": candidates,
        "runs": records,
    }


def random_runs(universe, rng, count, size=2):
    return [rng.sample(universe, size) for _ in range(count)]


def make_lot(protocol, sha, profile_id, seed, stocks_runs, fiis_runs, **fields):
    n = protocol["profiles"][profile_id]["expected_n_assets"]
    lot = {
        "schema": ca.LOT_SCHEMA,
        "mode": "measured",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha,
        "protocol_commit": "c0ffee",
        "protocol_committed_at": "2026-11-03T10:00:00-03:00",
        "profile_id": profile_id,
        "seed": seed,
        "policy_sha256": "p" * 64,
        "snapshot": {"manifest_id": "m", "manifest_sha256": "a" * 64, "sources": {"allocation": "1", "stock": "2", "fii": "3"}},
        "code": {
            "commit": "abc",
            "code_fingerprint": "code",
            "code_trees": {"py": "t"},
            "experiment_fingerprint": "exp",
            "experiment_trees": {"scripts/x": "t"},
        },
        "selectors": {
            "stocks": selector_lot("stocks", stocks_runs, n["stocks"]),
            "fiis": selector_lot("fiis", fiis_runs, n["fiis"]),
        },
    }
    lot.update(fields)
    return lot


def cost_file(profile_id, seed, mode="measured"):
    return {
        "lot": f"{profile_id}:{seed}",
        "mode": mode,
        "started_at": "2026-11-03T12:00:00+00:00",
        "finished_at": "2026-11-03T13:00:00+00:00",
        "wall_s": 3600.0 + seed,
        "cpu_s": 3500.0 + seed,
        "stages": {"stocks": {"wall_s": 1000.0 + seed, "cpu_s": 990.0 + seed}, "fiis": {"wall_s": 2000.0, "cpu_s": 1990.0}},
        "rss_peak_kb": {"sampled": 1},
        "jobs": 5,
        "load_avg_start": [1.0, 1.0, 1.0],
        "load_avg_end": [5.0, 5.0, 5.0],
        "machine": {"fingerprint": "machine-1", "cpu_model": "cpu", "python": "3.14.4", "libraries": {"numpy": "2.3.4", "pandas": "2.3.3"}},
    }


class LotSet:
    """A protocol file and a lots directory written to a temporary folder."""

    def __init__(self, protocol=None, seed=7, mode="measured", lot_changes=None):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.protocol = protocol or small_protocol()
        self.protocol_path = self.root / "protocol.json"
        self.protocol_path.write_text(json.dumps(self.protocol), encoding="utf-8")
        self.sha = ca.sha256_bytes(self.protocol_path.read_bytes())
        self.lots_dir = self.root / "lots"
        self.out_dir = self.root / "out"
        rng = random.Random(seed)
        run_count = self.protocol["ga_override"]["run_count"]
        self.lots = {}
        for profile_id in self.protocol["profiles"]:
            for s in self.protocol["seeds"]:
                lot = make_lot(
                    self.protocol,
                    self.sha,
                    profile_id,
                    s,
                    random_runs(STOCK_UNIVERSE, rng, run_count),
                    random_runs(FII_UNIVERSE, rng, run_count),
                    mode=mode,
                )
                self.lots[(profile_id, s)] = lot
        declared = self.protocol["production_run"]
        production_lot = self.lots[(declared["profile"], declared["seed"])]
        self.protocol["production_run"]["stocks"] = list(production_lot["selectors"]["stocks"]["selected_tickers"])
        self.protocol["production_run"]["fiis"] = list(production_lot["selectors"]["fiis"]["selected_tickers"])
        self.protocol_path.write_text(json.dumps(self.protocol), encoding="utf-8")
        self.sha = ca.sha256_bytes(self.protocol_path.read_bytes())
        for lot in self.lots.values():
            lot["protocol_sha256"] = self.sha
        self.write(mode)

    def write(self, mode="measured"):
        for (profile_id, s), lot in self.lots.items():
            path = ca.lot_path(self.lots_dir, profile_id, s)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(ca.dumps(lot), encoding="utf-8")
            ca.cost_path(self.lots_dir, profile_id, s).write_text(ca.dumps(cost_file(profile_id, s, mode)), encoding="utf-8")
        summary = {"jobs": 5, "experiment_wall_s": 7300.5, "first_started_at": "a", "last_finished_at": "b", "lots": []}
        (self.lots_dir / "run-summary.json").write_text(ca.dumps(summary), encoding="utf-8")

    def report(self):
        stderr = io.StringIO()
        with patch("sys.stderr", stderr):
            code = mc.main(
                ["report", "--protocol", str(self.protocol_path), "--lots-dir", str(self.lots_dir), "--out-dir", str(self.out_dir)]
            )
        return code, stderr.getvalue()

    def report_json(self):
        code, err = self.report()
        assert code == 0, err
        return json.loads((self.out_dir / "multirun-convergence.json").read_text(encoding="utf-8"))

    def close(self):
        self.temp.cleanup()


class Stub:
    """A seed's series reduced to what the zone and gap functions read."""

    def __init__(self, n, consensus, ranking, counts):
        self.n = n
        self._consensus = list(consensus)
        self._ranking = list(ranking)
        self._counts = Counter(counts)

    def consensus(self, k):
        return self._consensus

    def ranking(self, k):
        return self._ranking

    def counts(self, k):
        return self._counts


# --------------------------------------------------------------------------
# S1
# --------------------------------------------------------------------------


class ProtocolFileTests(unittest.TestCase):
    def test_committed_protocol_values(self):
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        ca.validate_protocol(protocol)
        expected = {
            "protocol_id": "multirun-convergence-v1",
            "issue": 9,
            "snapshot_declaration": "docs/validation/premium-performance-snapshot.json",
            "capital_brl": 100000,
            "seeds": [46641103, 42, 43, 44, 45],
            "profiles": {
                "premium-2026-11-02": {"record": PREMIUM_RECORD, "expected_n_assets": {"stocks": 13, "fiis": 13}},
                "arrojado-anchor": {"record": {"anchor": "arrojado"}, "expected_n_assets": {"stocks": 15, "fiis": 15}},
            },
            "production_run": {
                "id": "9e22cc16-cfda-495e-a8c9-c0d202ef32b6",
                "profile": "premium-2026-11-02",
                "seed": 46641103,
                "stocks": ["BBSE3", "BMOB3", "BRSR6", "CXSE3", "FESA4", "FIQE3", "GMAT3", "ISAE4", "LAVV3", "LOGG3", "MILS3", "RDOR3", "VULC3"],
                "fiis": ["MXRF11", "KNCR11", "BTLG11", "BPML11", "ALZR11", "GGRC11", "KNRI11", "XPML11", "RECT11", "BRCR11", "RBRF11", "KNUQ11", "VGIR11"],
            },
            "ga_override": {"adaptive_mode": False, "run_count": 150},
            "prefixes": [30, 60, 100, 150],
            "curve_step": 10,
            "rank_window_max": 30,
            "production_stop_rule": {"target_cv": 0.02, "target_jaccard": 0.75, "min_runs": 40, "check_every": {"stocks": 10, "fiis": 1}},
            "decision": {
                "run_count": {"candidates": [30, 60, 100, 150], "epsilon_swaps": 0.5, "statistics": ["mean", "median"], "leave_one_seed_out": "max"},
                "target": {"prefix": 150, "targets": ["target_jaccard", "target_cv"]},
                "zone": {
                    "quantile": 0.95,
                    "quantile_method": "inverted_cdf",
                    "delta_round_up": 0.01,
                    "delta_arithmetic": "fraction_from_counts",
                    "delta_grid": [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2],
                },
            },
        }
        self.assertEqual(protocol, expected)


class ReadmeTests(unittest.TestCase):
    def test_readme_section(self):
        text = (PROJECT_ROOT / "docs" / "validation" / "README.md").read_text(encoding="utf-8")
        section = text[text.index("# Convergência do multi-run"):]
        for needle in (
            "multirun-convergence-protocol.json",
            "multirun-convergence/lots/<perfil>/seed-<seed>.json",
            "seed-<seed>.cost.json",
            "run-summary.json",
            "multirun-convergence.json",
            "multirun-convergence-cost.json",
            "multirun-convergence.md",
            "**Giro entre seeds**",
            "Não é o\ngiro mês a mês",
            "ε = 0,5 troca na média e\n  na mediana",
            "**máximo**",
            "leave-one-seed-out",
            "quantil 0,95\n  por CDF invertida",
            "exatamente a partir das contagens",
            "arredondado para cima em 0,01",
            "`target_jaccard` 0,75",
            "`target_cv` 0,02",
            "julgados juntos a 150 execuções",
        ):
            self.assertIn(needle, section, needle)


# --------------------------------------------------------------------------
# S3 - metrics, curves, ranking, stop, integrity
# --------------------------------------------------------------------------


class MetricTests(unittest.TestCase):
    def test_fitness_cv_examples(self):
        self.assertAlmostEqual(ca.fitness_cv([1, 2, 3]), 0.408248290463863, delta=1e-15)
        self.assertAlmostEqual(ca.fitness_cv([1, 2, 3]), math.sqrt(2 / 3) / 2, delta=1e-15)
        self.assertEqual(ca.fitness_cv([5, 5, 5]), 0.0)
        self.assertEqual(ca.fitness_cv([-1, 1]), 0.0)
        self.assertLessEqual(abs(ca.fitness_cv([1.1, 2.7, 0.4]) - ca._production_cv([1.1, 2.7, 0.4])), 1e-12)

    def test_jaccard_examples(self):
        self.assertEqual(ca.jaccard_mean([{"A", "B"}, {"A", "C"}]), 1 / 3)
        self.assertEqual(ca.jaccard_mean([{"A", "B"}, {"A", "B"}, {"C", "D"}]), 1 / 3)
        self.assertEqual(ca.jaccard_mean([{"A"}, {"A"}]), 1.0)
        self.assertEqual(ca.jaccard_mean([{"A", "B"}]), 0.0)
        for a, b in (({"A", "B"}, {"A", "C"}), ({"A"}, {"B"}), (set(), set()), ({"A", "B", "C"}, {"B", "C"})):
            self.assertEqual(ca.jaccard(a, b), ca._production_jaccard(a, b))


class CurveTests(unittest.TestCase):
    def setUp(self):
        self.lots = LotSet()

    def tearDown(self):
        self.lots.close()

    def test_curves_and_production_crosscheck(self):
        report = self.lots.report_json()
        from fii_selection import _fii_stability
        from pipelines.multi_run import analyze_stability

        for profile_id in self.lots.protocol["profiles"]:
            for selector in ca.SELECTORS:
                cell = report["cells"][f"{profile_id}/{selector}"]
                for seed in self.lots.protocol["seeds"]:
                    points = cell["curves"]["by_seed"][str(seed)]
                    self.assertEqual([p["k"] for p in points], [2, 4, 6])
                    runs = self.lots.lots[(profile_id, seed)]["selectors"][selector]["runs"]
                    for point in points:
                        head = runs[: point["k"]]
                        fitness = [r["fitness"] for r in head]
                        self.assertEqual(point["fitness_cv"], ca.fitness_cv(fitness))
                        self.assertEqual(point["jaccard_mean"], ca.jaccard_mean([set(r["tickers"]) for r in head]))
                        self.assertEqual(point["fitness_mean"], math.fsum(fitness) / len(fitness))
                        self.assertEqual((point["fitness_min"], point["fitness_max"]), (min(fitness), max(fitness)))
                    if selector == "stocks":
                        stability = analyze_stability(copy.deepcopy(runs))
                        cv, jm = stability["fitness"]["cv"], stability["portfolio_similarity"]["jaccard_mean"]
                    else:
                        stability = _fii_stability(runs)
                        cv, jm = stability["fitness_cv"], stability["jaccard_mean"]
                    self.assertLessEqual(abs(points[-1]["fitness_cv"] - cv), 1e-12)
                    self.assertLessEqual(abs(points[-1]["jaccard_mean"] - jm), 1e-12)
                for index, row in enumerate(cell["curves"]["summary"]):
                    for metric in ("fitness_cv", "jaccard_mean", "fitness_mean", "fitness_min", "fitness_max"):
                        values = [cell["curves"]["by_seed"][str(s)][index][metric] for s in self.lots.protocol["seeds"]]
                        self.assertEqual(row[metric], {"mean": math.fsum(values) / 3, "min": min(values), "max": max(values)})

    def test_production_jaccard_mismatch_exits_4(self):
        original = ca._production_stability

        def off(selector, runs):
            cv, jm = original(selector, runs)
            return cv, jm + 1e-9

        with patch.object(ca, "_production_stability", side_effect=off):
            code, err = self.lots.report()
        self.assertEqual(code, ca.EXIT_INTEGRITY)
        self.assertIn("p:1", err)
        self.assertFalse(self.lots.out_dir.exists())


class FrequencyTests(unittest.TestCase):
    def test_frequencies(self):
        lots = LotSet()
        try:
            report = lots.report_json()
            for selector in ca.SELECTORS:
                cell = report["cells"][f"p/{selector}"]
                self.assertEqual([item["k"] for item in cell["frequencies"]], [2, 4, 6])
                for item in cell["frequencies"]:
                    k = item["k"]
                    for seed in lots.protocol["seeds"]:
                        runs = lots.lots[("p", seed)]["selectors"][selector]["runs"][:k]
                        counts = Counter(t for r in runs for t in r["tickers"])
                        self.assertEqual(
                            item["by_seed"][str(seed)],
                            {t: {"count": c, "frequency": c / k} for t, c in counts.items()},
                        )
        finally:
            lots.close()


class RankingTests(unittest.TestCase):
    def test_extended_ranking_examples(self):
        scores = {"POMO4": 0.5, "POMO3": 0.9, "VALE3": 0.1}
        counts = Counter({"POMO4": 10, "POMO3": 9, "VALE3": 8})
        self.assertEqual(ca.extended_ranking(counts, scores, "stocks"), ["POMO4", "VALE3"])
        self.assertEqual(
            ca.extended_ranking(Counter({"AAAA3": 5, "BBBB3": 5}), {"AAAA3": 0.2, "BBBB3": 0.8}, "stocks"),
            ["BBBB3", "AAAA3"],
        )
        self.assertEqual(
            ca.extended_ranking(Counter({"DDDD3": 5, "CCCC3": 5}), {"DDDD3": 0.5, "CCCC3": 0.5}, "stocks"),
            ["CCCC3", "DDDD3"],
        )
        self.assertEqual(
            ca.extended_ranking(Counter({"KNRI11": 4, "KNCR11": 4}), {"KNRI11": 0.7, "KNCR11": 0.6}, "fiis"),
            ["KNRI11", "KNCR11"],
        )

    def test_consensus_is_production_function(self):
        runs = [{"tickers": ["POMO3", "POMO4"]}, {"tickers": ["POMO4", "VALE3"]}, {"tickers": ["POMO3", "VALE3"]}]
        candidates = {"POMO3": {"score": 0.9, "sector": "A"}, "POMO4": {"score": 0.5, "sector": "A"}, "VALE3": {"score": 0.1, "sector": "B"}}
        self.assertEqual(ca.production_consensus(runs, candidates, "stocks", 2), ["POMO3", "VALE3"])
        with patch("pipelines.multi_run.build_consensus_portfolio") as production:
            import pandas as pd

            production.return_value = pd.DataFrame({"TICKER": ["ZZZZ3"]})
            self.assertEqual(ca.production_consensus(runs, candidates, "stocks", 2), ["ZZZZ3"])
        self.assertEqual(
            ca.production_consensus(
                [{"tickers": ["KNCR11", "KNRI11"]}],
                {"KNCR11": {"score": 0.1, "sector": None}, "KNRI11": {"score": 0.2, "sector": None}},
                "fiis",
                2,
            ),
            ["KNCR11", "KNRI11"],
        )

    def test_nan_score_keeps_production_order(self):
        rng = random.Random(3)
        universe = ["AAAA3", "BBBB3", "CCCC3", "DDDD3", "EEEE3", "FFFF4", "GGGG3"]
        scores = {"AAAA3": None, "BBBB3": 0.4, "CCCC3": None, "DDDD3": 0.9, "EEEE3": 0.4, "FFFF4": None, "GGGG3": 0.1}
        candidates = {t: {"score": s, "sector": "X"} for t, s in scores.items()}
        for _ in range(40):
            runs = [{"tickers": sorted(rng.sample(universe, 3))} for _ in range(6)]
            picked = {t for r in runs for t in r["tickers"]}
            subset = {t: candidates[t] for t in picked}
            ranking = ca.extended_ranking(
                ca.count_picks(runs), {t: ca.score_value(c) for t, c in subset.items()}, "stocks"
            )
            self.assertEqual(sorted(ranking[:3]), ca.production_consensus(runs, subset, "stocks", 3))


class StopTests(unittest.TestCase):
    RULE = {"target_cv": 0.02, "target_jaccard": 0.75, "min_runs": 40, "check_every": {"stocks": 10, "fiis": 1}}

    def metrics(self, first, run_count=150):
        return {k: ((0.01, 0.8) if first is not None and k >= first else (0.05, 0.1)) for k in range(1, run_count + 1)}

    def test_adaptive_stop_examples(self):
        stop = ca.adaptive_stop_at
        self.assertEqual(stop(self.metrics(43), 150, self.RULE, "fiis"), 43)
        self.assertEqual(stop(self.metrics(43), 150, self.RULE, "stocks"), 50)
        self.assertIsNone(stop(self.metrics(None), 150, self.RULE, "fiis"))
        self.assertIsNone(stop(self.metrics(None), 150, self.RULE, "stocks"))
        self.assertEqual(stop(self.metrics(1), 150, self.RULE, "fiis"), 40)
        self.assertEqual(stop(self.metrics(1), 150, self.RULE, "stocks"), 40)
        self.assertEqual(stop(self.metrics(145), 150, self.RULE, "stocks"), 150)
        self.assertEqual(stop(self.metrics(145), 150, self.RULE, "fiis"), 145)
        self.assertEqual(stop(self.metrics(145, 147), 147, self.RULE, "stocks"), 147)
        exact = {k: (0.02, 0.75) for k in range(1, 151)}
        self.assertEqual(stop(exact, 150, self.RULE, "fiis"), 40)
        just_off = {k: (0.0200001, 0.75) for k in range(1, 151)}
        self.assertIsNone(stop(just_off, 150, self.RULE, "fiis"))

    def test_identical_runs_stop_at_min_runs_in_the_report(self):
        protocol = small_protocol()
        protocol["production_stop_rule"]["min_runs"] = 4
        lots = LotSet(protocol)
        try:
            for lot in lots.lots.values():
                for selector, run in (("stocks", ["POMO4", "VALE3"]), ("fiis", ["KNCR11", "KNRI11"])):
                    lot["selectors"][selector] = selector_lot(selector, [run] * 6, 2, fitness=[2.0] * 6)
            first = lots.lots[("p", 1)]["selectors"]
            lots.protocol["production_run"]["stocks"] = first["stocks"]["selected_tickers"]
            lots.protocol["production_run"]["fiis"] = first["fiis"]["selected_tickers"]
            lots.protocol_path.write_text(json.dumps(lots.protocol), encoding="utf-8")
            lots.sha = ca.sha256_bytes(lots.protocol_path.read_bytes())
            for lot in lots.lots.values():
                lot["protocol_sha256"] = lots.sha
            lots.write()
            report = lots.report_json()
            self.assertEqual(report["cells"]["p/fiis"]["adaptive_stop"]["by_seed"], {"1": 4, "2": 4, "3": 4})
            self.assertEqual(report["cells"]["p/stocks"]["adaptive_stop"]["by_seed"], {"1": 4, "2": 4, "3": 4})
        finally:
            lots.close()


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.lots = LotSet()

    def tearDown(self):
        self.lots.close()

    def assert_exit_4(self, label, mutate, where="p:2"):
        lots = LotSet()
        try:
            mutate(lots)
            lots.write()
            code, err = lots.report()
            self.assertEqual(code, ca.EXIT_INTEGRITY, f"{label}: {err}")
            self.assertIn(where, err, label)
            self.assertEqual(len(err.strip().splitlines()), 1, label)
            self.assertFalse(lots.out_dir.exists(), label)
        finally:
            lots.close()

    def test_bad_lots_exit_4(self):
        def runs(lots):
            return lots.lots[("p", 2)]["selectors"]["stocks"]["runs"]

        def drop(lots):
            del runs(lots)[3]

        def duplicate(lots):
            runs(lots)[4] = copy.deepcopy(runs(lots)[3])

        def swap(lots):
            r = runs(lots)
            r[1], r[2] = r[2], r[1]

        def missing_lot(lots):
            ca.lot_path(lots.lots_dir, "p", 2).unlink()
            del lots.lots[("p", 2)]

        cases = {
            "missing run": drop,
            "duplicated run": duplicate,
            "out of order": swap,
            "protocol_sha256": lambda l: l.lots[("p", 2)].update(protocol_sha256="0" * 64),
            "snapshot": lambda l: l.lots[("p", 2)]["snapshot"].update(manifest_sha256="b" * 64),
            "code_fingerprint": lambda l: l.lots[("p", 2)]["code"].update(code_fingerprint="other"),
            "experiment_fingerprint": lambda l: l.lots[("p", 2)]["code"].update(experiment_fingerprint="other"),
            "mode": lambda l: l.lots[("p", 2)].update(mode="smoke"),
        }
        for label, mutate in cases.items():
            self.assert_exit_4(label, mutate)

        lots = LotSet()
        try:
            ca.lot_path(lots.lots_dir, "p", 2).unlink()
            code, err = lots.report()
            self.assertEqual(code, ca.EXIT_INTEGRITY)
            self.assertIn("p:2", err)
            self.assertFalse(lots.out_dir.exists())
        finally:
            lots.close()

    def test_consensus_mismatch_exits_4(self):
        def wrong_selected(lots):
            selected = lots.lots[("p", 2)]["selectors"]["fiis"]["selected_tickers"]
            other = next(t for t in FII_UNIVERSE if t not in selected)
            selected[0] = other

        def missing_candidate(lots):
            candidates = lots.lots[("p", 2)]["selectors"]["stocks"]["candidates"]
            del candidates[sorted(candidates)[0]]

        self.assert_exit_4("selected_tickers", wrong_selected)
        self.assert_exit_4("candidates", missing_candidate)

    def test_cv_production_mismatch_exits_4(self):
        original = ca._production_cv
        with patch.object(ca, "_production_cv", side_effect=lambda values: original(values) + 2e-12):
            code, err = self.lots.report()
        self.assertEqual(code, ca.EXIT_INTEGRITY)
        self.assertIn("p:1", err)
        self.assertIn("coefficient_of_variation", err)
        self.assertFalse(self.lots.out_dir.exists())
        with patch.object(ca, "_production_cv", side_effect=lambda values: original(values) + 5e-13):
            code, err = self.lots.report()
        self.assertEqual(code, ca.EXIT_OK, err)


# --------------------------------------------------------------------------
# S4 - turnover and rank gap
# --------------------------------------------------------------------------


class TurnoverTests(unittest.TestCase):
    def test_seed_pairs(self):
        a = [f"T{i:02d}" for i in range(13)]
        b = a[:10] + ["X1", "X2", "X3"]
        self.assertEqual(ca.swapped(13, a, b), 3)
        self.assertEqual(ca.jaccard(set(a), set(b)), 0.625)
        self.assertEqual((ca.swapped(13, a, a), ca.jaccard(set(a), set(a))), (0, 1.0))
        series = {
            1: Stub(13, a, a, {}),
            2: Stub(13, b, b, {}),
            3: Stub(13, a[:12] + ["Y"], a, {}),
        }
        turnover = ca.seed_turnover(series, [1, 2, 3], 30, 13)
        self.assertEqual([p["seeds"] for p in turnover["pairs"]], [[1, 2], [1, 3], [2, 3]])
        self.assertEqual([p["swapped"] for p in turnover["pairs"]], [3, 1, 3])
        self.assertEqual(turnover["swapped"], {"mean": 7 / 3, "median": 3, "min": 1, "max": 3})
        self.assertEqual(turnover["jaccard"]["max"], 12 / 14)


class RankGapTests(unittest.TestCase):
    def test_rank_gap(self):
        ranking = [f"T{i:02d}" for i in range(25)]
        counts = {t: 150 - 6 * i for i, t in enumerate(ranking)}
        counts[ranking[12]] = 60
        counts[ranking[13]] = 54
        gap = ca.rank_gap(Stub(13, ranking[:13], ranking, counts), 150, 30)
        self.assertEqual(gap["ranks_available"], 25)
        self.assertEqual([f["rank"] for f in gap["frequencies"]], list(range(1, 26)))
        self.assertEqual(gap["f_n"], 0.4)
        self.assertEqual(gap["f_n_plus_1"], 0.36)
        self.assertAlmostEqual(gap["gap"], 0.04, delta=1e-12)
        lots = LotSet()
        try:
            report = lots.report_json()
            for item in report["cells"]["p/stocks"]["rank_gap"]:
                gaps = [item["by_seed"][str(s)]["gap"] for s in (1, 2, 3)]
                self.assertEqual(item["median_gap"], sorted(gaps)[1])
                self.assertEqual(sorted(item["median_frequency_by_rank"]), ["3", "4", "5"])
        finally:
            lots.close()


class WithinSeedTests(unittest.TestCase):
    def test_within_seed(self):
        lots = LotSet()
        try:
            report = lots.report_json()
            for selector in ca.SELECTORS:
                cell = report["cells"][f"p/{selector}"]
                final = cell["consensus"][-1]["by_seed"]
                for item, consensus in zip(cell["within_seed_turnover"], cell["consensus"]):
                    values = {s: 2 - len(set(consensus["by_seed"][s]) & set(final[s])) for s in final}
                    self.assertEqual(item["by_seed"], values)
                    self.assertEqual(item["max"], max(values.values()))
                    self.assertEqual(item["mean"], math.fsum(values.values()) / 3)
                    self.assertEqual(item["median"], sorted(values.values())[1])
                self.assertEqual(set(cell["within_seed_turnover"][-1]["by_seed"].values()), {0})
        finally:
            lots.close()
        at150 = [f"T{i:02d}" for i in range(13)]
        at30 = at150[:10] + ["X1", "X2", "X3"]
        self.assertEqual(ca.swapped(13, at30, at150), 3)


# --------------------------------------------------------------------------
# S5 - zone
# --------------------------------------------------------------------------


class ZoneTests(unittest.TestCase):
    INCUMBENTS = ["A", "B", "C"]

    def scenario(self, e_count=14, e_score=0.6):
        counts = Counter({"D": 18, "A": 16, "E": e_count, "B": 13, "C": 12})
        scores = {"D": 0.9, "A": 0.8, "E": e_score, "B": 0.7, "C": 0.5}
        return counts, ca.extended_ranking(counts, scores, "fiis")

    def zone(self, K, delta, **kwargs):
        counts, ranking = self.scenario(**kwargs)
        return ca.zone_portfolio(self.INCUMBENTS, ranking, counts, 20, K, delta, "fiis")

    def test_zone_examples(self):
        self.assertEqual(self.zone(5, 0.1), ({"A", "B", "D"}, 1))
        self.assertEqual(self.zone(4, 0.1), ({"A", "B", "D"}, 1))
        self.assertEqual(self.zone(0, 0.1), ({"D", "A", "E"}, 2))
        self.assertEqual(self.zone(5, 0.1, e_count=15), ({"A", "D", "E"}, 2))
        self.assertEqual(self.zone(5, 0.0, e_count=13, e_score=0.6), ({"A", "B", "D"}, 1))

    def test_k0_is_pure_consensus(self):
        rng = random.Random(11)
        universe = [f"T{i}" for i in range(12)]
        for _ in range(30):
            counts = Counter({t: rng.randint(0, 20) for t in universe})
            counts = Counter({t: c for t, c in counts.items() if c})
            scores = {t: rng.random() for t in counts}
            ranking = ca.extended_ranking(counts, scores, "fiis")
            incumbents = rng.sample(universe, 4)
            for delta in (0.0, 0.01, 0.05, 0.2):
                portfolio, _ = ca.zone_portfolio(incumbents, ranking, counts, 20, 0, delta, "fiis")
                self.assertEqual(portfolio, set(ranking[:4]))

    def test_stock_dedup(self):
        counts = Counter({"POMO3": 10, "POMO4": 9, "VALE3": 8, "PETR4": 7})
        scores = {"POMO3": 0.5, "POMO4": 0.9, "VALE3": 0.4, "PETR4": 0.3}
        ranking = ca.extended_ranking(counts, scores, "stocks")
        self.assertEqual(ranking, ["POMO3", "VALE3", "PETR4"])
        # POMO4 has no rank (POMO3 holds its base company), so only PETR4 is protected.
        portfolio, turnover = ca.zone_portfolio(["POMO4", "PETR4"], ranking, counts, 10, 30, 0.15, "stocks")
        self.assertEqual((portfolio, turnover), ({"POMO3", "PETR4"}, 1))
        portfolio, turnover = ca.zone_portfolio(["POMO4", "PETR4"], ranking, counts, 10, 30, 0.1, "stocks")
        self.assertEqual((portfolio, turnover), ({"POMO3", "VALE3"}, 2))


class ZoneGridTests(unittest.TestCase):
    def test_grid_shape(self):
        lots = LotSet()
        try:
            report = lots.report_json()
            deltas = lots.protocol["decision"]["zone"]["delta_grid"]
            for cell_id, cell in report["cells"].items():
                self.assertEqual([g["k"] for g in cell["zone_grid"]], [2, 4, 6])
                for grid in cell["zone_grid"]:
                    points = [(row["K"], row["delta"]) for row in grid["rows"]]
                    self.assertEqual(points, [(K, d) for K in [0, 2, 3, 4, 5] for d in deltas])
                    self.assertEqual(grid["baseline"]["K"], 0)
                    k0 = [row for row in grid["rows"] if row["K"] == 0]
                    for row in k0:
                        self.assertEqual(row["turnover_mean"], grid["baseline"]["turnover_mean"])
            self.assertEqual(len(ca._ordered_pairs([1, 2, 3])), 6)
            self.assertEqual(len(ca._ordered_pairs([1, 2, 3, 4, 5])), 20)
        finally:
            lots.close()


# --------------------------------------------------------------------------
# S6 - recommendations
# --------------------------------------------------------------------------


class RunCountTests(unittest.TestCase):
    CANDIDATES = [30, 60, 100, 150]

    def rule(self, S, M=None):
        return ca.run_count_rule(S, M or S, self.CANDIDATES, 0.5)

    def test_rule_examples(self):
        S = {30: 3.1, 60: 2.4, 100: 2.0, 150: 1.8}
        self.assertEqual(self.rule(S), {"flattened": True, "k_min": 100})
        self.assertEqual(self.rule({**S, 60: 2.3}), {"flattened": True, "k_min": 60})
        M = {**S, 60: 3.0, 150: 2.0}
        self.assertEqual(ca.run_count_rule({**S, 60: 2.3}, M, self.CANDIDATES, 0.5), {"flattened": True, "k_min": 100})
        not_flat = self.rule({**S, 100: 2.4})
        self.assertEqual(not_flat, {"flattened": False, "k_min": None})
        self.assertEqual(ca.combine_run_count([self.rule(S), not_flat]), ca.DATA_INSUFFICIENT)
        self.assertEqual(ca.combine_run_count([{"flattened": True, "k_min": 60}, {"flattened": True, "k_min": 100}]), 100)

    def test_leave_one_seed_out(self):
        self.assertEqual(ca.final_run_count(60, {"1": 60, "2": 100, "3": 30}), 100)
        self.assertEqual(ca.final_run_count(60, {"1": 60, "2": ca.DATA_INSUFFICIENT}), ca.DATA_INSUFFICIENT)
        self.assertEqual(ca.final_run_count(ca.DATA_INSUFFICIENT, {"1": 30}), ca.DATA_INSUFFICIENT)
        self.assertEqual(ca.final_run_count(60, {}), 60)
        lots = LotSet()
        try:
            report = lots.report_json()
            rec = report["recommendation"]["run_count"]
            self.assertEqual(sorted(rec["leave_one_seed_out"]), ["1", "2", "3"])
            for cell in rec["per_cell"].values():
                self.assertEqual(sorted(cell["leave_one_seed_out"]), ["1", "2", "3"])
                self.assertEqual([p["k"] for p in cell["full"]["curve"]], [2, 4, 6])
            self.assertEqual(rec["value"], ca.final_run_count(rec["full_value"], rec["leave_one_seed_out"]))
        finally:
            lots.close()

    def test_fixture_candidates(self):
        candidates = [4, 8, 12]
        self.assertEqual(ca.run_count_rule({4: 5.0, 8: 2.0, 12: 1.8}, {4: 5.0, 8: 2.0, 12: 1.8}, candidates, 0.5), {"flattened": True, "k_min": 8})
        self.assertEqual(ca.run_count_rule({4: 5.0, 8: 2.6, 12: 2.0}, {4: 5.0, 8: 2.6, 12: 2.0}, candidates, 0.5)["flattened"], False)


class TargetTests(unittest.TestCase):
    def verdict(self, values, met=lambda v: v >= 0.75):
        return ca.reachability_verdict({str(i): v for i, v in enumerate(values)}, met)["verdict"]

    def test_reachability_examples(self):
        self.assertEqual(self.verdict([0.14, 0.13, 0.15, 0.12, 0.14]), "unreachable")
        self.assertEqual(self.verdict([0.80, 0.90, 0.76, 0.75, 0.99]), "reachable")
        self.assertEqual(self.verdict([0.74, 0.80, 0.90, 0.76, 0.99]), "mixed")
        self.assertEqual(self.verdict([0.02, 0.01], met=lambda v: v <= 0.02), "reachable")
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

        def cells(cv_second):
            return {
                "a/stocks": {"target_jaccard": {"verdict": "reachable"}, "target_cv": {"verdict": "reachable"}},
                "a/fiis": {"target_jaccard": {"verdict": "reachable"}, "target_cv": {"verdict": cv_second}},
            }

        mixed = ca.recommend_targets(protocol, cells("mixed"), 60)
        self.assertEqual(mixed["target_jaccard"]["verdict"], "reachable")
        self.assertEqual(mixed["target_cv"]["verdict"], "mixed")
        self.assertEqual((mixed["action"], mixed["fixed_run_count"]), ("use_fixed_run_count", 60))
        insufficient = ca.recommend_targets(protocol, cells("mixed"), ca.DATA_INSUFFICIENT)
        self.assertEqual(insufficient["fixed_run_count"], 150)
        keep = ca.recommend_targets(protocol, cells("reachable"), 60)
        self.assertEqual((keep["action"], keep["fixed_run_count"]), ("keep", None))
        self.assertEqual(ca.combine_verdicts(["unreachable", "unreachable"]), "unreachable")
        self.assertEqual(ca.combine_verdicts(["unreachable", "reachable"]), "mixed")


class ZoneRecommendationTests(unittest.TestCase):
    def test_quantile_examples(self):
        ranks = [1] * 246 + [17] + [18] * 13
        self.assertEqual(ca.k_star(ranks, 0.95, 13, 30), 17)
        self.assertEqual(ca.k_star(ranks, 0.95, 13, 30) - 13, 4)
        self.assertIsNone(ca.k_star([1] * 246 + [math.inf] * 14, 0.95, 13, 30))
        self.assertEqual(ca.k_star([1] * 247 + [math.inf] * 13, 0.95, 13, 30), 13)
        self.assertIsNone(ca.k_star([1] * 246 + [31] * 14, 0.95, 13, 30))
        self.assertEqual(ca.inverted_cdf(list(range(1, 11)), 0.1), 1)
        two, twenty = Fraction(2, 100), Fraction(20, 100)
        self.assertEqual(ca.delta_star([two] * 95 + [twenty] * 5, 0.95, 0.01), 0.02)
        self.assertEqual(ca.delta_star([two] * 94 + [twenty] * 6, 0.95, 0.01), 0.2)
        self.assertEqual(ca.round_up(Fraction("0.0333"), 0.01), 0.04)
        self.assertEqual(ca.round_up(Fraction("0.04"), 0.01), 0.04)

    def test_delta_exact(self):
        cases = (((8, 5), 0.02), ((21, 0), 0.14))
        for (a, b), expected in cases:
            series = {1: Stub(1, ["X"], ["X"], {"X": a}), 2: Stub(1, ["X"], ["X"], {"X": b})}
            differences = ca.frequency_differences(series, [1, 2], 150)
            self.assertEqual(differences, [Fraction(abs(a - b), 150)])
            self.assertEqual(ca.delta_star(differences, 0.95, 0.01), expected)
        self.assertEqual(math.ceil((8 / 150 - 5 / 150) / 0.01) * 0.01, 0.03)

    def test_recommendation_aggregation(self):
        protocol = small_protocol()
        protocol["profiles"]["q"] = {"record": {"anchor": "arrojado"}, "expected_n_assets": {"stocks": 3, "fiis": 3}}

        def cell(n, consensuses, rankings, counts):
            return {s: Stub(n, consensuses[s], rankings[s], counts[s]) for s in (1, 2, 3)}

        stable = ["A1", "B1"]
        p_series = cell(2, {s: stable for s in (1, 2, 3)}, {s: stable + ["C1"] for s in (1, 2, 3)}, {s: {"A1": 6, "B1": 5, "C1": 1} for s in (1, 2, 3)})
        q_series = cell(
            3,
            {1: ["A1", "B1", "C1"], 2: ["A1", "B1", "D1"], 3: ["A1", "B1", "C1"]},
            {1: ["A1", "B1", "C1", "D1"], 2: ["A1", "B1", "D1", "C1"], 3: ["A1", "B1", "C1", "D1"]},
            {1: {"A1": 6, "B1": 6, "C1": 4, "D1": 2}, 2: {"A1": 6, "B1": 6, "D1": 4, "C1": 3}, 3: {"A1": 6, "B1": 6, "C1": 4, "D1": 2}},
        )
        cells = {"p/stocks": (p_series, 2), "q/stocks": (q_series, 3), "p/fiis": (p_series, 2), "q/fiis": (q_series, 3)}
        zone = ca.recommend_zone(protocol, cells, 6)
        stocks = zone["stocks"]
        self.assertEqual(stocks["at_prefix"], 6)
        self.assertEqual(stocks["per_cell"]["p/stocks"]["k_star"], 2)
        self.assertEqual(stocks["per_cell"]["q/stocks"]["k_star"], 4)
        self.assertEqual(stocks["k_offset"], 1)
        self.assertEqual(stocks["delta"], max(item["delta"] for item in stocks["per_cell"].values()))
        self.assertEqual(stocks["per_cell"]["q/stocks"]["at_recommended"]["K"], 4)
        self.assertEqual(stocks["per_cell"]["p/stocks"]["at_recommended"]["K"], 3)
        self.assertEqual(ca.recommend_zone(protocol, cells, ca.DATA_INSUFFICIENT)["fiis"]["at_prefix"], 6)
        unranked = cell(2, {s: ["Z1", "Y1"] for s in (1, 2, 3)}, {s: ["A1"] for s in (1, 2, 3)}, {s: {"A1": 1} for s in (1, 2, 3)})
        cells["p/fiis"] = (unranked, 2)
        fiis = ca.recommend_zone(protocol, cells, 6)["fiis"]
        self.assertIsNone(fiis["per_cell"]["p/fiis"]["k_star"])
        self.assertIsNone(fiis["k_offset"])
        self.assertIsNone(fiis["delta"])
        self.assertIsNone(fiis["per_cell"]["q/fiis"]["at_recommended"])


class RunCountBoundaryTests(unittest.TestCase):
    """C38: the run_count rule's inclusive boundaries and their 1e-12 tolerance."""

    CANDIDATES = [30, 60, 100, 150]

    def rule(self, S, M=None):
        return ca.run_count_rule(S, M or S, self.CANDIDATES, 0.5)

    def test_boundaries(self):
        self.assertEqual(self.rule({30: 3.1, 60: 2.6, 100: 2.5, 150: 2.0}), {"flattened": True, "k_min": 100})
        self.assertEqual(self.rule({30: 3.1, 60: 2.5, 100: 2.2, 150: 2.0}), {"flattened": True, "k_min": 60})
        S = {30: 3.1, 60: 2.0, 100: 2.0, 150: 2.0}
        M = {30: 3.1, 60: 2.5, 100: 2.2, 150: 2.0}
        self.assertEqual(self.rule(S, M), {"flattened": True, "k_min": 60})
        self.assertEqual(1.1 - 0.6, 0.5000000000000001)
        self.assertEqual(self.rule({30: 3.1, 60: 2.4, 100: 1.1, 150: 0.6}), {"flattened": True, "k_min": 100})
        self.assertGreater(0.68, 0.18 + 0.5)
        self.assertEqual(self.rule({30: 3.1, 60: 0.68, 100: 0.3, 150: 0.18}), {"flattened": True, "k_min": 60})


class RunCountWiringTests(unittest.TestCase):
    """C39: S and M are the mean and median of criterion 21's swaps, in full and per left-out seed."""

    def test_curves_from_seed_turnover(self):
        protocol = small_protocol(seeds=[1, 2, 3, 4])
        lots = LotSet(protocol=protocol)
        try:
            report = lots.report_json()
        finally:
            lots.close()
        rec = report["recommendation"]["run_count"]
        self.assertEqual(sorted(rec["leave_one_seed_out"]), ["1", "2", "3", "4"])
        mean_differs = median_differs = subset_differs = False
        for cell_id, cell in report["cells"].items():
            turnover = {item["k"]: item["pairs"] for item in cell["seed_turnover"]}
            full = rec["per_cell"][cell_id]["full"]["curve"]
            self.assertEqual([point["k"] for point in full], [2, 4, 6])
            for point in full:
                swaps = [pair["swapped"] for pair in turnover[point["k"]]]
                self.assertEqual(len(swaps), 6)
                self.assertEqual(point["S"], ca.mean(swaps), cell_id)
                self.assertEqual(point["M"], ca.median(swaps), cell_id)
                mean_differs |= point["S"] != point["M"]
            for seed in (1, 2, 3, 4):
                curve = rec["per_cell"][cell_id]["leave_one_seed_out"][str(seed)]["curve"]
                for point, full_point in zip(curve, full):
                    swaps = [pair["swapped"] for pair in turnover[point["k"]] if seed not in pair["seeds"]]
                    self.assertEqual(len(swaps), 3)
                    self.assertEqual(point["S"], ca.mean(swaps), (cell_id, seed))
                    self.assertEqual(point["M"], ca.median(swaps), (cell_id, seed))
                    median_differs |= point["M"] != point["S"]
                    subset_differs |= point["S"] != full_point["S"]
        # The data must tell the mean from the median and a subset from all seeds.
        self.assertTrue(mean_differs and median_differs and subset_differs)


class PrefixStub(Stub):
    """A Stub whose consensus, ranking and counts depend on k."""

    def __init__(self, n, by_k):
        self.n = n
        self.by_k = {k: Stub(n, *data) for k, data in by_k.items()}

    def consensus(self, k):
        return self.by_k[k].consensus(k)

    def ranking(self, k):
        return self.by_k[k].ranking(k)

    def counts(self, k):
        return self.by_k[k].counts(k)


class ZoneInputTests(unittest.TestCase):
    """C40-C42: the K* and δ* input sets, K*'s window boundary and the zone's prefix."""

    def series(self):
        consensus = {1: ["A1", "B1"], 2: ["A1", "C1"], 3: ["B1", "D1"]}
        ranking = {1: ["A1", "B1", "C1", "D1"], 2: ["C1", "A1", "D1", "B1"], 3: ["D1", "B1", "A1"]}
        counts = {
            1: {"A1": 8, "B1": 6, "C1": 3, "D1": 1},
            2: {"A1": 7, "C1": 5, "D1": 2, "B1": 1},
            3: {"D1": 9, "B1": 4, "A1": 2},
        }
        return {s: Stub(2, consensus[s], ranking[s], counts[s]) for s in (1, 2, 3)}

    def test_incumbent_ranks_over_ordered_pairs(self):
        ranks = ca.incumbent_ranks(self.series(), [1, 2, 3], 10)
        self.assertEqual(len(ranks), 6 * 2)
        self.assertEqual(sorted(ranks), [1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, math.inf])

    def test_frequency_differences_over_union(self):
        values = ca.frequency_differences(self.series(), [1, 2, 3], 10)
        tenths = [Fraction(v, 10) for v in (1, 5, 2, 6, 2, 8, 5, 3, 5, 7)]
        self.assertEqual(values, tenths)
        self.assertEqual(len(values), 3 + 3 + 4)

    def test_k_star_window_inclusive(self):
        self.assertEqual(ca.k_star([1] * 246 + [30] * 14, 0.95, 13, 30), 30)
        self.assertIsNone(ca.k_star([1] * 246 + [31] * 14, 0.95, 13, 30))

    def test_zone_at_run_count_prefix(self):
        protocol = small_protocol()
        stable = (["A1", "B1"], ["A1", "B1", "C1"])

        def seed(s):
            at_4 = (*stable, {"A1": 4, "B1": 2 if s == 3 else 3, "C1": 1})
            at_6 = (["A1", "B1"], ["Z1"], {"A1": 6, "B1": 6})
            return PrefixStub(2, {4: at_4, 6: at_6})

        series = {s: seed(s) for s in (1, 2, 3)}
        cells = {"p/stocks": (series, 2), "p/fiis": (series, 2)}
        zone = ca.recommend_zone(protocol, cells, 4)
        for selector in ca.SELECTORS:
            self.assertEqual(zone[selector]["at_prefix"], 4)
            self.assertEqual(zone[selector]["per_cell"][f"p/{selector}"]["k_star"], 2)
            self.assertEqual(zone[selector]["k_offset"], 0)
            self.assertEqual(zone[selector]["delta"], 0.25)
        self.assertIsNone(ca.recommend_zone(protocol, cells, 6)["stocks"]["k_offset"])


class ReachabilityReportTests(unittest.TestCase):
    """C45: the reachability verdicts on a report at the registered targets' boundaries."""

    def lots_with(self, stock_fitness):
        protocol = small_protocol()
        protocol["production_stop_rule"]["target_jaccard"] = 1.0
        protocol["production_stop_rule"]["target_cv"] = 0.0
        lots = LotSet(protocol=protocol)
        for lot in lots.lots.values():
            lot["selectors"]["stocks"] = selector_lot("stocks", [["POMO4", "VALE3"]] * 6, 2, fitness=stock_fitness)
            lot["selectors"]["fiis"] = selector_lot("fiis", [["KNCR11", "KNRI11"]] * 6, 2, fitness=[1.0] * 6)
        lots.write()
        return lots

    def test_identical_runs_at_the_targets(self):
        lots = self.lots_with([1.0] * 6)
        try:
            report = lots.report_json()
        finally:
            lots.close()
        for cell_id, cell in report["cells"].items():
            reach = cell["reachability"]
            self.assertEqual(set(reach["target_jaccard"]["by_seed"].values()), {1.0}, cell_id)
            self.assertEqual(set(reach["target_cv"]["by_seed"].values()), {0.0}, cell_id)
            self.assertEqual(reach["target_jaccard"]["verdict"], "reachable", cell_id)
            self.assertEqual(reach["target_cv"]["verdict"], "reachable", cell_id)
        self.assertEqual(report["recommendation"]["adaptive_targets"]["action"], "keep")

    def test_fitness_spread_fails_only_cv(self):
        lots = self.lots_with([1.0, 1.1, 1.2, 1.0, 1.1, 1.2])
        try:
            report = lots.report_json()
        finally:
            lots.close()
        reach = report["cells"]["p/stocks"]["reachability"]
        self.assertEqual(reach["target_jaccard"]["verdict"], "reachable")
        self.assertEqual(reach["target_cv"]["verdict"], "unreachable")
        self.assertEqual(report["cells"]["p/fiis"]["reachability"]["target_cv"]["verdict"], "reachable")
        targets = report["recommendation"]["adaptive_targets"]
        self.assertEqual((targets["target_jaccard"]["verdict"], targets["target_cv"]["verdict"]), ("reachable", "mixed"))
        self.assertEqual(targets["action"], "use_fixed_run_count")


class NaNScoreTests(unittest.TestCase):
    """C48 and C50: a null FII score is refused; door 5's NaN tie example and its limitation."""

    def test_null_fii_score_exits_4(self):
        lots = LotSet()
        try:
            data = lots.lots[("p", 2)]["selectors"]["fiis"]
            ticker = sorted(data["candidates"])[0]
            data["candidates"][ticker]["score"] = None
            lots.write()
            code, err = lots.report()
            self.assertEqual(code, ca.EXIT_INTEGRITY, err)
            self.assertIn("p:2 fiis", err)
            self.assertIn(f"FII {ticker} has a null score", err)
            self.assertFalse(lots.out_dir.exists())
        finally:
            lots.close()

    def test_null_stock_score_accepted(self):
        lots = LotSet()
        try:
            for lot in lots.lots.values():
                data = lot["selectors"]["stocks"]
                for info in data["candidates"].values():
                    info["score"] = None
                data["selected_tickers"] = ca.production_consensus(data["runs"], data["candidates"], "stocks", 2)
            lots.write()
            code, err = lots.report()
            self.assertEqual(code, 0, err)
        finally:
            lots.close()

    def test_nan_breaks_the_whole_count_tie(self):
        counts = Counter({"AAAA3": 1, "BBBB3": 1, "CCCC3": 1})
        scores = {"AAAA3": 0.5, "BBBB3": math.nan, "CCCC3": 0.7}
        self.assertEqual(ca.extended_ranking(counts, scores, "stocks"), ["AAAA3", "BBBB3", "CCCC3"])
        del counts["BBBB3"]
        self.assertEqual(ca.extended_ranking(counts, scores, "stocks"), ["CCCC3", "AAAA3"])

    def test_limitation_in_markdown(self):
        lots = LotSet()
        try:
            lots.report_json()
            text = (lots.out_dir / "multirun-convergence.md").read_text(encoding="utf-8")
        finally:
            lots.close()
        limitations = text.split("## Limitações", 1)[1]
        for needle in ("SCORE NaN", "porta 5", "além de n", "rank gap", "K*", "δ = 0"):
            self.assertIn(needle, limitations)


class ProductionMatchTests(unittest.TestCase):
    def test_match_reported(self):
        lots = LotSet()
        try:
            report = lots.report_json()
            self.assertEqual(report["production_run_match"], {"stocks": True, "fiis": True})
            lots.protocol["production_run"]["fiis"] = ["ZZZZ11", "YYYY11"]
            lots.protocol_path.write_text(json.dumps(lots.protocol), encoding="utf-8")
            lots.sha = ca.sha256_bytes(lots.protocol_path.read_bytes())
            for lot in lots.lots.values():
                lot["protocol_sha256"] = lots.sha
            lots.write()
            code, err = lots.report()
            self.assertEqual(code, 0, err)
            report = json.loads((lots.out_dir / "multirun-convergence.json").read_text(encoding="utf-8"))
            self.assertEqual(report["production_run_match"], {"stocks": True, "fiis": False})
        finally:
            lots.close()


# --------------------------------------------------------------------------
# Report files
# --------------------------------------------------------------------------


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.lots = LotSet()

    def tearDown(self):
        self.lots.close()

    def test_provenance_copied(self):
        report = self.lots.report_json()
        self.assertEqual(report["protocol_commit"], "c0ffee")
        self.assertEqual(report["protocol_committed_at"], "2026-11-03T10:00:00-03:00")
        self.assertEqual(report["protocol_sha256"], self.lots.sha)
        self.lots.lots[("p", 3)]["protocol_commit"] = "other"
        self.lots.write()
        code, err = self.lots.report()
        self.assertEqual(code, ca.EXIT_INTEGRITY, err)

    def test_markdown_sections(self):
        self.lots.report_json()
        text = (self.lots.out_dir / "multirun-convergence.md").read_text(encoding="utf-8")
        for needle in (
            "# Convergência do multi-run (medido)",
            "### run_count:",
            "- sem a seed 1:",
            "### Alvos do critério adaptativo:",
            "### Zona de proteção (K, δ)",
            "## Execução de produção de 2026-11-02",
            "## Célula p/stocks (n = 2)",
            "## Célula p/fiis (n = 2)",
            "### Curvas (média entre seeds, [mín, máx])",
            "### Frequências a 6 execuções (top 5 por seed)",
            "### Giro entre seeds e dentro da seed (trocados)",
            "dentro da seed média",
            "### Folga de ranking",
            "### Zona simulada a ",
            "| 0 (consenso puro) |",
            "Ponto recomendado:",
            "## Custo (medido)",
            "CPU-horas",
            "## Limitações",
            "Um único snapshot",
            "não são independentes",
            "AD-022",
            "fora da amostra",
            "Outras configurações do GA",
            "0,697275 para `p`",
            "metade conservador-moderado",
            "Python 3.14.4, numpy 2.3.4, pandas 2.3.3",
            "aritmética do relatório é stdlib",
        ):
            self.assertIn(needle, text, needle)
        stocks = text.index("## Célula p/stocks")
        fiis = text.index("## Célula p/fiis")
        self.assertIn("### Giro entre seeds", text[stocks:fiis])
        self.assertIn("### Giro entre seeds", text[fiis:])

    def test_smoke_labels(self):
        smoke = LotSet(mode="smoke")
        try:
            report = smoke.report_json()
            self.assertEqual(report["mode"], "smoke")
            text = (smoke.out_dir / "multirun-convergence.md").read_text(encoding="utf-8")
            self.assertTrue(text.startswith("# Convergência do multi-run (SMOKE"))
            cost = json.loads((smoke.out_dir / "multirun-convergence-cost.json").read_text(encoding="utf-8"))
            self.assertEqual((cost["mode"], cost["label"]), ("smoke", "smoke"))
        finally:
            smoke.close()

    def test_serialization_is_stdlib(self):
        import numpy

        self.lots.report_json()
        baseline = {name: (self.lots.out_dir / name).read_bytes() for name in ca.REPORT_FILES}
        for name in ("multirun-convergence.json", "multirun-convergence-cost.json"):
            text = baseline[name].decode("utf-8")
            self.assertEqual(text, ca.dumps(json.loads(text)))
        self.assertNotIn("created_at", baseline["multirun-convergence.json"].decode("utf-8"))
        original_mean = numpy.mean
        with patch.object(numpy, "mean", side_effect=lambda *a, **k: original_mean(*a, **k) + 1e-14):
            code, err = self.lots.report()
        self.assertEqual(code, 0, err)
        for name in ca.REPORT_FILES:
            self.assertEqual((self.lots.out_dir / name).read_bytes(), baseline[name], name)


class CostTests(unittest.TestCase):
    def test_cost_report(self):
        lots = LotSet()
        try:
            lots.report_json()
            cost = json.loads((lots.out_dir / "multirun-convergence-cost.json").read_text(encoding="utf-8"))
            self.assertEqual(cost["label"], "measured")
            self.assertEqual([row["lot"] for row in cost["lots"]], ["p:1", "p:2", "p:3"])
            for row in cost["lots"]:
                source = cost_file("p", row["seed"])
                self.assertEqual(row["stocks"], source["stages"]["stocks"])
                self.assertEqual(row["fiis"], source["stages"]["fiis"])
                self.assertEqual(row["total"], {"wall_s": source["wall_s"], "cpu_s": source["cpu_s"]})
                self.assertEqual(row["machine_fingerprint"], "machine-1")
            self.assertEqual(cost["cpu_hours_total"], math.fsum(3500.0 + s for s in (1, 2, 3)) / 3600)
            self.assertEqual(cost["experiment"]["experiment_wall_s"], 7300.5)
            self.assertEqual(cost["experiment"]["jobs"], 5)
            self.assertEqual(cost["machines"][0]["fingerprint"], "machine-1")
            text = (lots.out_dir / "multirun-convergence.md").read_text(encoding="utf-8")
            self.assertIn("| p:1 | 1001,0 | 991,0 | 2000,0 | 1990,0 | 3601,0 | 3501,0 |", text)
            self.assertIn("7300,5 s com `jobs` 5", text)
            ca.cost_path(lots.lots_dir, "p", 2).unlink()
            code, err = lots.report()
            self.assertEqual(code, ca.EXIT_INTEGRITY)
            self.assertIn("p:2", err)
        finally:
            lots.close()


# --------------------------------------------------------------------------
# S7 - part (b): the committed measured report
# --------------------------------------------------------------------------


COMMITTED_DIR = PROJECT_ROOT / "docs" / "validation"
COMMITTED_LOTS = COMMITTED_DIR / "multirun-convergence" / "lots"


def _git(*args):
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True)


class CommittedReportTests(unittest.TestCase):
    def test_committed_report_is_measured(self):
        report = json.loads((COMMITTED_DIR / "multirun-convergence.json").read_text(encoding="utf-8"))
        protocol, sha = ca.load_protocol(PROTOCOL_PATH)
        self.assertEqual(report["mode"], "measured")
        self.assertEqual(report["protocol_sha256"], sha)
        expected = {ca.lot_id(p, s) for p in protocol["profiles"] for s in protocol["seeds"]}
        self.assertEqual(set(report["lots"]), expected)
        self.assertEqual(len(expected), 10)
        for item in report["lots"].values():
            self.assertEqual(item["runs"], {"stocks": 150, "fiis": 150})
        commit = report["protocol_commit"]
        shown = _git("show", f"{commit}:docs/validation/multirun-convergence-protocol.json")
        self.assertEqual(shown.returncode, 0)
        self.assertEqual(shown.stdout, PROTOCOL_PATH.read_bytes())
        for item in report["lots"].values():
            self.assertEqual(_git("merge-base", "--is-ancestor", commit, item["commit"]).returncode, 0)

    def test_report_regenerates_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = ca.run_report(PROTOCOL_PATH, COMMITTED_LOTS, Path(tmp))
            for path, text in files.items():
                self.assertEqual(text.encode("utf-8"), (COMMITTED_DIR / path.name).read_bytes(), path.name)


if __name__ == "__main__":
    unittest.main()
