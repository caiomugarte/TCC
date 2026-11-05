"""Tests for scripts/protection_zone_turnover.py (issue #46, PZ-32..PZ-34).

The fixture registry is built from the committed ``data/allocation`` series and the
``data/raw/status_invest_*.csv`` exports with ``snapshot_registry.build_manifest``,
then re-dated to a November cutoff. The selectors are fakes that apply the real
core zone to fixed frequencies; no GA runs.
"""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "py"))

import protection_zone_turnover as pzt  # noqa: E402
from core.protection_zone import select_with_fallback  # noqa: E402

PROTOCOL = json.loads(pzt.DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
OCTOBER = PROTOCOL["production_run"]
CODE = {"commit": "0123456789abcdef0123456789abcdef01234567", "dirty": False}


def build_registry(root, *, manifest_id, cutoff, state="published"):
    """A registry with one manifest whose cutoff is moved to ``cutoff``."""

    from snapshot_manifest import SnapshotManifest, manifest_canonical_sha256
    from snapshot_registry import build_manifest

    registry = Path(root) / "registry"
    if not registry.exists():
        (registry / "data").mkdir(parents=True)
        shutil.copytree(
            PROJECT_ROOT / "data" / "allocation",
            registry / "data" / "snap",
            ignore=shutil.ignore_patterns("caio_stocks.csv", "README.md"),
        )
        shutil.copy(PROJECT_ROOT / "data" / "raw" / "status_invest_fundamentals.csv", registry / "stocks.csv")
        shutil.copy(PROJECT_ROOT / "data" / "raw" / "status_invest_fii.csv", registry / "fiis.csv")
    payload = build_manifest(
        registry_root=registry,
        manifest_id=manifest_id,
        created_at="2026-11-03T00:00:00+00:00",
        snapshot_dir=registry / "data" / "snap",
        stock_universe=registry / "stocks.csv",
        fii_universe=registry / "fiis.csv",
    )
    if cutoff > payload["cutoff_date"]:
        payload["common_dates"].append(cutoff)
    else:
        payload["common_dates"] = [day for day in payload["common_dates"] if day <= cutoff]
    payload["cutoff_date"] = cutoff
    payload["state"] = state
    manifest = SnapshotManifest.from_dict(payload, base_dir=registry)
    payload["manifest_sha256"] = manifest_canonical_sha256(manifest)
    (registry / f"{manifest_id}.json").write_text(json.dumps(payload), encoding="utf-8")
    return registry


def frequencies(prefix, incumbents):
    """Fixed November frequencies: four new tickers lead, then the 13 October tickers."""

    result = dict(zip((f"{prefix}NEW{index}" for index in range(4)), (1.0, 0.97, 0.90, 0.89)))
    for index, ticker in enumerate(incumbents):
        result[ticker] = round(0.88 - 0.01 * index, 2)
    return result


def fake_engines(calls):
    def selector(key, prefix):
        def engine(**kwargs):
            config = kwargs.get("config", kwargs.get("selection_config"))
            calls[key] = kwargs
            n_assets = int(config["n_assets"])
            freq = frequencies(prefix, OCTOBER[key])
            selection, record = select_with_fallback(
                kwargs.get("incumbents", []),
                freq,
                {ticker: 0.5 for ticker in freq},
                n_assets,
                config["protection_zone"],
                eligible=set(freq),
                dedup_base_company=False,
                hhi_of=lambda _tickers: 0.0,
                hhi_max=None,
            )
            tickers = list(selection.tickers)
            return {
                "selected_tickers": tickers,
                "sleeve_weights": {ticker: 1 / len(tickers) for ticker in tickers},
                "protection_zone": record,
            }

        return engine

    return {
        "snapshot_loader": lambda _manifest: SimpleNamespace(rows=(), metadata={}),
        "allocation_engine": lambda *_args, **_kwargs: {
            "weights": {
                "brazilian_stocks": 0.25,
                "fiis": 0.25,
                "international_equity": 0.25,
                "fixed_income": 0.0,
                "crypto": 0.25,
            }
        },
        "stock_engine": selector("stocks", "S"),
        "fii_engine": selector("fiis", "F"),
    }


class ProtectionZoneTurnoverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.registry = build_registry(cls.temp.name, manifest_id="nov-fixture", cutoff="2026-11-03")
        build_registry(cls.temp.name, manifest_id="oct-fixture", cutoff="2026-10-28")
        build_registry(cls.temp.name, manifest_id="draft-fixture", cutoff="2026-11-03", state="draft")
        build_registry(cls.temp.name, manifest_id=pzt.OCTOBER_MANIFEST_ID, cutoff="2026-11-03")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.out = Path(tempfile.mkdtemp(dir=self.temp.name))

    def run_main(self, manifest_id, calls=None):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = pzt.main(
                [
                    "measure",
                    "--registry", str(self.registry),
                    "--manifest-id", manifest_id,
                    "--out-dir", str(self.out),
                ],
                engines=fake_engines({} if calls is None else calls),
                code=CODE,
            )
        return code, stderr.getvalue()

    def files(self):
        return sorted(path.name for path in self.out.iterdir())

    def test_measure_writes_both_reports_with_the_turnover_per_class(self):
        calls = {}
        code, stderr = self.run_main("nov-fixture", calls)
        self.assertEqual(code, pzt.EXIT_OK, stderr)
        self.assertEqual(self.files(), [pzt.JSON_NAME, pzt.MD_NAME])

        report = json.loads((self.out / pzt.JSON_NAME).read_text(encoding="utf-8"))
        manifest = json.loads((self.registry / "nov-fixture.json").read_text(encoding="utf-8"))
        self.assertEqual(report["seed"], 46641103)
        self.assertEqual(report["profile_id"], "premium-2026-11-02")
        self.assertEqual(report["policy_version"], "premium-policy-v3")
        self.assertEqual(report["october"]["run_id"], "9e22cc16-cfda-495e-a8c9-c0d202ef32b6")
        self.assertEqual(report["november"]["manifest_id"], "nov-fixture")
        self.assertEqual(report["november"]["manifest_sha256"], manifest["manifest_sha256"])
        self.assertEqual(report["november"]["cutoff_date"], "2026-11-03")
        self.assertEqual(report["code"]["commit"], CODE["commit"])

        # 13 + 13 profile, delta 0.11. Pure consensus: the 4 newcomers enter.
        # Stocks (K = 19): every October ticker is protected; NEW0, NEW1 and NEW2
        # lead the weakest protected ones (0.76, 0.77, 0.78) by >= 0.11, NEW3
        # (0.89 vs 0.79) does not: 3 entries. FIIs (K = 16): the October ticker at
        # rank 17 (0.76) is outside K and NEW0 fills its seat; NEW1 and NEW2 replace
        # 0.77 and 0.78, NEW3 does not beat 0.79: 3 entries.
        for key, k_value in (("stocks", 19), ("fiis", 16)):
            item = report["classes"][key]
            self.assertEqual(item["status"], "applied")
            self.assertEqual((item["K"], item["k_offset"], item["delta"]), (k_value, k_value - 13, 0.11))
            self.assertEqual(item["incumbents"], sorted(OCTOBER[key]))
            self.assertEqual(item["consensus_turnover"], 4)
            self.assertEqual(item["turnover"], 3)
            self.assertEqual(item["exits"], 3)
            self.assertEqual(item["turnover"], len(item["entered"]))
            self.assertEqual(item["exits"], len(item["departures"]))
            self.assertEqual(sorted(item["kept"] + item["entered"]), item["zone_tickers"])
            self.assertEqual(
                sorted(item["kept"] + list(item["departures"])), sorted(OCTOBER[key])
            )
            for swap in item["replaced"]:
                self.assertEqual(item["departures"][swap["out"]], f"saiu: substituído por {swap['in']}")
            self.assertEqual(calls[key]["incumbents"], list(OCTOBER[key]))
            self.assertEqual(calls[key]["seed"], 46641103)
        stock_swaps = report["classes"]["stocks"]["replaced"]
        self.assertEqual(
            stock_swaps,
            [
                {"out": OCTOBER["stocks"][12], "in": "SNEW0"},
                {"out": OCTOBER["stocks"][11], "in": "SNEW1"},
                {"out": OCTOBER["stocks"][10], "in": "SNEW2"},
            ],
        )
        self.assertEqual(
            report["classes"]["fiis"]["departures"][OCTOBER["fiis"][12]], "saiu: fora do top K"
        )

        markdown = (self.out / pzt.MD_NAME).read_text(encoding="utf-8")
        self.assertIn("nov-fixture", markdown)
        self.assertIn(manifest["manifest_sha256"], markdown)
        self.assertIn(CODE["commit"], markdown)
        stocks = report["classes"]["stocks"]
        self.assertIn(
            f"| stocks | 19 | 0.11 | {stocks['consensus_turnover']} | {stocks['turnover']} | "
            f"{stocks['exits']} | não |",
            markdown,
        )

    def test_report_records_the_october_confounds(self):
        code, stderr = self.run_main("nov-fixture")
        self.assertEqual(code, pzt.EXIT_OK, stderr)
        report = json.loads((self.out / pzt.JSON_NAME).read_text(encoding="utf-8"))
        limitations = " ".join(report["limitations"])
        self.assertIn("9e22cc16-cfda-495e-a8c9-c0d202ef32b6", limitations)
        self.assertIn("#52", limitations)
        self.assertIn("derive_premium_seed", limitations)
        markdown = (self.out / pzt.MD_NAME).read_text(encoding="utf-8")
        for text in report["limitations"]:
            self.assertIn(text, markdown)

    def test_rerun_is_byte_identical(self):
        code, stderr = self.run_main("nov-fixture")
        self.assertEqual(code, pzt.EXIT_OK, stderr)
        first = {name: (self.out / name).read_bytes() for name in self.files()}
        code, stderr = self.run_main("nov-fixture")
        self.assertEqual(code, pzt.EXIT_OK, stderr)
        second = {name: (self.out / name).read_bytes() for name in self.files()}
        self.assertEqual(first, second)
        self.assertNotIn(str(self.registry).encode(), first[pzt.JSON_NAME])
        self.assertNotIn(str(self.registry).encode(), first[pzt.MD_NAME])

    def test_refused_manifests_exit_3_and_write_nothing(self):
        cases = {
            "missing": "absent-fixture",
            "cutoff before 2026-11-01": "oct-fixture",
            "October manifest id": pzt.OCTOBER_MANIFEST_ID,
            "not published": "draft-fixture",
        }
        for label, manifest_id in cases.items():
            with self.subTest(case=label):
                calls = {}
                code, stderr = self.run_main(manifest_id, calls)
                self.assertEqual(code, pzt.EXIT_SNAPSHOT)
                self.assertEqual(self.files(), [])
                self.assertEqual(calls, {})
                self.assertIn("manifest", stderr)

    def test_manifest_failing_validation_exits_3(self):
        payload = json.loads((self.registry / "nov-fixture.json").read_text(encoding="utf-8"))
        payload["manifest_id"] = "tampered-fixture"
        (self.registry / "tampered-fixture.json").write_text(json.dumps(payload), encoding="utf-8")
        code, stderr = self.run_main("tampered-fixture")
        self.assertEqual(code, pzt.EXIT_SNAPSHOT)
        self.assertIn("failed validation", stderr)
        self.assertEqual(self.files(), [])

    def test_selector_without_the_zone_fails_and_writes_nothing(self):
        engines = fake_engines({})
        stock_engine = engines["stock_engine"]

        def ignoring_zone(**kwargs):
            result = stock_engine(**kwargs)
            result["protection_zone"] = dict(result["protection_zone"], status="no_incumbents")
            return result

        engines["stock_engine"] = ignoring_zone
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = pzt.main(
                ["measure", "--registry", str(self.registry), "--manifest-id", "nov-fixture",
                 "--out-dir", str(self.out)],
                engines=engines,
                code=CODE,
            )
        self.assertEqual(code, pzt.EXIT_RUN_FAILED)
        self.assertEqual(self.files(), [])


if __name__ == "__main__":
    unittest.main()
