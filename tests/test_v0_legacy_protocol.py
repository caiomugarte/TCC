"""S2: the protocol as a machine-readable contract (plan.md S2, AC 6-11).

`protocol.json` is compared with the archived modules through `ast`: the constants and literal
expressions are read out of the source without importing it. That proves equality of those
literals - including the order of literal sequences and the presence of both seeding calls -
and deliberately not their runtime semantics.
"""

import ast
import copy
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUNDLE = REPO / "baselines" / "v0-legacy"
CODE = BUNDLE / "code"
PROTOCOL = json.loads((BUNDLE / "protocol.json").read_text(encoding="utf-8"))
LEGACY_MODULES = ("ga", "ga_multiple_runs", "profiles")


def parse(name):
    path = CODE / f"{name}.py"
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def last_assignment(tree, name):
    found = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                found = node.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                found = node.value
    return found


def literal(tree, name):
    value = last_assignment(tree, name)
    if value is None:
        raise AssertionError(f"{name} is not assigned at module level")
    return ast.literal_eval(value)


def function(tree, name):
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is not defined at module level")


def seed_setup(tree):
    """(base literal, offset variable name, {seed targets}) from set_seed_for_run."""
    func = function(tree, "set_seed_for_run")
    base, offset = None, None
    for node in ast.walk(func):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "seed_value" for target in node.targets
        ):
            value = node.value
            if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
                if isinstance(value.left, ast.Constant):
                    base = value.left.value
                    offset = value.right.id if isinstance(value.right, ast.Name) else None
    seeded = set()
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        args = [arg.id for arg in node.args if isinstance(arg, ast.Name)]
        func_name = node.func
        if isinstance(func_name, ast.Name) and func_name.id == "seed" and args == ["seed_value"]:
            seeded.add("random")
        if (
            isinstance(func_name, ast.Attribute)
            and func_name.attr == "seed"
            and isinstance(func_name.value, ast.Attribute)
            and func_name.value.attr == "random"
            and isinstance(func_name.value.value, ast.Name)
            and func_name.value.value.id in ("np", "numpy")
            and args == ["seed_value"]
        ):
            seeded.add("numpy.random")
    return base, offset, seeded


def consensus_shape(tree):
    func = function(tree, "build_consensus_portfolio")
    shape = {"counter": False, "most_common": False, "n_assets_from_config": False, "reorders": False}
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            func_name = node.func
            if isinstance(func_name, ast.Name) and func_name.id == "Counter":
                shape["counter"] = True
            if isinstance(func_name, ast.Name) and func_name.id == "sorted":
                shape["reorders"] = True
            if isinstance(func_name, ast.Attribute) and func_name.attr in ("most_common", "reverse"):
                if func_name.attr == "reverse":
                    shape["reorders"] = True
                elif any(isinstance(arg, ast.Name) and arg.id == "n_assets" for arg in node.args):
                    shape["most_common"] = True
        if isinstance(node, ast.keyword) and node.arg == "reverse":
            if isinstance(node.value, ast.Constant) and node.value.value is True:
                shape["reorders"] = True
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "n_assets":
                    value = node.value
                    if (
                        isinstance(value, ast.Subscript)
                        and isinstance(value.slice, ast.Constant)
                        and value.slice.value == "n_assets"
                    ):
                        shape["n_assets_from_config"] = True
    return shape


def generation_loop(tree):
    """The `for _ in range(gens)` loop of run_ga, or None."""
    func = function(tree, "run_ga")
    for node in ast.walk(func):
        if isinstance(node, ast.For):
            iterator = node.iter
            if isinstance(iterator, ast.Call) and isinstance(iterator.func, ast.Name):
                if iterator.func.id == "range" and any(
                    isinstance(arg, ast.Name) and arg.id == "gens" for arg in iterator.args
                ):
                    return node
    return None


def violations(protocol):
    """Every disagreement between a protocol document and the archived code, by field path."""
    problems = []
    ga_tree, profiles_tree, runs_tree = parse("ga"), parse("profiles"), parse("ga_multiple_runs")

    # ---- ga parameters (AC 6)
    code_ga = literal(ga_tree, "PERFIL_CONFIG")["caio"]
    code_constants = {
        "crossover_rate": literal(ga_tree, "NCROSS_RATE"),
        "mutation_rate": literal(ga_tree, "MUT_RATE"),
    }
    declared_ga = protocol.get("ga") if isinstance(protocol.get("ga"), dict) else {}
    for key, code_value in list(code_ga.items()) + list(code_constants.items()):
        if key not in declared_ga:
            problems.append(f"ga.{key}: protocol declares no value, archived code has {code_value!r}")
        elif declared_ga[key] != code_value:
            problems.append(f"ga.{key}: protocol {declared_ga[key]!r} != archived code {code_value!r}")
    for key in sorted(set(declared_ga) - set(code_ga) - set(code_constants)):
        problems.append(f"ga.{key}: protocol declares a parameter the archived code does not have")

    # ---- profile group weights (AC 6)
    all_weights = literal(profiles_tree, "PROFILE_WEIGHTS")
    profile = protocol.get("profile")
    declared_groups = (protocol.get("scores") or {}).get("groups") or {}
    if profile not in all_weights:
        problems.append(f"profile: {profile!r} is not a profile the archived profiles.py declares")
    else:
        code_groups = all_weights[profile]
        for key, code_value in code_groups.items():
            if key not in declared_groups:
                problems.append(f"scores.groups.{key}: protocol declares no weight, archived code has {code_value!r}")
            elif declared_groups[key] != code_value:
                problems.append(
                    f"scores.groups.{key}: protocol {declared_groups[key]!r} != archived code {code_value!r}"
                )
        for key in sorted(set(declared_groups) - set(code_groups)):
            problems.append(f"scores.groups.{key}: protocol declares a weight the archived code does not have")

    # ---- runs and seed (AC 7)
    declared_runs = protocol.get("runs") if isinstance(protocol.get("runs"), dict) else {}
    code_n_runs = literal(runs_tree, "N_RUNS")
    if declared_runs.get("n_runs") != code_n_runs:
        problems.append(f"runs.n_runs: protocol {declared_runs.get('n_runs')!r} != archived N_RUNS {code_n_runs!r}")
    if declared_runs.get("run_id_start") != 0:
        problems.append(f"runs.run_id_start: protocol {declared_runs.get('run_id_start')!r}, archived range starts at 0")
    if declared_runs.get("run_id_end") != code_n_runs - 1:
        problems.append(
            f"runs.run_id_end: protocol {declared_runs.get('run_id_end')!r}, archived range ends at {code_n_runs - 1}"
        )
    base, offset, seeded = seed_setup(runs_tree)
    if base is None:
        problems.append("runs.seed_formula: set_seed_for_run has no `seed_value = <base> + <variable>`")
    else:
        if declared_runs.get("seed_base") != base:
            problems.append(f"runs.seed_base: protocol {declared_runs.get('seed_base')!r} != archived {base!r}")
        if offset != "run_id":
            problems.append(f"runs.seed_formula: archived offset is {offset!r}, not 'run_id'")
    declared_targets = declared_runs.get("seeded_on") or []
    if sorted(declared_targets) != sorted(seeded):
        problems.append(f"runs.seeded_on: protocol {sorted(declared_targets)!r} != archived {sorted(seeded)!r}")

    # ---- consensus and early stopping (AC 8)
    declared_consensus = protocol.get("consensus") if isinstance(protocol.get("consensus"), dict) else {}
    shape = consensus_shape(runs_tree)
    if declared_consensus.get("method") != "frequency":
        problems.append(f"consensus.method: protocol {declared_consensus.get('method')!r}, expected 'frequency'")
    if not shape["counter"]:
        problems.append("consensus.method: build_consensus_portfolio counts no frequencies")
    if not shape["most_common"]:
        problems.append("consensus.size_source: build_consensus_portfolio does not take most_common(n_assets)")
    if not shape["n_assets_from_config"]:
        problems.append("consensus.size_source: no n_assets read from the ga profile config")
    if declared_consensus.get("size_source") != "ga.n_assets":
        problems.append(f"consensus.size_source: protocol {declared_consensus.get('size_source')!r}")
    if declared_consensus.get("tie_break") != "first_appearance":
        problems.append(f"consensus.tie_break: protocol {declared_consensus.get('tie_break')!r}")
    if shape["reorders"]:
        problems.append("consensus.tie_break: the archived counter result is re-sorted or reversed")

    declared_early = protocol.get("early_stopping") if isinstance(protocol.get("early_stopping"), dict) else {}
    if declared_early.get("enabled") is not False:
        problems.append(f"early_stopping.enabled: protocol {declared_early.get('enabled')!r}, expected False")
    loop = generation_loop(ga_tree)
    if loop is None:
        problems.append("ga.generations: run_ga has no `for _ in range(gens)` loop")
    elif any(isinstance(node, ast.Break) for node in ast.walk(loop)):
        problems.append("early_stopping.enabled: run_ga breaks out of its generation loop")

    # ---- dependencies (AC 9)
    dependencies = protocol.get("dependencies")
    if not isinstance(dependencies, dict):
        problems.append("dependencies: protocol declares no dependencies block")
    else:
        for field, expected in (
            ("status", "UNVERIFIABLE"),
            ("declared_at_commit", "7d8d6a0"),
            ("evidence_path", "py/requirements.txt"),
        ):
            if dependencies.get(field) != expected:
                problems.append(f"dependencies.{field}: protocol {dependencies.get(field)!r} != {expected!r}")
        reason = dependencies.get("reason")
        if not isinstance(reason, str) or "declares no dependency set" not in reason:
            problems.append("dependencies.reason: does not state that 97ea327 declares no dependency set")

    return problems


def drifted(edit):
    """A deep copy of the approved protocol with `edit` applied."""
    document = copy.deepcopy(PROTOCOL)
    edit(document)
    return document


class ProtocolMatchesCodeTests(unittest.TestCase):
    def test_ga_parameters_equal_the_archived_literals(self):
        declared = PROTOCOL["ga"]
        self.assertEqual(
            (declared["n_assets"], declared["lambda"], declared["generations"], declared["pop_size"]),
            (10, 0.49, 460, 276),
        )
        code = literal(parse("ga"), "PERFIL_CONFIG")["caio"]
        self.assertEqual(declared["n_assets"], code["n_assets"])
        self.assertEqual(declared["lambda"], code["lambda"])
        self.assertEqual(declared["generations"], code["generations"])
        self.assertEqual(declared["pop_size"], code["pop_size"])

    def test_group_weights_equal_the_archived_literals(self):
        self.assertEqual(
            PROTOCOL["scores"]["groups"],
            {"liquidez": 0.22, "rent": 0.35, "value": 0.08, "growth": 0.05, "div": 0.30},
        )
        code = literal(parse("profiles"), "PROFILE_WEIGHTS")["caio"]
        self.assertEqual(PROTOCOL["scores"]["groups"], code)

    def test_runs_and_seed_formula_equal_the_archived_literals(self):
        runs = PROTOCOL["runs"]
        self.assertEqual((runs["n_runs"], runs["run_id_start"], runs["run_id_end"]), (30, 0, 29))
        self.assertEqual(runs["seed_base"], 42)
        base, offset, seeded = seed_setup(parse("ga_multiple_runs"))
        self.assertEqual(base, 42)
        self.assertEqual(offset, "run_id")
        self.assertEqual(seeded, {"random", "numpy.random"})
        self.assertEqual(sorted(runs["seeded_on"]), ["numpy.random", "random"])

    def test_consensus_rule_and_disabled_early_stop(self):
        consensus = PROTOCOL["consensus"]
        self.assertEqual(consensus["method"], "frequency")
        self.assertEqual(consensus["tie_break"], "first_appearance")
        self.assertEqual(consensus["size_source"], "ga.n_assets")
        shape = consensus_shape(parse("ga_multiple_runs"))
        self.assertTrue(shape["counter"])
        self.assertTrue(shape["most_common"])
        self.assertTrue(shape["n_assets_from_config"])
        self.assertFalse(shape["reorders"])
        self.assertIs(PROTOCOL["early_stopping"]["enabled"], False)
        loop = generation_loop(parse("ga"))
        self.assertIsNotNone(loop)
        self.assertFalse(any(isinstance(node, ast.Break) for node in ast.walk(loop)))

    def test_dependencies_are_declared_unverifiable(self):
        dependencies = PROTOCOL["dependencies"]
        self.assertEqual(dependencies["status"], "UNVERIFIABLE")
        self.assertEqual(dependencies["declared_at_commit"], "7d8d6a0")
        self.assertEqual(dependencies["evidence_path"], "py/requirements.txt")
        self.assertIn("declares no dependency set", dependencies["reason"])

    def test_ast_comparison_imports_no_legacy_module(self):
        self.assertEqual(violations(PROTOCOL), [])
        imported = [name for name in LEGACY_MODULES if name in sys.modules]
        self.assertEqual(imported, [])
        from_bundle = [
            name
            for name, module in sys.modules.items()
            if getattr(module, "__file__", None) and str(CODE) in str(module.__file__)
        ]
        self.assertEqual(from_bundle, [])


class ProtocolDriftTests(unittest.TestCase):
    def test_wrong_run_count_is_rejected_naming_the_field(self):
        problems = violations(drifted(lambda document: document["runs"].update(n_runs=29)))
        self.assertTrue(
            any(problem.startswith("runs.n_runs") for problem in problems),
            problems,
        )

    def test_missing_dependencies_block_is_rejected_naming_the_field(self):
        def drop_dependencies(document):
            del document["dependencies"]

        problems = violations(drifted(drop_dependencies))
        self.assertTrue(
            any(problem.startswith("dependencies") for problem in problems),
            problems,
        )

    def test_declared_early_stop_is_rejected_naming_the_field(self):
        problems = violations(drifted(lambda document: document["early_stopping"].update(enabled=True)))
        self.assertTrue(
            any(problem.startswith("early_stopping.enabled") for problem in problems),
            problems,
        )

    def test_unknown_ga_parameter_is_rejected_naming_the_field(self):
        problems = violations(drifted(lambda document: document["ga"].update(tournament_size=4)))
        self.assertTrue(
            any(problem.startswith("ga.tournament_size") for problem in problems),
            problems,
        )

    def test_wrong_seed_formula_is_rejected_naming_the_field(self):
        problems = violations(drifted(lambda document: document["runs"].update(seed_base=43)))
        self.assertTrue(
            any(problem.startswith("runs.seed_base") for problem in problems),
            problems,
        )


if __name__ == "__main__":
    unittest.main()
