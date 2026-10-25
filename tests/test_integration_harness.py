"""Unit tests for research_360_integration (synthetic data only; the real-data replay is reproduce.py)."""
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research_360_integration"))
import layers as L  # noqa: E402
import harness as H  # noqa: E402


def components(n=50, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"MVT_ID_mvt": [f"m{i}" for i in range(n)], "trees": rng.normal(0, 200, n), "local": rng.normal(0, 200, n),
                         "cat": rng.normal(0, 200, n), "sequence": rng.normal(0, 200, n), "gate": rng.uniform(0, 1, n),
                         "ersatz_s": rng.uniform(200, 1500, n), "schedule": rng.uniform(0, 3000, n)})


def supporting_evidence_formula(frame, airport_share=.3, neural_share=.35):
    """research_310/supporting_evidence.comparator's predict(), transcribed (9884150)."""
    tree = .4 * ((1 - airport_share) * frame.trees + airport_share * frame.local) + .6 * frame.cat
    return np.maximum((1 - frame.gate) * (frame.ersatz_s + (1 - neural_share) * tree + neural_share * frame.sequence)
                      + frame.gate * frame.schedule, 300.)


def codex_bootstrap(rows, gain, draws=5000):
    """research_350_base/metrics.bootstrap, transcribed (codex/v61-base-capacity d9cbef5)."""
    table = rows[["source_month", "airport", "date"]].copy()
    table["gain"], table["n"] = gain, 1
    groups = table.groupby(["source_month", "airport", "date"], observed=True)[["gain", "n"]].sum()
    rng = np.random.default_rng(350)
    numer, denom = np.zeros(draws), np.zeros(draws)
    for _, g in groups.groupby(level=0):
        idx = rng.integers(0, len(g), (draws, len(g)))
        numer += g.gain.to_numpy()[idx].sum(axis=1)
        denom += g.n.to_numpy()[idx].sum(axis=1)
    b = numer / denom
    return np.quantile(b, [.025, .975]).tolist(), np.quantile(b, [.00625, .99375]).tolist()


class ComposeAndApply(unittest.TestCase):
    def test_compose_equals_the_comparator_formula(self):
        rows = components()
        np.testing.assert_array_equal(L.compose_base(rows), supporting_evidence_formula(rows).to_numpy())

    def test_codex_arithmetic_differs_only_in_the_last_bits(self):
        rows = components()
        self.assertLess(float(np.max(np.abs(L.compose_codex(rows) - L.compose_base(rows)))), 1e-9)

    def test_compose_replaces_only_the_given_family(self):
        rows = components()
        new = rows.trees.to_numpy() + 10.0
        expect = supporting_evidence_formula(rows.assign(trees=new)).to_numpy()
        np.testing.assert_array_equal(L.compose_base(rows, trees=new), expect)

    def test_floor_and_residual(self):
        f = pd.DataFrame({"comparator": [250.0, 500.0], "y": [400.0, 450.0]})
        np.testing.assert_array_equal(L.apply(f, [10.0, -300.0], 1.0), [300.0, 300.0])
        np.testing.assert_array_equal(L.apply(f, [100.0, 20.0], 0.5), [300.0, 510.0])
        np.testing.assert_array_equal(L.residual_target(f), [150.0, -50.0])


class Substitution(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"MVT_ID_mvt": ["a", "b", "c"], "comparator": [400.0, 500.0, 600.0], "y": [1, 2, 3]})

    def test_same_base_is_identity(self):
        base = self.frame.set_index("MVT_ID_mvt").comparator
        pd.testing.assert_frame_equal(L.substitute_base(self.frame, base), self.frame)

    def test_new_base_is_matched_by_id(self):
        base = pd.Series([7.0, 8.0, 9.0], index=["c", "a", "b"])
        np.testing.assert_array_equal(L.substitute_base(self.frame, base).comparator, [8.0, 9.0, 7.0])

    def test_missing_or_duplicate_ids_refuse(self):
        with self.assertRaises(ValueError):
            L.substitute_base(self.frame, pd.Series([1.0, 2.0], index=["a", "b"]))
        with self.assertRaises(ValueError):
            L.substitute_base(self.frame, pd.Series([1.0, 2.0, 3.0, 4.0], index=["a", "b", "c", "c"]))


class Policy(unittest.TestCase):
    """policy_2025 with a stand-in correction that depends on the base column, as the real layers do."""

    def setUp(self):
        self.rows = pd.DataFrame({"MVT_ID_mvt": list("abcdef"), "y": [500.0] * 6})
        self.base = pd.Series([400.0, 450.0, 500.0, 550.0, 600.0, 650.0], index=list("abcdef"))
        self.frames = {"partial": pd.DataFrame({"MVT_ID_mvt": ["b", "c"], "comparator": [0.0, 0.0]}),
                       "stand": pd.DataFrame({"MVT_ID_mvt": ["e"], "comparator": [0.0]})}
        self.fake = lambda layer, frame, models=None: 0.1 * frame.comparator.to_numpy(float) - 20.0

    def test_unchanged_base_reproduces_and_owners_follow_their_rule(self):
        with mock.patch.object(L, "correction", self.fake):
            keep, pred = H.policy_2025(self.base, self.rows, self.frames, set_aside=["f"])
        self.assertEqual(list(keep.MVT_ID_mvt), list("abcde"))
        expect = {"a": 400.0, "b": 450 + 1.0 * (45 - 20), "c": 500 + 1.0 * (50 - 20), "d": 550.0, "e": 600 + 0.5 * (60 - 20)}
        np.testing.assert_allclose(pred, [expect[i] for i in "abcde"])

    def test_new_base_moves_base_rows_and_layer_inputs(self):
        new = self.base + 100.0
        with mock.patch.object(L, "correction", self.fake):
            _, p0 = H.policy_2025(self.base, self.rows, self.frames)
            _, p1 = H.policy_2025(new, self.rows, self.frames)
        np.testing.assert_allclose(p1[[0, 3, 5]] - p0[[0, 3, 5]], 100.0)          # base-owned rows
        np.testing.assert_allclose(p1[[1, 2]] - p0[[1, 2]], 100.0 + 1.0 * 10.0)     # partial: base + correction on new base
        np.testing.assert_allclose(p1[4] - p0[4], 100.0 + 0.5 * 10.0)               # stand at weight 0.5

    def test_layer_rows_must_be_scored_rows(self):
        with mock.patch.object(L, "correction", self.fake):
            with self.assertRaises(ValueError):
                H.policy_2025(self.base, self.rows, {"partial": pd.DataFrame({"MVT_ID_mvt": ["z"], "comparator": [0.0]})})


class Metrics(unittest.TestCase):
    def test_bootstrap_equals_codex_definition(self):
        rng = np.random.default_rng(3)
        n = 600
        rows = pd.DataFrame({"source_month": rng.choice([1, 7], n), "airport": rng.choice(["EDDF", "EGLL", "LEBL"], n),
                             "date": rng.choice([f"2025-01-{d:02d}" for d in range(1, 8)], n)})
        gain = rng.normal(10, 100, n)
        ours = H.bootstrap(rows, gain)
        c95, c98 = codex_bootstrap(rows, gain)
        self.assertEqual(ours["ci95"], c95)
        self.assertEqual(ours["ci98_75"], c98)

    def test_trimmed_removes_the_most_favourable(self):
        gain = np.arange(200, dtype=float)
        self.assertEqual(H.trimmed(gain), float(np.arange(100).mean()))
        self.assertIsNone(H.trimmed(np.arange(50.0)))

    def test_compare_identical_arms_is_zero(self):
        rows = pd.DataFrame({"y": [1.0, 2.0, 3.0] * 50, "source_month": [1, 7, 7] * 50, "airport": ["A", "B", "A"] * 50,
                             "date": ["d1", "d2", "d3"] * 50})
        p = rows.y.to_numpy() + 5.0
        r = H.compare(rows, p, p)
        self.assertEqual(r["mse_gain"], 0.0)
        self.assertEqual(r["best100_removed_mse_gain"], 0.0)


class Registry(unittest.TestCase):
    def test_layers_are_the_three_base_dependent_ones(self):
        self.assertEqual(sorted(L.LAYERS), ["eham_january", "partial", "stand"])
        self.assertEqual({k: v.weight for k, v in L.LAYERS.items()}, {"partial": 1.0, "stand": 0.5, "eham_january": 1.0})
        self.assertEqual((L.LAYERS["partial"].trees, L.LAYERS["partial"].leaves), (800, 31))
        cells = [c for v in L.LAYERS.values() if v.category == "partial_ground" for c in v.cells]
        self.assertEqual(len(cells), len(set(cells)))    # partial and EHAM January never share a cell


if __name__ == "__main__":
    unittest.main()
