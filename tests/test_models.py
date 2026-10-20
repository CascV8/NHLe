"""Sanity checks for every NHLe model script.  Run: python -m unittest"""

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATHS = sorted(ROOT.glob("Forwards/*.py")) + sorted(ROOT.glob("Defense/*.py"))


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MODELS = {p.relative_to(ROOT).as_posix(): load(p) for p in MODEL_PATHS}


class TestModels(unittest.TestCase):
    def test_found_models(self):
        self.assertEqual(len(MODELS), 14)

    def test_probabilities_valid_and_monotonic(self):
        for name, m in MODELS.items():
            with self.subTest(model=name):
                prev_n = prev_s = 0.0
                for pts82 in range(0, 151):
                    pn, ps = m.p_nhler(pts82), m.p_star(pts82)
                    self.assertTrue(0.0 <= ps <= pn <= 1.0, (pts82, pn, ps))
                    self.assertGreaterEqual(pn, prev_n)
                    self.assertGreaterEqual(ps, prev_s)
                    prev_n, prev_s = pn, ps

    def test_league_factors_consistent(self):
        # Every copy of the table must agree on shared keys, so a typo in one
        # model can't silently change a factor or a league name.
        first_seen = {}
        for name, m in MODELS.items():
            for key, factor in m.LEAGUE_FACTORS.items():
                with self.subTest(model=name, league=key):
                    self.assertEqual(first_seen.setdefault(key, factor), factor)
                    self.assertTrue(0 < factor <= 1)
        common = set.intersection(*(set(m.LEAGUE_FACTORS) for m in MODELS.values()))
        self.assertIn("u21-elite", common)

    def test_parse_league_rejects_bad_custom_factor(self):
        for name, m in MODELS.items():
            with self.subTest(model=name):
                self.assertEqual(m.parse_league(" OHL "), ("ohl", m.LEAGUE_FACTORS["ohl"]))
                self.assertEqual(m.parse_league("0.25"), ("0.25", 0.25))
                for bad in ("nan", "inf", "-0.3", "0", "1.5", "notaleague"):
                    with self.assertRaises(ValueError):
                        m.parse_league(bad)

    def test_parse_num_rejects_non_finite_and_negative(self):
        for name, m in MODELS.items():
            with self.subTest(model=name):
                self.assertEqual(m.parse_num("12.5"), 12.5)
                for bad in ("nan", "inf", "-inf", "-1", "abc"):
                    with self.assertRaises(ValueError):
                        m.parse_num(bad)

    def test_wjc_adjustment_neutral_and_bounded(self):
        for name, m in MODELS.items():
            if not hasattr(m, "wjc_log_odds_adj"):
                continue
            with self.subTest(model=name):
                neutral = m.wjc_log_odds_adj(m.WJC_ADJ_CENTER * 100, 100)
                self.assertAlmostEqual(neutral, 0.0)
                self.assertLessEqual(m.wjc_log_odds_adj(100, 100), m.WJC_ADJ_MAX_POS)
                self.assertGreaterEqual(m.wjc_log_odds_adj(0, 100), m.WJC_ADJ_MAX_NEG)


class TestCli(unittest.TestCase):
    def run_cli(self, path, *args):
        # stdout is a pipe here, which is exactly the case that used to crash
        # on Windows when printing the ★/█ bar
        return subprocess.run([sys.executable, str(path), *args],
                              capture_output=True, timeout=30)

    def test_pts82_flag(self):
        for path in MODEL_PATHS:
            with self.subTest(model=path.name):
                res = self.run_cli(path, "--pts82", "40")
                self.assertEqual(res.returncode, 0, res.stderr.decode(errors="replace"))
                self.assertIn("NHL player (200 GP+)", res.stdout.decode("utf-8"))

    def test_pts82_flag_rejects_bad_value(self):
        for path in MODEL_PATHS:
            for bad in ("nan", "-10000"):
                with self.subTest(model=path.name, value=bad):
                    res = self.run_cli(path, "--pts82", bad)
                    self.assertNotEqual(res.returncode, 0)
                    self.assertNotIn(b"Traceback", res.stderr)


if __name__ == "__main__":
    unittest.main()
