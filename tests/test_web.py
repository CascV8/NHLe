"""The website's model spec (web/build.py) must match the scripts.  Run: python -m unittest"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "web"))
import build  # noqa: E402


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestWebSpec(unittest.TestCase):
    spec = build.model_spec()

    def test_coefficients_match_scripts(self):
        for group, models in self.spec["models"].items():
            for rel, m in models.items():
                with self.subTest(model=m["script"]):
                    script = load(ROOT / m["script"])
                    self.assertEqual(m["nhler"], [script.NHLER_INTERCEPT, script.NHLER_COEF])
                    self.assertEqual(m["star"], [script.STAR_INTERCEPT, script.STAR_COEF])
                    # D+3 on leave the WJC out of their table; every factor they do have must agree.
                    self.assertEqual(script.LEAGUE_FACTORS,
                                     {k: v for k, v in self.spec["factors"].items() if k in script.LEAGUE_FACTORS})
                    self.assertTrue(set(m["tournaments"]) <= set(script.LEAGUE_FACTORS))
        self.assertEqual(sum(len(m) for m in self.spec["models"].values()), 14)

    def test_aliases_point_at_factors(self):
        for name, key in self.spec["aliases"].items():
            with self.subTest(name=name):
                self.assertIn(key, self.spec["factors"])
                self.assertEqual(name, name.lower())

    def test_logos_are_for_rated_leagues(self):
        self.assertIn("nhl", self.spec["logos"])
        for key, path in self.spec["logos"].items():
            with self.subTest(league=key):
                self.assertIn(key, self.spec["factors"])
                self.assertTrue((ROOT / "web" / "logos" / Path(path).name).is_file())

    def test_build_writes_the_site_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = build.build(Path(tmp) / "site", Path(tmp) / "functions")
            for name in ("nhle.html", "nhle.js", "nhle-model.json"):
                self.assertTrue((site / name).is_file(), name)
            self.assertTrue((Path(tmp) / "functions" / "api" / "[[path]].js").is_file())
            self.assertTrue((site / "nhle-logos" / "nhl.svg").is_file())


if __name__ == "__main__":
    unittest.main()
