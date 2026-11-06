"""web/prospects.py's matching rules.  Needs pandas and puckscrape; skipped without them.  Run: python -m unittest"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "web"))
try:
    import pandas as pd
    import prospects
except ImportError:   # the CI test job has neither
    prospects = None


@unittest.skipIf(prospects is None, "needs pandas and puckscrape")
class TestSamePerson(unittest.TestCase):
    def test_spellings_of_one_player(self):
        for a, b in [("Timo Kazda", "Timothy Kazda"), ("A.J. Gladman", "AJ Gladman"), ("Alex Campeau", "Alexandre Campeau"),
                     ("Nate Kydd", "Nathan Kydd"), ("Kosta Housseas", "Konstantinos Housseas"), ("Xavier Rowe", "Xavvier Rowe"),
                     ("Zachary Wang", "Zackary Wang"), ("Noah Gaudet-Barton", "Noah Barton"), ("Ben Geiger", "Benjamin Geiger"),
                     ("Mateo Lobas Worger", "Mateo Lobos-Worger"), ("John McLaughlin", "Johnny McLaughlin"),
                     ("Artem Katsuro", "Artyom Katsuro"), ("Evgeny Yakunin", "Yevgeni Yakunin"),
                     ("Andrei Pustovoy", "Andrei Pustovoi"), ("Neuenschwander Jonah", "Jonah Neuenschwander"),
                     ("Jake Smith", "Jacob Smith")]:
            with self.subTest(a=a, b=b):
                self.assertTrue(prospects.same_person(a, b))

    def test_twins_stay_two_players(self):
        for a, b in [("Oliver Antell", "Theo Antell"), ("Samuel Skrtic", "Simon Skrtic"), ("Hannu Packalen", "Henri Packalen"),
                     ("Becker Wenkus", "Miller Wenkus"), ("Braden Guolla", "Brady Guolla"), ("Grady Dale", "Graiden Dale"),
                     ("Andrei I. Volkov", "Sergei I. Volkov"), ("Daniil Miftakhov", "Nikita Miftakhov")]:
            with self.subTest(a=a, b=b):
                self.assertFalse(prospects.same_person(a, b))

    def test_person_keys_merge_and_keep_the_old_id(self):
        t = pd.DataFrame([
            {"name": "Timothy Kazda", "born": "2008-10-04", "season": 2025, "type": "reg", "gp": 59},
            {"name": "Timo Kazda", "born": "2008-10-04", "season": 2026, "type": "reg", "gp": 12},
            {"name": "Theo Antell", "born": "2009-01-01", "season": 2026, "type": "reg", "gp": 5},
            {"name": "Oliver Antell", "born": "2009-01-01", "season": 2026, "type": "reg", "gp": 6}])
        t["key"] = t.name.map(prospects.name_key)
        keys = prospects.person_keys(t)
        self.assertEqual(keys[0], keys[1])
        self.assertEqual(keys[1], prospects.name_key("Timo Kazda"))   # the name he plays under this season
        self.assertNotEqual(keys[2], keys[3])
        old = {"kazda-timothy-2008-10-04"}
        self.assertEqual(prospects.person_keys(t, old)[1], prospects.name_key("Timothy Kazda"))   # his card's id stays

    def test_same_team(self):
        self.assertTrue(prospects.same_team("HC Kometa Brno U20", "HC KOMETA BRNO"))
        self.assertTrue(prospects.same_team("CSKA Moskva", "CSKA"))
        self.assertTrue(prospects.same_team("EHC Biel-Bienne", "EHC Biel-Bienne Spirit"))
        self.assertFalse(prospects.same_team("Guelph Storm", "Chicago Steel"))

    def test_ep_pool_updates_a_stale_team_and_fills_the_country(self):
        players = {"kazda-timothy-2008-10-04": {
            "id": "kazda-timothy-2008-10-04", "name": "Timothy Kazda", "birthDate": "2008-10-04", "team": "Chicago Steel",
            "league": "ushl", "nation": "", "class": 2027,
            "stats": [{"season": 2025, "type": "reg", "league": "ushl", "team": "Chicago Steel", "gp": 59, "g": 1, "a": 1, "pts": 2}]}}
        ids = list(players)
        pool = pd.DataFrame([{"player": "Timothy Kazda", "player_id": 1, "position": "C", "nation": "Slovakia",
                              "team": "Guelph Storm", "league": "OHL", "born": "2008-10-04"},
                             {"player": "Petr Tomek", "player_id": 2, "position": "D", "nation": "Czechia",
                              "team": "Energie Karlovy Vary", "league": "Czechia", "born": "2008-10-23"}])
        pool["key"] = pool.player.map(prospects.name_key)
        prospects.apply_ep_pool(players, ids, pool, 2027, 2026)
        p = players["kazda-timothy-2008-10-04"]
        self.assertEqual((p["team"], p["league"], p["nation"]), ("Guelph Storm", "ohl", "Slovakia"))
        self.assertIn("petr-tomek-2008-10-23", players)
        self.assertEqual(players["petr-tomek-2008-10-23"]["league"], "czechia")


if __name__ == "__main__":
    unittest.main()
