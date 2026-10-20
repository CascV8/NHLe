"""
Scrape the training data for the NHLe models with puckscrape.

  data/draft.csv          every NHL draft pick (records.nhl.com)
  data/landings.jsonl     each pick's NHL player landing: season-by-season totals
                          for every league (junior, Europe, WJC, AHL, NHL...)
  data/moneypuck.csv      MoneyPuck skater season summaries (star outcome metric)

Elite Prospects would be the natural source for prospect stats, but it answers
puckscrape with 403s. The NHL landing pages carry the same league-by-league
history and come back fast.

Run:  python training/scrape.py [first_draft_year] [last_draft_year]
Landings are cached, so a re-run only fetches players it doesn't have yet.
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd
import puckscrape as ps
from puckscrape import _http

DATA = Path(__file__).resolve().parent / "data"
DRAFT_API = "https://records.nhl.com/site/api/draft"
LANDING = "https://api-web.nhle.com/v1/player/{}/landing"

log = logging.getLogger("scrape")


def scrape_draft(first, last):
    rows = []
    for year in range(first, last + 1):
        data = _http.get_json(DRAFT_API, params={"cayenneExp": f"draftYear={year}"})["data"]
        log.info("draft %d: %d picks", year, len(data))
        rows.extend(data)
    df = pd.DataFrame(rows)
    keep = ["draftYear", "overallPickNumber", "roundNumber", "playerId", "playerName",
            "position", "birthDate", "countryCode", "amateurLeague", "amateurClubName", "triCode"]
    return df[keep].sort_values(["draftYear", "overallPickNumber"])


def scrape_landings(player_ids, path):
    have = set()
    if path.exists():
        with path.open(encoding="utf-8") as f:
            have = {json.loads(line)["playerId"] for line in f}
    todo = [pid for pid in player_ids if pid not in have]
    log.info("landings: %d cached, %d to fetch", len(have), len(todo))
    with path.open("a", encoding="utf-8") as f:
        for i, pid in enumerate(todo, 1):
            try:
                p = _http.get_json(LANDING.format(pid))
            except Exception as e:  # one bad player shouldn't stop the run
                log.warning("skip %s: %s", pid, e)
                continue
            f.write(json.dumps({"playerId": pid, "position": p.get("position"),
                                "seasonTotals": p.get("seasonTotals", [])}) + "\n")
            if i % 100 == 0:
                f.flush()
                log.info("  %d/%d", i, len(todo))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    first, last = (int(a) for a in sys.argv[1:3]) if len(sys.argv) >= 3 else (2006, 2019)
    DATA.mkdir(exist_ok=True)

    draft = scrape_draft(first, last)
    draft.to_csv(DATA / "draft.csv", index=False)
    ids = draft["playerId"].dropna().astype(int).tolist()
    log.info("%d picks, %d with an NHL player id", len(draft), len(ids))

    scrape_landings(ids, DATA / "landings.jsonl")

    mp = ps.moneypuck.skaters(list(range(2008, 2026)))
    mp.to_csv(DATA / "moneypuck.csv", index=False)
    log.info("moneypuck: %d rows", len(mp))


if __name__ == "__main__":
    main()
