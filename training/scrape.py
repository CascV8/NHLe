"""
Scrape the training data for the NHLe models with puckscrape.

  data/draft.csv          every NHL draft pick (records.nhl.com)
  data/landings.jsonl     NHL player landings (every pick, plus every NHL skater since
                          2000-01): season-by-season totals for every league
                          (junior, Europe, WJC, AHL, NHL...)
  data/moneypuck.csv      MoneyPuck skater season summaries (star outcome metric)
  data/tables/*.csv       league-website skater tables (ps.leagues), for the network
                          NHLe factors in training/network.py

Elite Prospects would be the natural source for prospect stats, but it answers
puckscrape with 403s. The NHL landing pages carry the same league-by-league
history and come back fast.

Run:  python training/scrape.py [first_draft_year] [last_draft_year]
Landings and league tables are cached, so a re-run only fetches what it doesn't have yet.
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
            name = f"{p.get('firstName', {}).get('default', '')} {p.get('lastName', {}).get('default', '')}"
            f.write(json.dumps({"playerId": pid, "position": p.get("position"), "name": name.strip(),
                                "birthDate": p.get("birthDate"),
                                "seasonTotals": p.get("seasonTotals", [])}) + "\n")
            if i % 100 == 0:
                f.flush()
                log.info("  %d/%d", i, len(todo))


def nhl_skater_ids(first, last):
    """Every skater who played an NHL game in seasons first..last (start years)."""
    ids = set()
    for y in range(first, last + 1):
        df = ps.nhl.skater_stats(y)
        if "player_id" in df:  # 2004-05 (lockout) is empty
            ids |= set(df["player_id"].dropna().astype(int))
    return ids


# League-website tables for the network factors: every league ps.leagues has
# whose level is useful for NHLe (all but the women's leagues).
TABLE_SEASONS = range(2010, 2026)
SKIP_TABLES = {"sdhl", "auroraliiga"}


def scrape_tables(seasons):
    out = DATA / "tables"
    out.mkdir(exist_ok=True)
    for code in ps.leagues.leagues().code:
        if code in SKIP_TABLES:
            continue
        for y in seasons:
            for phase in ("regular",):   # the network factors are regular season only
                path = out / f"{code}_{y}_{phase}.csv"
                miss = path.with_suffix(".none")
                if path.exists() or miss.exists():
                    continue
                try:
                    df = ps.leagues.skaters(code, y, phase=phase)
                except Exception as e:  # season not on the site, phase unsupported...
                    log.info("tables %s %d %s: %s", code, y, phase, str(e)[:80])
                    miss.write_text(str(e)[:500], encoding="utf-8")
                    continue
                if df.empty:   # a site having a bad moment; try again next run
                    log.warning("tables %s %d %s: empty, not cached", code, y, phase)
                    continue
                df.to_csv(path, index=False)
                log.info("tables %s %d %s: %d skaters", code, y, phase, len(df))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    first, last = (int(a) for a in args[:2]) if len(args) >= 2 else (2000, 2025)
    DATA.mkdir(exist_ok=True)

    draft = scrape_draft(first, last)
    draft.to_csv(DATA / "draft.csv", index=False)
    ids = draft["playerId"].dropna().astype(int).tolist()
    log.info("%d picks, %d with an NHL player id", len(draft), len(ids))
    # NHL skaters who were never drafted (or drafted before `first`) widen the
    # network: their AHL, NCAA and European seasons link leagues too.
    extra = sorted(nhl_skater_ids(2000, 2025) - set(ids))
    log.info("%d more NHL skaters", len(extra))

    scrape_landings(ids + extra, DATA / "landings.jsonl")
    if "--no-tables" not in sys.argv:
        scrape_tables(TABLE_SEASONS)

    mp = ps.moneypuck.skaters(list(range(2008, 2026)))
    mp.to_csv(DATA / "moneypuck.csv", index=False)
    log.info("moneypuck: %d rows", len(mp))


if __name__ == "__main__":
    main()
