"""
The next two draft classes for the website: every skater in the league-website tables
(puckscrape.leagues) born inside a class's eligibility window, with his stats this
season and last, and NHL Central Scouting's rankings once they're published.

  python web/prospects.py        # writes web/prospects.json (run monthly by .github/workflows/monthly.yml)

With the season under way starting in year Y, the classes are the Y+1 draft (players in
their draft year) and the Y+2 draft (their D-1 season), so they roll forward every year.
A class's players are 18 by September 15 of its draft year and not 18 by September 15
of the year before.

Players are matched across leagues by name and birth date. Sites that don't publish
birth dates (Liiga, the Swiss and German leagues) are matched by name to a player with
one in the same country, when the name is unique there.

Needs puckscrape (pip install git+https://github.com/CascV8/Puckscrape.git).
"""

import datetime as dt
import json
import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd
import puckscrape as ps
from puckscrape import _http

WEB = Path(__file__).resolve().parent
OUT = WEB / "prospects.json"
NATIONS = WEB / "prospect_nations.json"   # "client:player id" -> country, from HockeyTech profiles (kept)
RANKINGS = "https://api-web.nhle.com/v1/draft/rankings/{year}/{category}"
SKIP = {"ncaa", "sdhl", "auroraliiga"}          # NCAA: too slow for a monthly job; women's leagues
KEYS = {"hockeyallsvenskan": "allsvenskan", "extraliga": "czechia", "maxa-liga": "czechia2"}
FINNISH_JUNIOR = {"u18-sm-sarja", "u20-sm-sarja"}

log = logging.getLogger("prospects")


def name_key(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\(.*?\)|[^a-z\s'-]", " ", s).replace("-", " ").replace("'", "")
    return " ".join(sorted(s.split()))


def birth(v):
    """'May 28, 2008', 'Jun  1, 2007' or '2008-01-10' -> '2008-05-28'."""
    if not isinstance(v, str) or not v.strip():
        return None
    for fmt in ("%Y-%m-%d", "%b %d, %Y", "%B %d, %Y", "%d.%m.%Y"):
        try:
            return dt.datetime.strptime(re.sub(r"\s+", " ", v.strip()), fmt).date().isoformat()
        except ValueError:
            pass
    return None


def position(p):
    p = str(p or "").strip().lower()
    if p in {"g", "gk", "mv", "torhüter", "torhuter", "go", "goalie"}:
        return "G"
    if p in {"d", "de", "ld", "rd", "p", "op", "vp", "verteidiger", "defense", "defence", "def"} or p.startswith("d"):
        return "D"
    return "F" if p else None


def season_start(today=None):
    today = today or dt.date.today()
    return today.year if today.month >= 9 else today.year - 1


def window(draft_year):
    """Birth dates of a class: 18 by Sept 15 of its draft year, not by Sept 15 of the year before."""
    return dt.date(draft_year - 19, 9, 16).isoformat(), dt.date(draft_year - 18, 9, 15).isoformat()


def tables(seasons):
    leagues = ps.leagues.leagues()
    rows = []
    for code, country in zip(leagues.code, leagues.country):
        if code in SKIP:
            continue
        for y in seasons:
            for phase in ("regular", "playoffs"):
                try:
                    df = ps.leagues.skaters(code, y, phase=phase)
                except Exception as e:
                    log.info("%s %d %s: %s", code, y, phase, str(e)[:70])
                    continue
                if df.empty:
                    continue
                bcol = next((c for c in ("birthdate", "birth_date") if c in df), None)
                for r in df.to_dict("records"):
                    rows.append({
                        "code": code, "country": country, "season": y, "type": "reg" if phase == "regular" else "po",
                        "name": r.get("player"), "key": name_key(r.get("player")), "born": birth(r.get(bcol)) if bcol else None,
                        "pos": position(r.get("position")), "team": r.get("team") or "",
                        "nation": r.get("birthcntry") or r.get("nationality") or "", "site_id": r.get("player_id"),
                        "gp": int(r.get("gp") or 0), "g": int(r.get("g") or 0), "a": int(r.get("a") or 0), "pts": int(r.get("pts") or 0)})
                log.info("%s %d %s: %d", code, y, phase, len(df))
    return pd.DataFrame(rows)


def hockeytech_nation(code, site_id, cache):
    """A HockeyTech player's country from his profile (the stats tables leave it blank): his home
    country, else his birth country. Looked up once and kept in web/prospect_nations.json."""
    from puckscrape.leagues import LEAGUES, hockeytech
    lg = LEAGUES.get(code)
    if lg is None or lg.adapter is not hockeytech or site_id is None or site_id != site_id:
        return ""
    client = lg.opts["client"]
    k = f"{client}:{int(site_id)}"
    if k not in cache:
        try:
            info = hockeytech._kit(client, view="player", category="profile", player_id=int(site_id)).get("Player") or {}
            cache[k] = info.get("homecntry") or info.get("birthcntry") or ""
        except Exception as e:
            log.info("profile %s: %s", k, str(e)[:60])
            return ""
    return cache[k]


def central_scouting(year):
    """NHL Central Scouting's lists for a draft year (once published): {list name: [players]}."""
    out = {}
    for category in (1, 2):           # North American skaters, international skaters
        try:
            d = _http.get_json(RANKINGS.format(year=year, category=category))
        except Exception:
            continue
        rows = d.get("rankings") or []
        if rows:
            name = next((c["name"] for c in d.get("categories", []) if c.get("id") == category), f"list {category}")
            out[f"Central Scouting: {name}"] = rows
    return out


def build(today=None):
    y = season_start(today)
    t = tables([y - 1, y])
    t = t[t.pos != "G"]
    classes = {y + 1: "draft year", y + 2: "D-1"}
    # Birth dates for players listed without one: the unique same-name player with one in the same country
    known = t.dropna(subset=["born"]).groupby(["country", "key"]).born.agg(lambda b: b.iloc[0] if b.nunique() == 1 else None)
    t["born"] = [b if b else known.get((c, k)) for b, c, k in zip(t.born, t.country, t.key)]
    t = t.dropna(subset=["born"])
    players, lists = {}, {}
    nations = json.loads(NATIONS.read_text(encoding="utf-8")) if NATIONS.exists() else {}
    for draft_year, label in classes.items():
        lo, hi = window(draft_year)
        c = t[(t.born >= lo) & (t.born <= hi)]
        ids = []
        for (key, born), g in c.groupby(["key", "born"]):
            pid = f"{key.replace(' ', '-')}-{born}"
            reg = g[(g.type == "reg") & (g.season == y)]
            main = (reg if len(reg) else g).sort_values("gp", ascending=False).iloc[0]
            nation = next((n for n in g.nation if n), "")
            for r in g.itertuples():           # still blank: a HockeyTech profile has it
                if nation:
                    break
                nation = hockeytech_nation(r.code, r.site_id, nations)
            if not nation and set(g.code) & FINNISH_JUNIOR:
                nation = "Finland"   # Finland's junior leagues publish no nationality; nearly all their players are Finnish
            players[pid] = {
                "id": pid, "name": main["name"], "birthDate": born, "position": g.pos.mode().iloc[0],
                "team": main["team"], "league": KEYS.get(main["code"], main["code"]), "nation": nation,
                "draft": None, "class": draft_year,
                "stats": [{"season": int(r.season), "type": r.type, "league": KEYS.get(r.code, r.code), "team": r.team,
                           "gp": r.gp, "g": r.g, "a": r.a, "pts": r.pts} for r in g.itertuples() if r.gp > 0]}
            ids.append(pid)
        # Central Scouting ranks, matched by name and birth date; ranked players we have no table for are added
        ranks = {}
        for list_name, rows in central_scouting(draft_year).items():
            ranks[list_name] = {}
            for r in rows:
                nm = f"{r.get('firstName', '')} {r.get('lastName', '')}".strip()
                born = (r.get("birthDate") or "")[:10]
                pid = f"{name_key(nm).replace(' ', '-')}-{born}"
                if pid not in players:
                    players[pid] = {"id": pid, "name": nm, "birthDate": born, "position": position(r.get("positionCode")),
                                    "team": r.get("lastAmateurClub", ""), "league": (r.get("lastAmateurLeague") or "").lower(),
                                    "nation": r.get("birthCountry", ""), "draft": None, "class": draft_year, "stats": []}
                    ids.append(pid)
                ranks[list_name][pid] = r.get("finalRank") or r.get("midtermRank")
        lists[str(draft_year)] = {"label": label, "players": sorted(set(ids)), "rankings": ranks}
        log.info("%d class (%s): %d players, rankings: %s", draft_year, label, len(set(ids)), list(ranks) or "none yet")
    NATIONS.write_text(json.dumps(dict(sorted(nations.items())), ensure_ascii=False), encoding="utf-8")
    out = {"generated": (today or dt.date.today()).isoformat(), "season": y, "classes": lists, "players": players}
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    build()
