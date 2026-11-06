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
birth dates (the NCAA, Liiga, the Swiss and German leagues) are matched by name to a
player with one in the same region (North America, or the same European country), when
the name is unique there: an NCAA freshman by his junior seasons.

Needs puckscrape (pip install git+https://github.com/CascV8/Puckscrape.git).
"""

import datetime as dt
import io
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
# Each player's games per team this season as of the last build, and the day each last went up: a player traded or
# moved mid-season shows his new team once he plays there, not the team he has the most games with.
GAMES = WEB / "prospect_games.json"       # {player id: {"league|team": [games, "YYYY-MM-DD" they last went up]}}
RANKINGS = "https://api-web.nhle.com/v1/draft/rankings/{year}/{category}"
# Other rankings, saved by hand from wherever they're published (EP's consolidated ranking, TSN's Bob McKenzie,
# The Athletic's Scott Wheeler, Daily Faceoff...): web/rankings/<draft year>/<any name>.csv, one row per player,
#     # title: Daily Faceoff (Oct 9, Steven Ellis)      <- the name in the Ranking menu (else the file name)
#     rank,name,born                                     <- born (YYYY-MM-DD) optional: it settles two players
#     1,Landon DuPont,                                      of the same name
OWN_RANKINGS = WEB / "rankings"
# Women's leagues; and the VHL, whose puckscrape table (HockeyDB) has no birth dates: russia_tables() reads it
# with each teenager's date instead.
SKIP = {"sdhl", "auroraliiga", "vhl"}
# Name matching for sites without birth dates runs within a region: North America is one (a freshman's
# NCAA line matches his junior seasons in Canada or the US), each European country its own.
REGION = {"Canada": "North America", "USA": "North America"}
KEYS = {"hockeyallsvenskan": "allsvenskan", "extraliga": "czechia", "maxa-liga": "czechia2",
        "iihf-wm20": "wjc-20", "iihf-wm18": "wjc-18",   # the World Juniors as the models know them
        "czech-u20": "czechia-u20", "czech-u18": "czechia-u18", "u21-elit": "u21-elite"}
# Leagues whose season takes many requests (the KHL app feed pages its players 16 at a time): a finished season's
# rows are kept in KEPT and read once, like HockeyDB's.
SLOW = {"khl", "mhl"}
KEPT = WEB / "prospect_kept_seasons.json"   # "<code>:<year>:<phase>" -> that finished season's rows (kept)
# Elite Prospects fills in what the league sites leave out. Its Draft Center pool (the upcoming draft's prospects,
# with birth dates) gives a draft-year player's country and current team; for everyone else still without a
# country, EP's league table for his league this season does (matched by name and team).
EP_LEAGUES = {"ohl": "ohl", "whl": "whl", "qmjhl": "qmjhl", "ushl": "ushl", "bchl": "bchl", "ajhl": "ajhl",
              "sjhl": "sjhl", "mjhl": "mjhl", "ojhl": "ojhl", "cchl": "cchl", "mjahl": "mjahl", "gohl": "gojhl",
              "kijhl": "kijhl", "vijhl": "vijhl", "csshl": "csshl-u18"}
# The same person written two ways ("Timo"/"Timothy Kazda", "A.J."/"AJ") is merged by birth date and surname
# when the first names agree: one starts the other, they're a common nickname pair, or one letter apart.
NICKNAMES = {frozenset(p) for p in [
    ("nate", "nathan"), ("nate", "nathaniel"), ("kosta", "konstantinos"), ("kosta", "konstantin"), ("mike", "michael"),
    ("mikey", "michael"), ("danny", "daniel"), ("tony", "anthony"), ("zack", "zachary"), ("zack", "zachery"),
    ("jake", "jacob"), ("jim", "james"), ("jimmy", "james"), ("joe", "joseph"), ("joey", "joseph"), ("bill", "william"),
    ("billy", "william"), ("bob", "robert"), ("bobby", "robert"), ("dick", "richard"), ("rick", "richard"),
    ("ricky", "richard"), ("teddy", "edward"), ("ted", "edward"), ("sasha", "alexander"), ("sasha", "aleksandr"),
    ("misha", "mikhail"), ("dima", "dmitri"), ("dima", "dmitry"), ("vova", "vladimir"), ("kolya", "nikolai"),
    ("johnny", "jonathan"), ("jon", "jonathan"), ("nick", "nicholas"), ("nicky", "nicholas"), ("chris", "christopher"),
    ("matty", "matthew"), ("mitch", "mitchell"), ("ollie", "oliver"), ("bo", "beau")]}
TOURNAMENTS = {"iihf-wm20", "iihf-wm18"}   # the only national-team events kept (the models use them)
FINNISH_JUNIOR = {"u18-sm-sarja", "u20-sm-sarja"}
SWEDISH = {"shl", "hockeyallsvenskan", "hockeyettan", "u20-nationell", "u18-nationell"}   # SHL tables give "N/A"
# Russia's leagues block scripts on their own sites; HockeyDB has their rosters (a season once it's well under way).
# Its rows give age and birthplace but not birth dates, so each teenager's date is read from his HockeyDB page once
# and kept (HOCKEYDB_BIRTHS); HockeyDB asks for a few seconds between requests.
# The KHL and MHL now come from the KHL's own app feed through puckscrape (with birth dates); the VHL still
# comes from HockeyDB.
RUSSIA = {352: "vhl"}   # HockeyDB league ids
HOCKEYDB_BIRTHS = WEB / "prospect_hockeydb_births.json"   # HockeyDB player id -> birth date (kept)
HOCKEYDB_SEASONS = WEB / "prospect_hockeydb_seasons.json"   # "<league id>:<year>" -> a complete season's rows (kept:
                                                            # it won't change, so it's read from HockeyDB once)
MONTHS = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
BLANK = {"", "n/a", "na", "-", "--", "none", "unknown"}
# A HockeyTech profile with no country often has its province or state (homeprov): AB, ON, Alaska...
PROVINCES = {"AB", "BC", "MB", "NB", "NL", "NS", "NT", "NU", "ON", "PE", "QC", "SK", "YT", "PEI", "PQ", "NF", "ALBERTA",
             "BRITISH COLUMBIA", "MANITOBA", "NEW BRUNSWICK", "NEWFOUNDLAND", "NEWFOUNDLAND AND LABRADOR", "NOVA SCOTIA",
             "NORTHWEST TERRITORIES", "NUNAVUT", "ONTARIO", "PRINCE EDWARD ISLAND", "QUEBEC", "QUÉBEC", "SASKATCHEWAN", "YUKON"}
STATES = {"AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA",
          "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR",
          "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "ALABAMA", "ALASKA", "ARIZONA",
          "ARKANSAS", "CALIFORNIA", "COLORADO", "CONNECTICUT", "DELAWARE", "FLORIDA", "GEORGIA", "HAWAII", "IDAHO",
          "ILLINOIS", "INDIANA", "IOWA", "KANSAS", "KENTUCKY", "LOUISIANA", "MAINE", "MARYLAND", "MASSACHUSETTS",
          "MICHIGAN", "MINNESOTA", "MISSISSIPPI", "MISSOURI", "MONTANA", "NEBRASKA", "NEVADA", "NEW HAMPSHIRE",
          "NEW JERSEY", "NEW MEXICO", "NEW YORK", "NORTH CAROLINA", "NORTH DAKOTA", "OHIO", "OKLAHOMA", "OREGON",
          "PENNSYLVANIA", "RHODE ISLAND", "SOUTH CAROLINA", "SOUTH DAKOTA", "TENNESSEE", "TEXAS", "UTAH", "VERMONT",
          "VIRGINIA", "WASHINGTON", "WEST VIRGINIA", "WISCONSIN", "WYOMING"}


def nation_of(v):
    """A table's nationality, or "" for the placeholders some sites print ("N/A")."""
    v = str(v or "").strip()
    return "" if v.lower() in BLANK else v


def from_province(prov):
    p = str(prov or "").strip().upper().rstrip(".")
    return "Canada" if p in PROVINCES else "USA" if p in STATES else ""

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


def tables(seasons, current=None):
    leagues = ps.leagues.leagues()
    kept = json.loads(KEPT.read_text(encoding="utf-8")) if KEPT.exists() else {}
    rows = []
    for code, country, level in zip(leagues.code, leagues.country, leagues.level):
        # men's club leagues, and the U20 and U18 World Championships
        if code in SKIP or "women" in str(level) or ("national" in str(level) and code not in TOURNAMENTS):
            continue
        for y in seasons:
            for phase in ("regular", "playoffs"):
                slot = f"{code}:{y}:{phase}"
                if code in SLOW and y != current and slot in kept:
                    rows += kept[slot]
                    continue
                try:
                    df = ps.leagues.skaters(code, y, phase=phase)
                except Exception as e:
                    log.info("%s %d %s: %s", code, y, phase, str(e)[:70])
                    continue
                if df.empty:
                    continue
                bcol = next((c for c in ("birthdate", "birth_date") if c in df), None)
                start = len(rows)
                for r in df.to_dict("records"):
                    site_id = r.get("player_id")
                    rows.append({
                        "code": code, "country": REGION.get(country, country), "season": y, "type": "reg" if phase == "regular" else "po",
                        "name": r.get("player"), "key": name_key(r.get("player")), "born": birth(r.get(bcol)) if bcol else None,
                        "pos": position(r.get("position")), "team": r.get("team") or "",
                        "nation": nation_of(r.get("birthcntry")) or nation_of(r.get("nationality")),
                        "site_id": None if site_id is None or site_id != site_id else site_id,
                        "gp": int(r.get("gp") or 0), "g": int(r.get("g") or 0), "a": int(r.get("a") or 0), "pts": int(r.get("pts") or 0)})
                if code in SLOW and current is not None and y != current:
                    kept[slot] = rows[start:]
                log.info("%s %d %s: %d", code, y, phase, len(df))
    KEPT.write_text(json.dumps(dict(sorted(kept.items())), ensure_ascii=False, separators=(",", ":"), default=str),
                    encoding="utf-8")
    return pd.DataFrame(rows)


def name_tokens(n):
    """'A.J. Gladman' and 'AJ Gladman' -> ['aj', 'gladman']; accents and punctuation dropped."""
    s = unicodedata.normalize("NFKD", str(n or "")).encode("ascii", "ignore").decode().lower()
    toks = re.sub(r"[^a-z\s]", " ", s.replace(".", " ").replace("-", " ").replace("'", "")).split()
    i = 0
    while i < len(toks) and len(toks[i]) == 1:
        i += 1
    return ["".join(toks[:i])] + toks[i:] if i >= 2 else toks


# Transliteration variants of the same Russian/Ukrainian name ("Artyom"/"Artem", "Yevgeni"/"Evgeny",
# "Pustovoy"/"Pustovoi"), folded before names are compared.
FOLD = [("yo", "e"), ("jo", "e"), ("ye", "e"), ("je", "e"), ("iy", "y"), ("ii", "i"), ("yi", "i"), ("oy", "oi"),
        ("ay", "ai"), ("ey", "ei"), ("kh", "h"), ("ks", "x"), ("tz", "ts"), ("j", "y")]


def fold(word):
    for a, b in FOLD:
        word = word.replace(a, b)
    return re.sub(r"(.)\1", r"\1", word)   # doubled letters: "Kirillov"/"Kirilov"


def match_key(name):
    """A name for matching across sites: tokens folded and sorted ('Pustovoi Andrei' = 'Andrei Pustovoy'), without
    the middle initial the KHL feed adds to tell namesakes apart ('Sergei A. Skvortsov' = 'Sergei Skvortsov'; two
    namesakes then share the key, and a ranking that matches both is left unmatched)."""
    toks = name_tokens(name)
    return " ".join(sorted(fold(t) for i, t in enumerate(toks) if len(t) > 1 or i == 0))


def one_edit(a, b):
    """True if a and b differ by one inserted, deleted or changed letter."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    short, long_ = sorted((a, b), key=len)
    return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


def same_person(a, b):
    """Two spellings of one player born the same day: same surname and agreeing first names ("Timo"/"Timothy",
    "Nate"/"Nathan", "Xavier"/"Xavvier"), or the same first name with one surname inside the other ("Noah
    Gaudet-Barton"/"Noah Barton"). Different first names (twins: "Samuel"/"Simon Skrtic") are two people."""
    ra, rb = name_tokens(a), name_tokens(b)
    if len(ra) < 2 or len(rb) < 2:
        return False
    ta, tb = [fold(w) for w in ra], [fold(w) for w in rb]
    if sorted(ta) == sorted(tb):   # the same words in another order ("Neuenschwander Jonah")
        return True
    if ta[-1] != tb[-1]:
        return ta[0] == tb[0] and (set(ta[1:]) <= set(tb[1:]) or set(tb[1:]) <= set(ta[1:]))
    if frozenset((ra[0], rb[0])) in NICKNAMES:
        return True
    fa, fb = ta[0], tb[0]
    if fa == fb:
        return True
    short, long_ = sorted((fa, fb), key=len)
    return (len(short) >= 3 and long_.startswith(short)) or (len(short) >= 5 and one_edit(fa, fb))


def person_keys(t, old_ids=()):
    """One name key per player: rows born the same day whose names are two spellings of one person (same_person)
    share a key, the one his card already had (``old_ids``) or else the one he plays under this season."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    names = t.groupby(["born", "key"]).name.first()
    for born, keys in names.groupby(level=0):
        items = list(keys.items())
        for i, ((_, ka), na) in enumerate(items):
            find((born, ka))
            for (_, kb), nb in items[i + 1:]:
                if same_person(na, nb):
                    parent[find((born, ka))] = find((born, kb))
    members = {}
    for node in list(parent):
        members.setdefault(find(node), []).append(node)
    current = t.assign(_gp=t.gp.where(t.type == "reg", 0)).groupby(["born", "key"]).agg(season=("season", "max"), gp=("_gp", "sum"))
    canon = {}
    for nodes in members.values():
        def rank(node):
            born, key = node
            pid = f"{key.replace(' ', '-')}-{born}"
            s = current.loc[node] if node in current.index else None
            return (pid in old_ids, int(s.season) if s is not None else 0, int(s.gp) if s is not None else 0)
        best = max(nodes, key=rank)[1]
        for node in nodes:
            canon[node] = best
    return [canon.get((b, k), k) for b, k in zip(t.born, t.key)]


def team_key(s):
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower())


def same_team(a, b):
    """'HC Kometa Brno U20' ~ 'HC KOMETA BRNO', 'CSKA Moskva' ~ 'CSKA': one name inside the other, or a shared
    word of five letters or more."""
    ka, kb = team_key(a), team_key(b)
    if not ka or not kb:
        return False
    if ka in kb or kb in ka:
        return True
    words = lambda s: {w for w in re.findall(r"[a-z]{5,}", unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower())}
    return bool(words(a) & words(b) - {"hockey", "junior", "juniors", "academy", "prep", "club"})


EP_POOL = WEB / "prospect_ep_pool.json"   # the last Draft Center pool read (kept: EP challenges GitHub's runners)
EP_COLUMNS = ["draft_year", "player", "player_id", "position", "dob", "nation", "team", "league"]
EP_BLOCKED = False   # set once EP answers with a Cloudflare challenge: nothing more is asked of it this run


def ep_pool(draft_year):
    """Elite Prospects' Draft Center pool for a draft year (its ranked and unranked prospects, with birth dates,
    country and current team). Elite Prospects puts a Cloudflare challenge on datacenter addresses like GitHub's
    runners, so the pool is kept in web/prospect_ep_pool.json whenever it's read, and that copy is used when EP
    can't be reached (run web/prospects.py from a home connection now and then to refresh it)."""
    global EP_BLOCKED
    df = pd.DataFrame()
    try:
        df = ps.eliteprospects.draft_rankings("consolidated-ranking", unranked=True)
    except Exception as e:
        log.info("Elite Prospects pool: %s", str(e)[:80])
    if not df.empty and "draft_year" in df and (df.draft_year == draft_year).any():
        df = df[df.draft_year == draft_year].drop_duplicates("player_id")
        EP_POOL.write_text(json.dumps({"draft_year": draft_year, "read": dt.date.today().isoformat(),
                                       "players": df[[c for c in EP_COLUMNS if c in df]].astype(object)
                                       .where(df.notna(), None).to_dict("records")}, ensure_ascii=False, default=str),
                           encoding="utf-8")
        log.info("Elite Prospects pool for %d: %d players", draft_year, len(df))
    else:
        EP_BLOCKED = True
        kept = json.loads(EP_POOL.read_text(encoding="utf-8")) if EP_POOL.exists() else {}
        if kept.get("draft_year") != draft_year:
            log.info("Elite Prospects pool: not reachable, and no kept pool for %d", draft_year)
            return pd.DataFrame()
        df = pd.DataFrame(kept["players"])
        log.info("Elite Prospects pool: not reachable; using the one kept from %s (%d players)", kept.get("read"), len(df))
    df = df.copy()
    df["key"] = df.player.map(name_key)
    df["born"] = df.dob.astype(str).str[:10]
    return df


def ep_league(name):
    """An EP league name as the card's league key: 'OHL' -> 'ohl', 'Czechia U20' -> 'czechia-u20'."""
    return re.sub(r"\s+", "-", str(name or "").strip().lower())


def seed_births(t, pool, season):
    """Birth dates from EP's pool for rows without one (the Czech and Swiss sites publish none): the row's name is
    a pool player's and it's at the team EP lists him with this season."""
    if pool.empty:
        return t
    by_key = {k: g for k, g in pool.groupby("key")}
    born = list(t.born)
    for i, (b, k, team, y) in enumerate(zip(t.born, t.key, t.team, t.season)):
        if isinstance(b, str) or k not in by_key or y != season:
            continue
        hit = [r for r in by_key[k].itertuples() if same_team(team, r.team)]
        if len(hit) == 1:
            born[i] = hit[0].born
    seeded = sum(isinstance(a, str) and not isinstance(b, str) for a, b in zip(born, t.born))
    log.info("Birth dates from the Elite Prospects pool: %d rows", seeded)
    return t.assign(born=born)


def ep_league_nations(players, season, cache):
    """Countries for players still without one, from EP's league table for the league they play in this season
    (matched by name and team). Found countries are kept in ``cache`` ("ep:<player id>"), so a player is
    looked up once."""
    for pid, p in players.items():
        if not p["nation"] and cache.get(f"ep:{pid}"):
            p["nation"] = cache[f"ep:{pid}"]
    blank = {}
    for pid, p in players.items():
        if not p["nation"] and p["league"] in EP_LEAGUES:
            blank.setdefault(p["league"], []).append(pid)
    if EP_BLOCKED:
        log.info("Elite Prospects isn't answering this run: countries kept from earlier runs only")
        return
    for league, pids in blank.items():
        try:
            df = ps.eliteprospects.skaters(EP_LEAGUES[league], f"{season}-{season + 1}")
        except Exception as e:
            log.info("Elite Prospects %s: %s; no more EP lookups this run", league, str(e)[:70])
            break
        if df.empty:   # EP returns nothing (rather than raising) when Cloudflare keeps refusing
            log.info("Elite Prospects %s: nothing returned; no more EP lookups this run", league)
            break
        df = df.assign(key=df.player.map(name_key))
        found = 0
        for pid in pids:
            p = players[pid]
            rows = df[(df.key == name_key(p["name"])) & df.nation.notna()]
            if len(rows) > 1:
                rows = rows[[same_team(p["team"], tm) for tm in rows.team]]
            if len(rows) == 1:
                p["nation"] = cache[f"ep:{pid}"] = rows.nation.iloc[0]
                found += 1
        log.info("Elite Prospects %s: countries for %d of %d players", league, found, len(pids))


def apply_ep_pool(players, ids, pool, draft_year, season, spellings=None):
    """EP's pool on a class: a matched player missing a country gets EP's, and one whose current team the league
    sites don't show (no line this season, or no team) gets EP's current team and league. Pool players the sites
    don't have are added as EP lists them."""
    if pool.empty:
        return
    mine = {}
    for pid in ids:
        p = players[pid]
        mine.setdefault(p["birthDate"], []).append(pid)
    added = updated = 0
    for r in pool.itertuples():
        hits = [pid for pid in mine.get(r.born, [])
                if any(same_person(nm, r.player) for nm in (spellings or {}).get(pid, set()) | {players[pid]["name"]})]
        if len(hits) == 1:
            p = players[hits[0]]
            if spellings is not None:
                spellings.setdefault(hits[0], set()).add(r.player)
            p["ep"] = (int(r.player_id) if pd.notna(r.player_id) else None)
            if not p["nation"] and isinstance(r.nation, str):
                p["nation"] = r.nation
            stale = not any(s["season"] == season and s["type"] == "reg" for s in p["stats"])
            if (stale or not p["team"]) and isinstance(r.team, str) and r.team:
                p["team"], p["league"] = r.team, KEYS.get(ep_league(r.league), ep_league(r.league))
                updated += 1
        elif not hits:
            pid = f"{r.key.replace(' ', '-')}-{r.born}"
            if pid not in players:
                players[pid] = {"id": pid, "name": r.player, "birthDate": r.born, "position": position(str(r.position).split("/")[0]),
                                "team": r.team if isinstance(r.team, str) else "", "league": ep_league(r.league),
                                "nation": r.nation if isinstance(r.nation, str) else "", "draft": None, "class": draft_year,
                                "stats": [], "ep": (int(r.player_id) if pd.notna(r.player_id) else None)}
                ids.append(pid)
                added += 1
    log.info("%d class: Elite Prospects updated %d teams, added %d players", draft_year, updated, added)


def hockeydb_birth(pid, cache):
    """A HockeyDB player's birth date from his page ("Born Feb 22 2007"), kept in ``cache``."""
    k = str(int(pid))
    if k not in cache:
        try:
            text = ps.hockeydb._soup(f"/ihdb/stats/pdisplay.php?pid={k}").get_text(" ", strip=True)
            m = re.search(r"Born\s+([A-Z][a-z]{2})[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})", text)
            cache[k] = f"{m.group(3)}-{MONTHS[m.group(1)]:02d}-{int(m.group(2)):02d}" if m and m.group(1) in MONTHS else ""
        except Exception as e:
            log.info("hockeydb %s: %s", k, str(e)[:60])
            return ""
    return cache[k]


def russia_tables(seasons, current, max_age=18):
    """The KHL, VHL and MHL from HockeyDB: rows like tables()'s for every player ``max_age`` or younger that season.
    A complete season is read once and kept; the season under way is read each run. If HockeyDB refuses (403),
    nothing more is asked of it this run."""
    births = json.loads(HOCKEYDB_BIRTHS.read_text(encoding="utf-8")) if HOCKEYDB_BIRTHS.exists() else {}
    kept = json.loads(HOCKEYDB_SEASONS.read_text(encoding="utf-8")) if HOCKEYDB_SEASONS.exists() else {}
    rows, refused = [], False
    for lid, code in RUSSIA.items():
        for y in seasons:
            key = f"{lid}:{y}"
            if y != current and key in kept:
                rows += kept[key]
                continue
            if refused:
                continue
            try:
                df = ps.hockeydb.skaters(lid, y)
            except Exception as e:
                log.info("%s %d: %s", code, y, str(e)[:70])
                refused = refused or "403" in str(e)
                continue
            if df.empty:
                log.info("%s %d: not on HockeyDB yet", code, y)
                continue
            young = df[pd.to_numeric(df.age, errors="coerce") <= max_age]
            start = len(rows)
            for r in young.to_dict("records"):
                born = hockeydb_birth(r["pid"], births)
                if not born:
                    continue
                place = str(r.get("birthplace") or "")
                nation = place.split(":")[0].strip() if ":" in place else "Russia"
                base = {"code": code, "country": "Russia", "season": y, "name": r["player"], "key": name_key(r["player"]),
                        "born": born, "pos": position(r.get("pos")), "team": r.get("team") or "", "nation": nation,
                        "site_id": None}
                rows.append(base | {"type": "reg", "gp": int(r.get("gp") or 0), "g": int(r.get("g") or 0),
                                    "a": int(r.get("a") or 0), "pts": int(r.get("pts") or 0)})
                if int(r.get("po_gp") or 0):
                    rows.append(base | {"type": "po", "gp": int(r["po_gp"]), "g": int(r.get("po_g") or 0),
                                        "a": int(r.get("po_a") or 0), "pts": int(r.get("po_pts") or 0)})
            if y != current and all(hockeydb_birth(p, births) for p in young.pid):   # complete: keep it
                kept[key] = rows[start:]
            log.info("%s %d: %d players %d or younger", code, y, len(young), max_age)
    HOCKEYDB_BIRTHS.write_text(json.dumps(dict(sorted(births.items())), separators=(",", ":")), encoding="utf-8")
    HOCKEYDB_SEASONS.write_text(json.dumps(dict(sorted(kept.items())), ensure_ascii=False, separators=(",", ":")),
                                encoding="utf-8")
    return pd.DataFrame(rows)


def hockeytech_nation(code, site_id, cache):
    """A HockeyTech player's country from his profile (the stats tables leave it blank): his home country, else his
    birth country, else the country of his home province or state. Kept in web/prospect_nations.json; one cached
    blank is looked up again."""
    from puckscrape.leagues import LEAGUES, hockeytech
    lg = LEAGUES.get(code)
    if lg is None or lg.adapter is not hockeytech or site_id is None or site_id != site_id:
        return ""
    client = lg.opts["client"]
    k = f"{client}:{int(site_id)}"
    if not cache.get(k):
        try:
            info = hockeytech._kit(client, view="player", category="profile", player_id=int(site_id)).get("Player") or {}
            cache[k] = (nation_of(info.get("homecntry")) or nation_of(info.get("birthcntry"))
                        or from_province(info.get("homeprov")) or from_province(info.get("birthprov")))
        except Exception as e:
            log.info("profile %s: %s", k, str(e)[:60])
            return ""
    return cache[k]


def own_rankings(year):
    """The saved rankings for a draft year: {list name: [{"rank", "name", "born"}]}."""
    out = {}
    for path in sorted((OWN_RANKINGS / str(year)).glob("*.csv")):
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        title = next((l.split(":", 1)[1].strip() for l in lines if l.lower().startswith("# title:")), path.stem)
        rows = [l for l in lines if l.strip() and not l.lstrip().startswith("#")]
        df = pd.read_csv(io.StringIO("\n".join(rows)), dtype=str).fillna("")
        out[title] = [{"rank": int(r["rank"]), "name": r["name"].strip(), "born": birth(r.get("born", "")) or ""}
                      for r in df.to_dict("records") if r.get("rank", "").strip().isdigit() and r.get("name", "").strip()]
    return out


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
    old_ids = set(json.loads(OUT.read_text(encoding="utf-8")).get("players", {})) if OUT.exists() else set()
    t = pd.concat([tables([y - 1, y], current=y), russia_tables([y - 1, y], current=y)], ignore_index=True)
    t = t[t.pos != "G"]
    pool = ep_pool(y + 1)
    t = seed_births(t, pool, y)
    classes = {y + 1: "draft year", y + 2: "D-1"}
    # Birth dates for players listed without one: the unique same-name player with one in the same country
    known = t.dropna(subset=["born"]).groupby(["country", "key"]).born.agg(lambda b: b.iloc[0] if b.nunique() == 1 else None)
    t["born"] = [b if isinstance(b, str) else known.get((c, k)) for b, c, k in zip(t.born, t.country, t.key)]   # NaN is truthy
    t = t.dropna(subset=["born"])
    # An NCAA line in a season the same name also played junior is two people (an older college player who
    # shares a prospect's name): a teenager doesn't play both in one season, so the NCAA line goes.
    junior = set(zip(t[(t.code != "ncaa") & (t.type == "reg")].key, t[(t.code != "ncaa") & (t.type == "reg")].born,
                     t[(t.code != "ncaa") & (t.type == "reg")].season))
    namesake = pd.Series([(k, b, y) in junior for k, b, y in zip(t.key, t.born, t.season)], index=t.index)
    # Nor is anyone in college before 17 (on October 1 of the season): a younger "match" is a namesake too
    too_young = pd.Series([b > f"{y - 17}-10-01" for b, y in zip(t.born, t.season)], index=t.index)
    t = t[~((t.code == "ncaa") & (namesake | too_young))]
    # Two spellings of one player ("Timo"/"Timothy Kazda") become one, so his new team isn't left on a second card
    t = t.assign(key=person_keys(t, old_ids))
    players, lists = {}, {}
    nations = json.loads(NATIONS.read_text(encoding="utf-8")) if NATIONS.exists() else {}
    games = json.loads(GAMES.read_text(encoding="utf-8")) if GAMES.exists() else {}
    day = (today or dt.date.today()).isoformat()
    new_games, spellings = {}, {}
    for draft_year, label in classes.items():
        lo, hi = window(draft_year)
        c = t[(t.born >= lo) & (t.born <= hi)]
        ids = []
        for (key, born), g in c.groupby(["key", "born"]):
            pid = f"{key.replace(' ', '-')}-{born}"
            reg = g[(g.type == "reg") & (g.season == y)]
            # this season's teams: the one where his games went up most recently, else the one with the most games
            seen = {}
            for r in reg.itertuples():
                k, before = f"{r.code}|{r.team}", games.get(pid, {}).get(f"{r.code}|{r.team}")
                went_up = day if before is not None and r.gp > before[0] else (before[1] if before else "")
                seen[r.Index] = went_up
                new_games.setdefault(pid, {})[k] = [int(r.gp), went_up]
            main = (reg.assign(_seen=reg.index.map(seen)).sort_values(["_seen", "gp"], ascending=False).iloc[0]
                    if len(reg) else g.sort_values("gp", ascending=False).iloc[0])
            nation = next((n for n in g.nation if n), "")
            for r in g.itertuples():           # still blank: a HockeyTech profile has it
                if nation:
                    break
                nation = hockeytech_nation(r.code, r.site_id, nations)
            if not nation and set(g.code) & FINNISH_JUNIOR:
                nation = "Finland"   # Finland's junior leagues publish no nationality; nearly all their players are Finnish
            if not nation and set(g.code) & SWEDISH:
                nation = "Sweden"    # nor do Sweden's ("N/A"); nearly all their draft-age players are Swedish
            players[pid] = {
                "id": pid, "name": main["name"], "birthDate": born, "position": g.pos.mode().iloc[0],
                "team": main["team"], "league": KEYS.get(main["code"], main["code"]), "nation": nation,
                "draft": None, "class": draft_year,
                "stats": [{"season": int(r.season), "type": r.type, "league": KEYS.get(r.code, r.code), "team": r.team,
                           "gp": r.gp, "g": r.g, "a": r.a, "pts": r.pts} for r in g.itertuples() if r.gp > 0]}
            spellings[pid] = set(g.name)
            ids.append(pid)
        if draft_year == y + 1:
            apply_ep_pool(players, ids, pool, draft_year, y, spellings)
        # Every spelling a player was found under, for matching rankings ("Timothy" in a list, "Timo" in the OHL's table)
        in_class = {}
        for pid in ids:
            for nm in spellings.get(pid, set()) | {players[pid]["name"]}:
                if pid not in in_class.setdefault(match_key(nm), []):
                    in_class[match_key(nm)].append(pid)
        # Central Scouting ranks, matched by name and birth date; ranked players we have no table for are added
        ranks = {}
        for list_name, rows in central_scouting(draft_year).items():
            ranks[list_name] = {}
            for r in rows:
                nm = f"{r.get('firstName', '')} {r.get('lastName', '')}".strip()
                born = (r.get("birthDate") or "")[:10]
                found = [p for p in in_class.get(match_key(nm), []) if players[p]["birthDate"] == born]
                pid = found[0] if len(found) == 1 else f"{name_key(nm).replace(' ', '-')}-{born}"
                if pid not in players:
                    players[pid] = {"id": pid, "name": nm, "birthDate": born, "position": position(r.get("positionCode")),
                                    "team": r.get("lastAmateurClub", ""), "league": (r.get("lastAmateurLeague") or "").lower(),
                                    "nation": r.get("birthCountry", ""), "draft": None, "class": draft_year, "stats": []}
                    ids.append(pid)
                ranks[list_name][pid] = r.get("finalRank") or r.get("midtermRank")
        # Saved rankings, matched by name within the class (and birth date when the list gives one)
        for list_name, rows in own_rankings(draft_year).items():
            ranks[list_name], missing = {}, []
            for r in rows:
                found = [p for p in in_class.get(match_key(r["name"]), []) if not r["born"] or players[p]["birthDate"] == r["born"]]
                if len(found) == 1:
                    pid = found[0]
                else:   # not in the tables we read (or two of that name): listed as the ranking has him
                    pid = f"{name_key(r['name']).replace(' ', '-')}-{r['born'] or 'ranked'}"
                    missing.append(r["name"])
                    if pid not in players:
                        players[pid] = {"id": pid, "name": r["name"], "birthDate": r["born"], "position": None, "team": "",
                                        "league": "", "nation": "", "draft": None, "class": draft_year, "stats": []}
                        ids.append(pid)
                ranks[list_name][pid] = r["rank"]
            log.info("%d %s: %d ranked, %d not matched to a table%s", draft_year, list_name, len(rows), len(missing),
                     f" ({', '.join(missing)})" if missing else "")
        lists[str(draft_year)] = {"label": label, "players": sorted(set(ids)), "rankings": ranks}
        log.info("%d class (%s): %d players, rankings: %s", draft_year, label, len(set(ids)), list(ranks) or "none yet")
    ep_league_nations(players, y, nations)
    NATIONS.write_text(json.dumps(dict(sorted(nations.items())), ensure_ascii=False), encoding="utf-8")
    GAMES.write_text(json.dumps(dict(sorted(new_games.items())), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    out = {"generated": (today or dt.date.today()).isoformat(), "season": y, "classes": lists, "players": players}
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    build()
