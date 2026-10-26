"""
Network NHLe: league factors from players who played in two leagues in the same
season, solved jointly over every league pair.

  python training/network.py      # writes training/factors.json + training/NETWORK.md

The method is the one in "What's a point worth?" (Beyond the Weiwu): a league's
NHLe is the NHL points per game of the players who played in both it and the NHL
that season, over their points per game in the league (AHL 2025-26: 0.242 / 0.620
= 0.389). Leagues with few NHL crossovers go through connector leagues
(SHL -> AHL -> NHL), each path weighted by its share of players:

  NHLe(X via C) = PPG_C / PPG_X  x  NHLe(C)      (C = NHL is the direct path)
  NHLe(X)       = sum over C of  share_C x NHLe(X via C)
  share_C       = players in both X and C / players in X and any linked league

PPG is pooled over the players in both leagues that season (points and games
summed), over every season. Every league with enough shared players is a
connector for every other, so the connectors' own NHLe values are solved at the
same time: a first pass solves leagues one at a time outward from the NHL
(fixed at 1), each from the connectors solved before it, then every league is
recomputed once from all of its connections. A one-shot weighted least-squares
solve in log space is reported next to it as a cross-check (`wls` in factors.json).

Data (training/scrape.py):
  landings.jsonl   NHL career histories (exact ids): every pick 2000-2025 and every
                   NHL skater since 2000-01. These cover the leagues whose own
                   sites are blocked or missing (KHL, VHL, MHL, Slovakia, the WJC...).
  hockeydb/*.csv   HockeyDB rosters (puckscrape.hockeydb) for the KHL, VHL, MHL,
                   Slovakia and NAHL, whose own sites are blocked or missing, and
                   the leagues their players move to and from. HockeyDB ids link
                   stints exactly; only pairs with one of those five are used.
  ep/*.csv         Elite Prospects tables for the same leagues plus the WJC-20,
                   linked by Elite Prospects id under the same rule.
  tables/*.csv     league-website tables (puckscrape.leagues): the players who
                   never got an NHL id (AHL/ECHL lifers, Allsvenskan, Mestis,
                   U20 leagues...). Linked by name within a season, only between
                   leagues of the same country or system (never the NCAA, whose
                   name matches with pro leagues are almost all two people).

Rules: regular season only; seasons 2010-11 to 2025-26; 10+ games in each league
(3+ at a tournament); a pair needs MIN_PAIR players; a league needs MIN_LEAGUE
players across its pairs and a path to the NHL. Standard errors are from a
bootstrap over players.
"""

import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit import LEAGUE_ALIASES as ALIASES, TOURNAMENTS, load_script  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

SEASONS = range(2010, 2026)      # start years: 2010-11 .. 2025-26
MIN_GP = 10
MIN_GP_TOURNAMENT = 3
MIN_PAIR = 10
MIN_LEAGUE = 25
BOOTSTRAP = 200


# ps.leagues codes -> factor keys, and the system each belongs to (website names
# are linked only within one system, where a shared name is one player).
TABLES = {
    # The NCAA is its own system: a college player can't play 10 pro games and stay
    # in school, so a name in both is two people. NCAA links come from the landings.
    "ncaa": ("ncaa", "ncaa"), "ohl": ("ohl", "na"), "whl": ("whl", "na"), "qmjhl": ("qmjhl", "na"),
    "ahl": ("ahl", "na"), "echl": ("echl", "na"), "ushl": ("ushl", "na"), "bchl": ("bchl", "na"),
    "ajhl": ("ajhl", "na"), "sjhl": ("sjhl", "na"), "mjhl": ("mjhl", "na"), "ojhl": ("ojhl", "na"),
    "cchl": ("cchl", "na"), "mjahl": ("mjahl", "na"), "kijhl": ("kijhl", "na"), "vijhl": ("vijhl", "na"),
    "shl": ("shl", "se"), "hockeyallsvenskan": ("allsvenskan", "se"),
    "liiga": ("liiga", "fi"), "mestis": ("mestis", "fi"), "suomi-sarja": ("suomi-sarja", "fi"),
    "u20-sm-sarja": ("u20-sm-sarja", "fi"), "u18-sm-sarja": ("u18-sm-sarja", "fi"),
    "nl": ("nl", "ch"), "sl": ("sl", "ch"), "myhl": ("myhl", "ch"),
    "del": ("del", "de"), "del2": ("del2", "de"),
    "extraliga": ("czechia", "cz"), "maxa-liga": ("czechia2", "cz"),
    "icehl": ("icehl", "at"), "eihl": ("eihl", "uk"), "metal-ligaen": ("denmark", "dk"),
}


def name_key(s):
    """'Kaprizov, Kirill', 'KAPRIZOV Kirill' and 'Kirill Kaprizov' -> 'kaprizov kirill'."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\(.*?\)|«.*?»|[^a-z\s'-]", " ", s).replace("-", " ").replace("'", "")
    return " ".join(sorted(s.split()))


# ── Stints: one row per player, season and league ────────────────────────────

def landing_stints():
    rows, names = [], []
    with (DATA / "landings.jsonl").open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pid = rec["playerId"]
            if rec.get("position") == "G":
                continue
            names.append((pid, name_key(rec.get("name", ""))))
            for s in rec["seasonTotals"]:
                y = int(str(s["season"])[:4])
                key = ALIASES.get(s.get("leagueAbbrev"))
                if s["gameTypeId"] != 2 or key is None or y not in SEASONS:
                    continue
                rows.append((f"n{pid}", y, key, s.get("gamesPlayed") or 0, s.get("points") or 0))
    df = pd.DataFrame(rows, columns=["player", "season", "league", "gp", "pts"])
    names = pd.DataFrame(names, columns=["pid", "key"])
    names["player"] = "n" + names.pid.astype(str)
    return df.groupby(["player", "season", "league"], as_index=False)[["gp", "pts"]].sum(), names


def table_stints(landing, names):
    """League-website rows, linked by name within a season and system."""
    frames = []
    for path in sorted((DATA / "tables").glob("*_regular.csv")):
        code, y, _ = path.stem.rsplit("_", 2)
        if code not in TABLES or int(y) not in SEASONS:
            continue
        df = pd.read_csv(path, usecols=lambda c: c in {"player", "gp", "pts"})
        key, system = TABLES[code]
        df = df.assign(season=int(y), league=key, system=system, name=df.player.map(name_key))
        frames.append(df[df.name.str.count(" ") >= 1])
    if not frames:
        return pd.DataFrame(columns=["player", "season", "league", "gp", "pts"])
    t = pd.concat(frames, ignore_index=True)
    t[["gp", "pts"]] = t[["gp", "pts"]].apply(pd.to_numeric, errors="coerce").fillna(0)
    # A name on two teams in one league-season is one player traded (sites split
    # by team); two different people with one name is rarer, and unfixable.
    t = t.groupby(["name", "season", "league", "system"], as_index=False)[["gp", "pts"]].sum()
    # A name in two systems in one season is two people, or a move we can't confirm.
    systems = t.groupby(["name", "season"]).system.transform("nunique")
    t = t[(systems == 1) & (t.system != "ncaa")]
    # Players with an NHL id are already in the landings, league by league.
    lid = landing.merge(names[["player", "key"]], on="player")
    known = set(zip(lid.key, lid.season, lid.league))
    t = t[[(n, s, lg) not in known for n, s, lg in zip(t.name, t.season, t.league)]]
    t["player"] = "t" + t.system + ":" + t.name
    return t[["player", "season", "league", "gp", "pts"]]


# HockeyDB codes (training/scrape.py) -> factor keys. Its pairs count only when
# one side is a league HockeyDB is here for; the rest are already covered.
HOCKEYDB = {"khl": "khl", "vhl": "vhl", "mhl": "mhl", "slovakia": "slovakia", "nahl": "nahl",
            "czechia": "czechia", "liiga": "liiga", "shl": "shl", "nl": "nl", "del": "del",
            "ushl": "ushl", "bchl": "bchl", "ajhl": "ajhl", "sjhl": "sjhl", "mjhl": "mjhl", "ojhl": "ojhl"}
HOCKEYDB_TARGETS = {"khl", "vhl", "mhl", "slovakia", "nahl"}


# Elite Prospects tables (data/ep/{league}_{year}.csv: ep_id, player, team, gp, pts,
# skaters only) for the same leagues plus the WJC-20; the same rule applies.
EP = {k: k for k in ("khl", "vhl", "mhl", "slovakia", "nahl", "wjc-20", "czechia", "liiga", "shl",
                     "nl", "del", "ushl", "bchl", "ajhl", "sjhl", "mjhl", "ojhl")}
EP_TARGETS = {"khl", "vhl", "mhl", "slovakia", "nahl", "wjc-20"}


def id_stints(folder, codes, id_col, prefix, landing, names, skaters=lambda df: df):
    """Rows from a source with its own player ids (HockeyDB, Elite Prospects),
    one per id, season and league (team splits summed). Players with an NHL id
    are left to the landings."""
    frames = []
    for path in sorted((DATA / folder).glob("*.csv")):
        code, y = path.stem.rsplit("_", 1)
        if code not in codes or int(y) not in SEASONS:
            continue
        df = skaters(pd.read_csv(path))
        frames.append(df[[id_col, "player", "gp", "pts"]].assign(season=int(y), league=codes[code]))
    if not frames:
        return pd.DataFrame(columns=["player", "season", "league", "gp", "pts"])
    h = pd.concat(frames, ignore_index=True)
    h[["gp", "pts"]] = h[["gp", "pts"]].apply(pd.to_numeric, errors="coerce").fillna(0)
    h["name"] = h.player.map(name_key)
    lid = landing.merge(names[["player", "key"]], on="player")
    known = set(zip(lid.key, lid.season))
    h = h[[(n, y) not in known for n, y in zip(h.name, h.season)]]
    h = h.groupby([id_col, "season", "league"], as_index=False)[["gp", "pts"]].sum()
    h["player"] = prefix + h[id_col].astype(str)
    return h[["player", "season", "league", "gp", "pts"]]


def hockeydb_stints(landing, names):
    return id_stints("hockeydb", HOCKEYDB, "pid", "h", landing, names, skaters=lambda df: df[df.pos != "G"])


def ep_stints(landing, names):
    return id_stints("ep", EP, "ep_id", "e", landing, names)


# ── Pairs and the solve ──────────────────────────────────────────────────────

def pair_rows(stints):
    """One row per (player, season, league A, league B) with A < B."""
    min_gp = np.where(stints.league.isin(TOURNAMENTS), MIN_GP_TOURNAMENT, MIN_GP)
    s = stints[stints.gp >= min_gp]
    multi = s.groupby(["player", "season"]).league.transform("size") >= 2
    s = s[multi]
    m = s.merge(s, on=["player", "season"], suffixes=("_a", "_b"))
    return m[m.league_a < m.league_b]


def pool(pairs):
    g = pairs.groupby(["league_a", "league_b"]).agg(
        n=("player", "size"), gp_a=("gp_a", "sum"), pts_a=("pts_a", "sum"),
        gp_b=("gp_b", "sum"), pts_b=("pts_b", "sum")).reset_index()
    g = g[(g.n >= MIN_PAIR) & (g.pts_a > 0) & (g.pts_b > 0)]
    g["ppg_a"], g["ppg_b"] = g.pts_a / g.gp_a, g.pts_b / g.gp_b
    g["log_ratio"] = np.log(g.ppg_b / g.ppg_a)       # = log f_A - log f_B
    return g


def connected(g, root="nhl"):
    adj = defaultdict(set)
    for a, b in zip(g.league_a, g.league_b):
        adj[a].add(b)
        adj[b].add(a)
    seen, todo = {root}, [root]
    while todo:
        for nxt in adj[todo.pop()] - seen:
            seen.add(nxt)
            todo.append(nxt)
    return seen


def links(g):
    """Each league's paths: {X: [(connector C, players, PPG_C / PPG_X)]}."""
    out = defaultdict(list)
    for a, b, n, pa, pb in zip(g.league_a, g.league_b, g.n, g.ppg_a, g.ppg_b):
        out[a].append((b, int(n), pb / pa))
        out[b].append((a, int(n), pa / pb))
    return out


def solve(g):
    """The article's connector paths. Leagues are solved one at a time from the
    NHL (fixed at 1) outward: next is the league sharing the most players with
    the leagues already solved, and its NHLe is the share-weighted sum of its
    paths through them, NHLe(X) = sum_C share_C * (PPG_C / PPG_X) * NHLe(C).
    The AHL comes out direct-only, the ECHL through the AHL and NHL, the SHL
    through the NHL, AHL, KHL... A league is never its own connector, so there
    are no loops for a ratio to compound around."""
    paths = links(g)
    f, order = {"nhl": 1.0}, {}
    while True:
        todo = {x: [(c, n, r) for c, n, r in ps if c in f] for x, ps in paths.items() if x not in f}
        todo = {x: ps for x, ps in todo.items() if ps}
        if not todo:
            break
        x = max(todo, key=lambda k: sum(n for _, n, _ in todo[k]))
        total = sum(n for _, n, _ in todo[x])
        f[x] = sum(n / total * r * f[c] for c, n, r in todo[x])
        order[x] = todo[x]
    solve.order = order
    # Then every league again from all of its connections, now that every
    # connector has a value: the AHL through the NHL, ECHL, KHL, SHL..., not just
    # the NHL it was solved from. One pass, not repeated to a fixed point, which
    # compounds small ratio errors around loops (that version put the SHL above 1).
    final = {"nhl": 1.0}
    for x, ps in paths.items():
        if x == "nhl" or x not in f:
            continue
        ps = [(c, n, r) for c, n, r in ps if c in f]
        total = sum(n for _, n, _ in ps)
        final[x] = sum(n / total * r * f[c] for c, n, r in ps)
    solve.first = f
    return final


def path_table(order, f):
    """Each league's paths, with their share and the NHLe each one gives."""
    out = {}
    for x, ps in order.items():
        total = sum(n for _, n, _ in ps)
        out[x] = sorted(((c, n / total, r * f[c]) for c, n, r in ps), key=lambda t: -t[1])
    return out


def solve_wls(g):
    """Weighted least squares for log factors, NHL fixed at 0 (the cross-check)."""
    leagues = sorted((set(g.league_a) | set(g.league_b)) - {"nhl"})
    idx = {lg: i for i, lg in enumerate(leagues)}
    A = np.zeros((len(g), len(leagues)))
    for r, (a, b) in enumerate(zip(g.league_a, g.league_b)):
        if a != "nhl":
            A[r, idx[a]] += 1
        if b != "nhl":
            A[r, idx[b]] -= 1
    w = np.sqrt(g.n.to_numpy(float))
    theta, *_ = np.linalg.lstsq(A * w[:, None], g.log_ratio.to_numpy() * w, rcond=None)
    return {"nhl": 1.0, **{lg: float(np.exp(theta[idx[lg]])) for lg in leagues}}


def usable(g):
    """Pairs among leagues with enough players and a path to the NHL."""
    while True:
        per = pd.concat([g[["league_a", "n"]].rename(columns={"league_a": "lg"}),
                         g[["league_b", "n"]].rename(columns={"league_b": "lg"})]).groupby("lg").n.sum()
        keep = (set(per[per >= MIN_LEAGUE].index) | {"nhl"}) & connected(g)
        g2 = g[g.league_a.isin(keep) & g.league_b.isin(keep)]
        if len(g2) == len(g):
            return g2, per
        g = g2


def main():
    landing, names = landing_stints()
    tables = table_stints(landing, names)
    hdb = hockeydb_stints(landing, names)
    ep = ep_stints(landing, names)
    stints = pd.concat([landing, tables, hdb, ep], ignore_index=True)
    stints[["gp", "pts"]] = stints[["gp", "pts"]].astype(float)
    pairs = pair_rows(stints)
    # HockeyDB and Elite Prospects pairs count only with one of the leagues they're here for
    for prefix, targets in (("h", HOCKEYDB_TARGETS), ("e", EP_TARGETS)):
        mine = pairs.player.str[0] == prefix
        pairs = pairs[~mine | pairs.league_a.isin(targets) | pairs.league_b.isin(targets)]
    g, per = usable(pool(pairs))
    factors = solve(g)
    wls = solve_wls(g)
    first = solve.first
    paths = path_table(links(g), first)
    rank = {lg: i + 1 for i, lg in enumerate(solve.order)}

    # Bootstrap over players: resample whole player-seasons' pair sets.
    rng = np.random.default_rng(0)
    players = pairs.player.unique()
    by_player = {p: i for i, p in enumerate(players)}
    pidx = pairs.player.map(by_player).to_numpy()
    order = np.argsort(pidx)
    starts = np.searchsorted(pidx[order], np.arange(len(players) + 1))
    boot = defaultdict(list)
    for _ in range(BOOTSTRAP):
        draw = rng.integers(0, len(players), len(players))
        rows = np.concatenate([order[starts[i]:starts[i + 1]] for i in draw])
        bg = pool(pairs.iloc[rows])
        bg = bg[bg.league_a.isin(factors) & bg.league_b.isin(factors)]
        bg = bg[bg.league_a.isin(connected(bg)) & bg.league_b.isin(connected(bg))]
        for lg, f in solve(bg).items():
            boot[lg].append(np.log(f))

    direct = g[(g.league_a == "nhl") | (g.league_b == "nhl")]
    direct_n = {(b if a == "nhl" else a): int(n) for a, b, n in zip(direct.league_a, direct.league_b, direct.n)}
    out = {}
    for lg, f in sorted(factors.items(), key=lambda kv: -kv[1]):
        se = float(np.std(boot[lg])) if lg != "nhl" and len(boot[lg]) > 10 else 0.0
        lg_pairs = g[(g.league_a == lg) | (g.league_b == lg)]
        out[lg] = {"factor": round(f, 6), "log_se": round(se, 4), "wls": round(wls[lg], 6),
                   "first_pass": round(first[lg], 6),
                   "solved": rank.get(lg, 0),
                   "paths": [[c, round(sh, 4), round(v, 6)] for c, sh, v in paths.get(lg, [])],
                   "players": int(per.get(lg, 0)),
                   "nhl_direct": direct_n.get(lg, 0), "links": int(len(lg_pairs)),
                   "from_ep": int(pairs[(pairs.player.str[0] == "e") &
                                        ((pairs.league_a == lg) | (pairs.league_b == lg))].shape[0]),
                   "from_hockeydb": int(pairs[(pairs.player.str[0] == "h") &
                                              ((pairs.league_a == lg) | (pairs.league_b == lg))].shape[0]),
                   "from_tables": int(pairs[(pairs.player.str[0] == "t") &
                                            ((pairs.league_a == lg) | (pairs.league_b == lg))].shape[0])}
    meta = {"seasons": [SEASONS.start, SEASONS.stop - 1], "min_gp": MIN_GP, "min_pair": MIN_PAIR,
            "min_league": MIN_LEAGUE, "pairs": int(len(g)), "player_seasons": int(pairs[["player", "season"]].drop_duplicates().shape[0])}
    (HERE / "factors.json").write_text(json.dumps({"meta": meta, "factors": out}, indent=1), encoding="utf-8")
    write_report(meta, out, g)
    for lg, r in out.items():
        print(f"{lg:15} {r['factor']:.4f} +/-{r['log_se']:.3f} (wls {r['wls']:.4f})  players {r['players']:6d}  NHL direct {r['nhl_direct']:5d}")
    dropped = sorted(set(per.index) - set(factors))
    print("dropped (too few players or no NHL path):", dropped)


def write_report(meta, out, g):
    # The factors in the scripts now, to compare with (fwd-d0 has the full table)
    old = load_script("Forwards/fwd-d0.py").LEAGUE_FACTORS
    lines = ["# Network NHLe factors", "",
             "Generated by `training/network.py` — don't edit by hand.", "",
             f"Seasons {meta['seasons'][0]}-{str(meta['seasons'][1] + 1)[2:]} (regular season), "
             f"{meta['player_seasons']:,} player-seasons in two or more leagues, {meta['pairs']} league pairs "
             f"with {meta['min_pair']}+ shared players ({meta['min_gp']}+ GP in each league).",
             "Factor = NHL points per league point: the share-weighted sum of all the league's paths, each "
             "connector PPG / league PPG × the connector's NHLe (the NHL itself is the direct path, at 1). "
             "The connectors' values come from a first pass (`1st`) that solves leagues one at a time outward "
             "from the NHL, next the one sharing the most players with those already solved (`#` is the order); "
             "the factor then uses every connection a league has. `WLS` is a log-space least-squares solve over the same "
             "pairs, as a cross-check. `±` is the bootstrap standard error of the log factor (0.05 ≈ ±5%). "
             "`Players` counts shared player-seasons across the league's pairs, `NHL direct` those shared with the NHL, "
             "`from sites` those that came from league-website tables, `from EP` from Elite Prospects tables and "
             "`from HockeyDB` from HockeyDB rosters, "
             "rather than NHL career histories.", "",
             "| # | League | Factor | ± | 1st | WLS | Old | Players | NHL direct | Pairs | From sites | From EP | From HockeyDB | Biggest paths (share → NHLe via it) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for lg, r in out.items():
        o = f"{old[lg]:.3f}" if lg in old else "–"
        top = ", ".join(f"{c} {sh:.0%} → {v:.3f}" for c, sh, v in r["paths"][:4])
        lines.append(f"| {r['solved']} | {lg} | {r['factor']:.3f} | {r['log_se']:.3f} | {r['first_pass']:.3f} | {r['wls']:.3f} | {o} | {r['players']:,} | "
                     f"{r['nhl_direct']:,} | {r['links']} | {r['from_tables']:,} | {r['from_ep']:,} | {r['from_hockeydb']:,} | {top} |")
    lines += ["", "Re-run with `python training/scrape.py`, `python training/network.py`, then `python training/fit.py --write`.", ""]
    (HERE / "NETWORK.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
