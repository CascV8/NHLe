"""
Star outcomes from on-ice expected goals, scored with the xG model
(github.com/CascV8/xG-model).

  uv run --project ../xg-model python training/xg_star.py collect 2010 2025   # per-game cache
  uv run --project ../xg-model python training/xg_star.py seasons             # player-seasons + stars

For every regular-season game (shift charts start in 2010-11): each unblocked
5-on-5 shot is scored with the xG model and credited, for or against, to the
skaters on the ice for it (from the NHL's shift charts), and each skater's
5-on-5 time on ice is summed from the same charts.

A player-season's value has three parts, all at his usual workload over a full
season of 84 games:
  - what happens at 5-on-5 with him on the ice, against his own team without him:
        (on-ice net xG/60 - his team's net xG/60 with him off) x 5-on-5 hours per game
    (xGA counts as negative xGF; comparing with his team's off-ice rate takes out
    team quality; a traded player is compared with each team, over his minutes there).
    Half weight for forwards: theirs follows the line they're on more than for
    defensemen, and two seasons on a hot line shouldn't make a star
  - his own production: the xG of his own shots, every situation (power play
    included), per game, above the average regular at his position that season
  - his playmaking: his 5-on-5 primary assists per game (the NHL's scoring-rates
    report), above the average regular at his position that season.
Using per-game rates, not totals, means an injury doesn't count against him while
his role still does; the games floor (MIN_GP, scaled for short seasons) keeps
call-ups from projecting a few hot games. The star line is the top 10% of regulars
at his position each season (the Pareto split of "What's a point worth?" takes
about the top 22%, which is a good top-six forward or top-four defenseman rather
than a star).

A player is a star once the mean of his best 3 regular seasons is above the
line (each season measured against its own line; seasons he doesn't have count
as zero value), or straight away after one season 10+ above it. The season it
happens is when the website confirms him at 100%.

Writes training/data/xg_games/<season>/<game>.parquet (cache, not committed),
training/primary_assists.csv, training/xg_seasons.csv and training/stars.json.
"""

import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import puckscrape as ps

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data" / "xg_games"
MIN_GP = 40           # games (of an 82-game season; scaled for shorter ones) to be a regular
SEASON_GAMES = 84     # value is per full season, at the 2026-27 schedule's 84 games
STAR_PCT = 0.10       # the star line: the top 10% of regulars at his position each season
BEST_N = 3            # a star's best 3 seasons average above the line...
PEAK = 10.0           # ...or one season this far above it confirms him at once
FORWARD_ONICE = 0.5   # weight on a forward's on-ice term (defensemen 1)
MIN_SEASON_GAMES = 41 # a season under way counts once teams are halfway through (no 15-game hot starts)
log = logging.getLogger("xg_star")


# ── One game ──────────────────────────────────────────────────────────────────

def game_rows(game_id, goalies):
    """Per skater for one game: team, 5v5 TOI (seconds), on-ice 5v5 xGF and xGA."""
    from xg_model import predict

    pbp = ps.nhl.play_by_play(game_id)
    shifts = ps.nhl.shifts(game_id)
    if pbp.empty or shifts.empty:
        return pd.DataFrame()
    shots = predict.score(pbp)
    home, away = pbp.home_team.iloc[0], pbp.away_team.iloc[0]
    shifts = shifts[shifts.duration_seconds > 0]
    shifts = shifts.assign(goalie=shifts.player_id.isin(goalies), home=shifts.team_abbrev == home)

    toi, xgf, xga, team_of = {}, {}, {}, {}
    for pid, team in zip(shifts.player_id, shifts.team_abbrev):
        team_of[pid] = team
    for period, sp in shifts.groupby("period"):
        if period > 3:      # regular-season overtime is 3-on-3
            continue
        n = int(sp.end_seconds.max()) + 1
        sk = {True: np.zeros(n, int), False: np.zeros(n, int)}
        g = {True: np.zeros(n, int), False: np.zeros(n, int)}
        for s, e, gl, hm in zip(sp.start_seconds, sp.end_seconds, sp.goalie, sp.home):
            (g if gl else sk)[hm][int(s):int(e)] += 1
        five = (sk[True] == 5) & (sk[False] == 5) & (g[True] == 1) & (g[False] == 1)
        for pid, s, e, gl in zip(sp.player_id, sp.start_seconds, sp.end_seconds, sp.goalie):
            if not gl:
                toi[pid] = toi.get(pid, 0) + int(five[int(s):int(e)].sum())
        # 5-on-5 shots this period, credited to the skaters on for them (start < t <= end)
        ps_ = shots[(shots.period == period) & (shots.shooter_skaters == 5) & (shots.opp_skaters == 5)
                    & (shots.own_goalie_pulled == 0)]
        sk_shifts = sp[~sp.goalie]
        for t, team, x in zip(ps_.period_seconds, ps_.event_team, ps_.xg):
            on = sk_shifts[(sk_shifts.start_seconds < t) & (sk_shifts.end_seconds >= t)]
            for pid, pteam in zip(on.player_id, on.team_abbrev):
                d = xgf if pteam == team else xga
                d[pid] = d.get(pid, 0.0) + float(x)
    ixg = individual_xg(shots)
    rows = [(game_id, pid, team_of[pid], toi.get(pid, 0), xgf.get(pid, 0.0), xga.get(pid, 0.0), ixg.get(pid, 0.0))
            for pid in set(toi) | set(xgf) | set(xga)]
    return pd.DataFrame(rows, columns=["game_id", "player_id", "team", "toi5", "xgf", "xga", "ixg"])


def individual_xg(shots):
    """His own production: the xG of his unblocked shots, every situation (power play included)."""
    return shots.groupby("shooter_id").xg.sum().to_dict() if len(shots) else {}


def add_ixg(path):
    """Backfill individual xG into a game cached before it was recorded (play-by-play only)."""
    from xg_model import predict
    df = pd.read_parquet(path)
    ixg = individual_xg(predict.score(ps.nhl.play_by_play(int(path.stem))))
    df["ixg"] = df.player_id.map(ixg).fillna(0.0)
    df.to_parquet(path)


def collect(first, last, workers=4):
    from xg_model.collect import game_ids
    for y in range(first, last + 1):
        folder = CACHE / str(y)
        folder.mkdir(parents=True, exist_ok=True)
        stats = ps.nhl.goalie_stats(y)
        goalies = set(stats["player_id"].dropna().astype(int)) if "player_id" in stats else set()
        todo = [g for g in game_ids(y) if not (folder / f"{g}.parquet").exists()
                and not (folder / f"{g}.none").exists()]
        old = [p for p in folder.glob("*.parquet") if "ixg" not in pd.read_parquet(p, columns=None).columns]
        log.info("%d-%02d: %d games to do, %d to add individual xG to", y, (y + 1) % 100, len(todo), len(old))
        with ThreadPoolExecutor(workers) as pool:
            for i, f in enumerate(as_completed([pool.submit(add_ixg, p) for p in old]), 1):
                try:
                    f.result()
                except Exception as e:
                    log.warning("ixg: %s", str(e)[:120])
                if i % 200 == 0:
                    log.info("  ixg %d/%d", i, len(old))

        def one(g):
            df = game_rows(g, goalies)
            if df.empty:
                (folder / f"{g}.none").write_text("no shifts or play-by-play")
            else:
                df.to_parquet(folder / f"{g}.parquet")
            return len(df)

        with ThreadPoolExecutor(workers) as pool:
            futs = {pool.submit(one, g): g for g in todo}
            for i, f in enumerate(as_completed(futs), 1):
                try:
                    f.result()
                except Exception as e:
                    log.warning("game %s: %s", futs[f], str(e)[:120])
                if i % 200 == 0:
                    log.info("  %d/%d", i, len(todo))


# ── Seasons, the star line, and stars ─────────────────────────────────────────

POSITIONS = HERE / "positions.json"   # F / D by NHL id, committed


def positions(seasons=()):
    """F / D for every skater: the committed training/positions.json, the NHL landings when
    training/scrape.py has fetched them, and the NHL's season stats for anyone still missing in
    ``seasons``. Saved back, so the monthly job only looks up new players."""
    out = {int(k): v for k, v in json.loads(POSITIONS.read_text(encoding="utf-8")).items()} if POSITIONS.exists() else {}
    landings = HERE / "data" / "landings.jsonl"
    if landings.exists():
        with landings.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                p = r.get("position")
                if p and p != "G":
                    out.setdefault(r["playerId"], "D" if p == "D" else "F")
    for y in seasons:
        stats = ps.nhl.skater_stats(int(y))
        if "position_code" in stats:
            for pid, p in zip(stats.player_id, stats.position_code):
                if pid == pid and p and p != "G":
                    out.setdefault(int(pid), "D" if p == "D" else "F")
    POSITIONS.write_text(json.dumps({str(k): v for k, v in sorted(out.items())}), encoding="utf-8")
    return out


def pareto_cut(values, n_players):
    """Your Pareto split over the positive values: the top N whose share of the
    players plus share of the (positive) total comes closest to 1. Returns N."""
    v = np.sort(values[values > 0])[::-1]
    if not len(v):
        return 0
    share_players = np.arange(1, len(v) + 1) / n_players
    share_value = np.cumsum(v) / v.sum()
    return int(np.argmin(np.abs(share_players + share_value - 1))) + 1


TOTALS = HERE / "xg_totals.csv"   # per season, team and player, committed: the stars rebuild without the game cache


def totals():
    """Season totals per (season, team, player): games, 5-on-5 seconds, on-ice xGF/xGA, own xG,
    and the team's games. From the per-game cache for the seasons it has, the committed
    training/xg_totals.csv for the rest; the merged table is written back."""
    frames = []
    for folder in sorted(CACHE.iterdir()) if CACHE.exists() else []:
        files = sorted(folder.glob("*.parquet"))
        if not files:
            continue
        g = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        t = g.groupby(["team", "player_id"], as_index=False).agg(
            gp=("game_id", "nunique"), toi5=("toi5", "sum"), xgf=("xgf", "sum"), xga=("xga", "sum"), ixg=("ixg", "sum"))
        t = t.merge(g.groupby("team").game_id.nunique().rename("team_gp").reset_index(), on="team")
        frames.append(t.assign(season=int(folder.name)))
    fresh = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if TOTALS.exists():
        old = pd.read_csv(TOTALS)
        old = old[~old.season.isin(set(fresh.season))] if len(fresh) else old
        fresh = pd.concat([old, fresh], ignore_index=True)
    fresh = fresh.sort_values(["season", "team", "player_id"])[
        ["season", "team", "player_id", "gp", "team_gp", "toi5", "xgf", "xga", "ixg"]]
    fresh.round(4).to_csv(TOTALS, index=False)
    return fresh


ASSISTS = HERE / "primary_assists.csv"   # 5-on-5 primary assists by season and player, committed


def primary_assists(seasons, refresh=()):
    """5-on-5 primary assists and games per (season, player): the committed training/primary_assists.csv,
    with any of ``seasons`` it lacks (and every season in ``refresh``) fetched from the NHL and saved back."""
    old = pd.read_csv(ASSISTS) if ASSISTS.exists() else pd.DataFrame(columns=["season", "player_id", "games_played", "pa5"])
    fetch = sorted((set(seasons) - set(old.season)) | set(refresh))
    frames = [old[~old.season.isin(fetch)]]
    for y in fetch:
        df = ps.nhl.skater_stats(int(y), report="scoringRates")
        if len(df):
            frames.append(df[["player_id", "games_played", "primary_assists5v5"]].rename(
                columns={"primary_assists5v5": "pa5"}).assign(season=int(y)))
    out = pd.concat(frames, ignore_index=True).sort_values(["season", "player_id"])[["season", "player_id", "games_played", "pa5"]]
    out.to_csv(ASSISTS, index=False)
    return out


def seasons(refresh=()):
    pt = totals()
    known = positions()
    missing = sorted(set(pt[~pt.player_id.isin(known)].season))   # seasons with players we can't place yet
    pos = positions(missing) if missing else known
    # Team 5-on-5 totals: every shot and second is credited to five skaters a side.
    team = (pt.groupby(["season", "team"])[["toi5", "xgf", "xga"]].sum() / 5).add_prefix("team_")
    pt = pt.merge(team.reset_index(), on=["season", "team"])
    # His team with him off the ice, and his on-ice rate above it, per team (for traded players)
    off_toi = (pt.team_toi5 - pt.toi5).clip(lower=1)
    pt["off60"] = ((pt.team_xgf - pt.xgf) - (pt.team_xga - pt.xga)) / (off_toi / 3600)
    pt["rel_xg"] = (pt.xgf - pt.xga) - pt.off60 * pt.toi5 / 3600   # net xG above his team's off-ice rate, in his minutes
    s = pt.groupby(["season", "player_id"], as_index=False)[["gp", "toi5", "xgf", "xga", "rel_xg", "ixg"]].sum()
    s["pos"] = s.player_id.map(pos)
    s = s.dropna(subset=["pos"])
    s["toi5_min"] = s.toi5 / 60
    s["net60"] = (s.xgf - s.xga) / (s.toi5 / 3600).where(s.toi5 > 0)
    s["rel60"] = s.rel_xg / (s.toi5 / 3600).where(s.toi5 > 0)
    # Regulars: MIN_GP of an 82-game season, scaled to the season's length (48 in 2012-13, 56 in 2020-21...)
    team_games = pt.groupby("season").team_gp.max()
    s["min_gp"] = (MIN_GP * s.season.map(team_games) / 82).round()
    s["regular"] = (s.gp >= s.min_gp) & (s.toi5 > 0) & (s.season.map(team_games) >= MIN_SEASON_GAMES)
    s["ixg_pg"] = s.ixg / s.gp
    avg_ixg = s[s.regular].groupby(["season", "pos"]).ixg_pg.mean().rename("avg_ixg_pg")
    s = s.merge(avg_ixg.reset_index(), on=["season", "pos"], how="left")
    s["onice_value"] = s.rel60 * (s.toi5 / s.gp / 3600) * SEASON_GAMES
    s["own_value"] = (s.ixg_pg - s.avg_ixg_pg) * SEASON_GAMES
    pa = primary_assists(sorted(set(s.season)), refresh)
    s = s.merge(pa, on=["season", "player_id"], how="left")
    s["pa5_pg"] = (s.pa5 / s.games_played).fillna(0)
    avg_pa = s[s.regular].groupby(["season", "pos"]).pa5_pg.mean().rename("avg_pa5_pg")
    s = s.merge(avg_pa.reset_index(), on=["season", "pos"], how="left")
    s["play_value"] = (s.pa5_pg - s.avg_pa5_pg) * SEASON_GAMES
    s["value"] = s.onice_value * np.where(s.pos == "F", FORWARD_ONICE, 1.0) + s.own_value + s.play_value
    s["cut"], s["star_share"] = np.inf, np.nan
    for (y, p), grp in s[s.regular].groupby(["season", "pos"]):
        n = max(int(round(STAR_PCT * len(grp))), 1)
        cut = np.sort(grp.value.to_numpy())[::-1][n - 1]
        s.loc[(s.season == y) & (s.pos == p), ["cut", "star_share"]] = cut, n / len(grp)
    s["star_season"] = s.regular & (s.value >= s.cut)
    # Each regular season against its own line; a season he doesn't have counts as zero value
    s["above"] = np.where(s.regular, s.value - s.cut, np.nan)
    s = s.sort_values(["player_id", "season"])
    confirmed = {}
    for pid, grp in s.groupby("player_id"):
        seen = []
        for season, above, cut, reg in zip(grp.season, grp.above, grp.cut, grp.regular):
            if reg:
                seen.append(above)
            best = sorted(seen, reverse=True)[:BEST_N]
            best += [-c for c in [cut] * (BEST_N - len(best))] if np.isfinite(cut) else []
            if (reg and above >= PEAK) or (len(best) == BEST_N and np.mean(best) >= 0):
                confirmed[pid] = int(season)
                break
    s = s.drop(columns="toi5").round(4)
    s.to_csv(HERE / "xg_seasons.csv", index=False)
    stars = pd.Series(confirmed, dtype=int)
    (HERE / "stars.json").write_text(json.dumps({
        "about": "Season (start year) each skater was confirmed a star by on-ice and individual xG "
                 "and 5-on-5 primary assists (training/xg_star.py): his best 3 regular seasons average above that "
                 "season's star line (the top 10% of regulars at his position), or one season 10+ above it.",
        "stars": {str(int(k)): int(v) for k, v in stars.items()}}, indent=0), encoding="utf-8")
    share = s[s.regular].groupby(["pos"]).star_share.mean()
    print(f"{len(s)} player-seasons {s.season.min()}-{s.season.max()}; star line ~top {share['F']:.0%} of F, "
          f"{share['D']:.0%} of D; {s.star_season.sum()} seasons above the line; {len(stars)} stars confirmed")
    return s


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if sys.argv[1] == "collect":
        collect(int(sys.argv[2]), int(sys.argv[3]), *(int(a) for a in sys.argv[4:5]))
    elif sys.argv[1] == "seasons":
        seasons()
    elif sys.argv[1] == "update":   # the monthly job: this season's new games, then the stars
        today = pd.Timestamp.today()
        y = today.year if today.month >= 9 else today.year - 1
        collect(y, y, 2)
        seasons(refresh=[y])
    elif sys.argv[1] == "game":
        g = int(sys.argv[2])
        y = int(str(g)[:4])
        stats = ps.nhl.goalie_stats(y)
        print(game_rows(g, set(stats["player_id"].astype(int))).sort_values("toi5", ascending=False).to_string())
