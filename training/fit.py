"""
Refit every NHLe model (forwards and defense, D-1 through D+5) from the data
scraped by training/scrape.py and the league factors from training/network.py.

  python training/fit.py           # fit, write training/REPORT.md + coefficients.json
  python training/fit.py --write   # also regenerate the model scripts from
                                   # training/model_template.py.in

Sample:   NHL draft picks from CLASSES, by draft-listed position (C/L/R = F, D = D).
Factors:  network NHLe (training/factors.json): every league's points per NHL
          point, solved jointly from same-season crossovers (training/NETWORK.md).
Feature:  NHLe pts/82 for the season N years from the draft: regular season +
          playoffs in every league with a factor. A player-season is used only
          if at least half of its league games are in leagues with a factor
          (and 10+ of them).
WJC:      The World Juniors are not pooled into pts/82, where 7 games would be
          drowned by 60 league games. Each tournament a model uses gets its own
          two terms, fit with the rest: PLAYED (made the team) and PER_PTS82
          (times that tournament's NHLe pts/82). A light ridge (WJC_RIDGE) keeps
          the rare cases (17-year-olds at the WJC-20) from running away.
Outcomes: NHL player = 200+ NHL regular-season games (through the latest season).
          Star = NHL player whose peak GAR (mean of best 3 seasons) is in the top
          15% of NHL skaters at the position with 200+ GP and a season since
          2008-09. GAR is Neil Paine's (data/neil-paine, MIT) through 2024-25;
          later seasons are estimated from MoneyPuck stats (see gar_seasons).
Model:    log-odds = INTERCEPT + COEF * pts82 + sum over WJCs (PLAYED + PER_PTS82 * wjc82),
          for each outcome separately.
"""

import importlib.util
import json
import re
import sys
import unicodedata
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "data"
GAR_CSV = ROOT / "data" / "neil-paine" / "nhl_gar_historical.csv"
PAINE_LAST_SEASON = 2024       # start year of the last season in Paine's file (2024-25)
ESTIMATE_TRAIN_LAST = 2020     # MoneyPuck->GAR estimate trains through 2020-21

CLASSES = range(2006, 2019)    # 2019+ haven't had time to reach 200 GP / a GAR peak
MIN_GP = 10
MIN_COVERAGE = 0.5
STAR_PCT = 0.85
WJC_RIDGE = 1.0                # L2 penalty on the WJC terms: a N(0, 1) prior on PLAYED and
WJC_SCALE = 20.0               # on PER_PTS82 per 20 NHLe pts/82 (about the spread of WJC scoring)
MAX_LOG_SE = 0.25              # leagues whose factor is less certain than ~±28% stay out of the scripts

# Scripts per (position, season relative to draft year) and the tournaments
# each one uses.
MODELS = [
    ("F", -1, "Forwards/fwd-d-1.py", {"wjc-18", "wjc-20"}),
    ("F",  0, "Forwards/fwd-d0.py",  {"wjc-18", "wjc-20"}),
    ("F",  1, "Forwards/fwd-d1.py",  {"wjc-20"}),
    ("F",  2, "Forwards/fwd-d2.py",  {"wjc-20"}),
    ("F",  3, "Forwards/fwd-d3.py",  set()),
    ("F",  4, "Forwards/fwd-d4.py",  set()),
    ("F",  5, "Forwards/fwd-d5.py",  set()),
    ("D", -1, "Defense/d-d-1.py",    {"wjc-18", "wjc-20"}),
    ("D",  0, "Defense/d-d0.py",     {"wjc-18", "wjc-20"}),
    ("D",  1, "Defense/d-d1.py",     {"wjc-20"}),
    ("D",  2, "Defense/d-d2.py",     {"wjc-20"}),
    ("D",  3, "Defense/d-d3.py",     set()),
    ("D",  4, "Defense/d-d4.py",     set()),
    ("D",  5, "Defense/d-d5.py",     set()),
]
TOURNAMENTS = {"wjc-20", "wjc-18"}

# NHL API league names -> factor keys. Names changed over the years (Sweden ->
# SHL, CzRep -> Czechia, EBEL -> ICEHL...) and the NCAA is listed by conference.
# training/network.py solves a factor for every key here that has enough players.
LEAGUE_ALIASES = {
    "NHL": "nhl", "AHL": "ahl", "ECHL": "echl",
    "OHL": "ohl", "WHL": "whl", "QMJHL": "qmjhl", "USHL": "ushl", "NAHL": "nahl",
    "BCHL": "bchl", "AJHL": "ajhl", "SJHL": "sjhl", "MJHL": "mjhl", "OJHL": "ojhl",
    "CCHL": "cchl", "MJAHL": "mjahl", "MHL-Canada": "mjahl", "KIJHL": "kijhl", "VIJHL": "vijhl",
    "NCAA": "ncaa", "WCHA": "ncaa", "H-East": "ncaa", "CCHA": "ncaa", "ECAC": "ncaa",
    "NCHC": "ncaa", "Big Ten": "ncaa", "Big-10": "ncaa", "Atlantic": "ncaa", "AHA": "ncaa", "CHA": "ncaa",
    "USports": "usports", "U SPORTS": "usports", "CIS": "usports",
    "SHL": "shl", "Sweden": "shl",
    "HockeyAllsvenskan": "allsvenskan", "Allsvenskan": "allsvenskan", "Sweden-2": "allsvenskan",
    "HockeyEttan": "hockeyettan", "Sweden-3": "hockeyettan", "Division 1": "hockeyettan",
    "J20 Nationell": "u20-nationell", "J20 SuperElit": "u20-nationell", "Swe-Jr.": "u20-nationell",
    "J18 Nationell": "u18-nationell", "J18 Allsvenskan": "u18-nationell",
    "Liiga": "liiga", "Finland": "liiga", "SM-liiga": "liiga",
    "Mestis": "mestis", "Finland-2": "mestis",
    "Suomi-sarja": "suomi-sarja", "Finland-3": "suomi-sarja",
    "U20 SM-sarja": "u20-sm-sarja", "U20 SM-liiga": "u20-sm-sarja", "Fin-Jr.": "u20-sm-sarja",
    "U18 SM-sarja": "u18-sm-sarja", "Fin-U18": "u18-sm-sarja",
    "KHL": "khl", "Rus-KHL": "khl", "Russia": "khl",
    "VHL": "vhl", "Russia2": "vhl", "Russia-2": "vhl",
    "MHL": "mhl", "Russia-Jr.": "mhl", "Rus-MHL": "mhl",
    "Czechia": "czechia", "CzRep": "czechia", "Czech": "czechia",
    "Czechia2": "czechia2", "CzRep-2": "czechia2", "Czech2": "czechia2",
    "Czechia3": "czechia3", "Czech3": "czechia3",
    "Czechia U20": "czechia-u20", "CzRep-Jr.": "czechia-u20", "Czech U20": "czechia-u20",
    "Czechia U18": "czechia-u18", "Czech U18": "czechia-u18",
    "Slovakia": "slovakia", "Slovakia2": "slovakia2", "Slovak-2": "slovakia2",
    "Slovakia U20": "slovakia-u20", "Slovak-Jr.": "slovakia-u20",
    "NL": "nl", "NLA": "nl", "Swiss": "nl",
    "SL": "sl", "NLB": "sl", "Swiss-2": "sl",
    "MyHL": "myhl", "Swiss-3": "myhl",
    "U20-Elit": "swiss-u20", "Swiss U20": "swiss-u20", "Swiss-Jr.": "swiss-u20",
    "DEL": "del", "Germany": "del",
    "DEL2": "del2", "Germany2": "del2", "German-2": "del2",
    "Oberliga": "oberliga", "Germany3": "oberliga",
    "DNL": "dnl", "Ger-Jr.": "dnl", "DNL U20": "dnl",
    "ICEHL": "icehl", "EBEL": "icehl", "Austria": "icehl",
    "AlpsHL": "alpshl",
    "Norway": "norway", "Denmark": "denmark", "Metal Ligaen": "denmark",
    "EIHL": "eihl", "France": "france", "Ligue Magnus": "france",
    "Italy": "italy", "Poland": "poland", "Belarus": "belarus", "Latvia": "latvia",
    "Kazakhstan": "kazakhstan", "Erste Liga": "erste-liga",
    "Czech U20": "czechia-u20", "Czech U18": "czechia-u18", "Fin-U18": "u18-sm-sarja",
    "Swe-U18": "u18-nationell", "Swiss-Jr.": "swiss-u20", "OPJHL": "ojhl", "QAAA": "qc-u18-aaa",
    "CHL": "central-hl", "CzRep-3": "czechia3", "Swiss-2": "sl",
    "WJC-20": "wjc-20", "WJC-A": "wjc-20",
    "WJC-18": "wjc-18", "WJ18-A": "wjc-18",
}

# Short events that aren't a league season; left out of the coverage check.
EVENTS = {"WC", "WC-A", "WCup", "Olympics", "OG", "OGQ", "Champions HL", "Spengler Cup",
          "M-Cup", "Exhib.", "International", "Ivan Hlinka Memorial", "Hlinka Gretzky Cup",
          "WHC-17", "WJAC-19", "JCWC", "Tel-Cup", "Continental Cup", "WJC-B", "WJ18-B",
          "EHT", "TV-Pucken", "U17-Dev", "WJC-20 D1A", "WJC-18 D1A", "WC D1A",
          "U-18", "U-17"}   # the USNTDP schedule outside the USHL: its USHL games rate the season


def load_script(rel_path):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_factors():
    """Network factors precise enough to use (training/factors.json)."""
    out = json.loads((HERE / "factors.json").read_text(encoding="utf-8"))["factors"]
    return {k: v["factor"] for k, v in out.items() if v["log_se"] <= MAX_LOG_SE}


def norm_name(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return s.lower().replace(".", "").replace("-", " ").strip()


# ── NHL outcomes ──────────────────────────────────────────────────────────────

GAR_FEATURES = ["I_F_goals", "I_F_primaryAssists", "I_F_secondaryAssists", "xgd", "gd",
                "toi", "games_played", "I_F_xGoals", "penalties", "I_F_takeaways",
                "I_F_giveaways", "shotsBlockedByPlayer"]


def gar_seasons():
    """One GAR value per (NHL playerId, season start year), plus career GP.

    Neil Paine's GAR where it exists (through 2024-25). Seasons after that use
    a per-position linear fit of his GAR on MoneyPuck season stats; trained on
    2008-09 to 2020-21, it explains 95% (F) / 94% (D) of the variance on
    2021-22 to 2024-25. Its level drifts with league scoring (+16% for F on
    those seasons), so estimated seasons are rescaled to his league-wide GAR
    per team-game.
    """
    pg = pd.read_csv(GAR_CSV, encoding="latin-1", low_memory=False)
    pg = pg[(pg.lg_ID == "NHL") & pg.maj_pos.isin(["F", "D"])]
    pg = pg.groupby(["player_ID", "player_name", "year_ID", "maj_pos"], as_index=False).agg(
        gar=("GAR", "sum"), gp=("GP", "sum"))
    pg["season"] = pg.year_ID - 1
    pg = pg[pg.season <= PAINE_LAST_SEASON]
    pg["key"] = pg.player_name.map(norm_name)

    mp = pd.read_csv(DATA / "moneypuck.csv")
    mp = mp[mp.situation == "all"].copy()
    mp["key"] = mp.name.map(norm_name)
    mp["xgd"] = mp.OnIce_F_xGoals - mp.OnIce_A_xGoals
    mp["gd"] = mp.OnIce_F_goals - mp.OnIce_A_goals
    mp["toi"] = mp.icetime / 3600
    mp["pos"] = np.where(mp.position == "D", "D", "F")

    both = pg.merge(mp, on=["key", "season"])
    both = both[(both.gp - both.games_played).abs() <= 3]

    mp["est_gar"] = np.nan
    for pos in ("F", "D"):
        train = both[(both.maj_pos == pos) & (both.season <= ESTIMATE_TRAIN_LAST)]
        X = np.column_stack([np.ones(len(train))] + [train[f].to_numpy(float) for f in GAR_FEATURES])
        beta, *_ = np.linalg.lstsq(X, train.gar.to_numpy(float), rcond=None)
        rows = mp.pos == pos
        Xp = np.column_stack([np.ones(rows.sum())] + [mp.loc[rows, f].to_numpy(float) for f in GAR_FEATURES])
        mp.loc[rows, "est_gar"] = Xp @ beta

    # Paine player -> NHL id, from the seasons that matched
    xwalk = both.groupby("player_ID").playerId.agg(lambda s: s.mode().iloc[0])
    pg["playerId"] = pg.player_ID.map(xwalk)
    pg_rows = pg.dropna(subset=["playerId"])[["playerId", "season", "gar", "gp", "maj_pos"]]
    pg_rows = pg_rows.rename(columns={"maj_pos": "pos"})
    # GAR is measured against replacement level, so the league total per
    # team-game barely moves, while raw stats drift with scoring. Rescale each
    # estimated season to the GAR per team-game of the last real seasons.
    def team_games(season):
        teams = mp[(mp.season == season) & (mp.team.str.len() == 3)].team.nunique()
        return teams * 82
    ref = range(PAINE_LAST_SEASON - 3, PAINE_LAST_SEASON + 1)
    rate = pg[pg.season.isin(ref)].gar.sum() / sum(team_games(s) for s in ref)
    mp_rows = mp[mp.season > PAINE_LAST_SEASON][["playerId", "season", "est_gar", "games_played", "pos"]]
    mp_rows = mp_rows.rename(columns={"est_gar": "gar", "games_played": "gp"})
    for s in mp_rows.season.unique():
        rows = mp_rows.season == s
        mp_rows.loc[rows, "gar"] *= rate * team_games(s) / mp_rows.loc[rows, "gar"].sum()
    # Matched Paine seasons win; unmatched MP seasons up to his last fill the gaps
    gap = mp[mp.season <= PAINE_LAST_SEASON][["playerId", "season", "est_gar", "games_played", "pos"]]
    gap = gap.rename(columns={"est_gar": "gar", "games_played": "gp"})
    gap = gap.merge(pg_rows[["playerId", "season"]], how="left", indicator=True)
    gap = gap[gap._merge == "left_only"].drop(columns="_merge")

    seasons = pd.concat([pg_rows, gap, mp_rows], ignore_index=True)
    seasons["playerId"] = seasons.playerId.astype(int)
    return seasons


def star_thresholds(seasons):
    career = seasons.groupby("playerId").agg(
        gp=("gp", "sum"), last=("season", "max"), pos=("pos", lambda s: s.mode().iloc[0]),
        peak=("gar", lambda g: g.nlargest(3).sum() / 3))
    pool = career[(career.gp >= 200) & (career["last"] >= 2008)]
    return {pos: float(pool[pool.pos == pos].peak.quantile(STAR_PCT)) for pos in ("F", "D")}, career


# ── Prospect features ─────────────────────────────────────────────────────────

def player_seasons():
    """Draft picks, their stints (playerId, season end year, kind, league key, gp,
    pts) in every league with a key, and their league games for the coverage check."""
    draft = pd.read_csv(DATA / "draft.csv").dropna(subset=["playerId"])
    draft["playerId"] = draft.playerId.astype(int)
    draft = draft[draft.draftYear.isin(CLASSES)].copy()
    draft["pos"] = np.where(draft.position == "D", "D", "F")
    ids = set(draft.playerId)

    stints, cover, nhl_gp = [], [], {}
    with (DATA / "landings.jsonl").open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pid = rec["playerId"]
            if pid not in ids:
                continue
            nhl_gp[pid] = 0
            for s in rec["seasonTotals"]:
                end = int(str(s["season"])[4:])
                league = s.get("leagueAbbrev")
                gp, pts = s.get("gamesPlayed") or 0, s.get("points") or 0
                key = LEAGUE_ALIASES.get(league)
                reg = s["gameTypeId"] == 2
                if league == "NHL" and reg:
                    nhl_gp[pid] += gp
                if reg and league not in EVENTS and key not in TOURNAMENTS:
                    cover.append((pid, end, key, gp))
                if key is None or gp <= 0 or (key in TOURNAMENTS and not reg):
                    continue
                kind = key if key in TOURNAMENTS else "reg" if reg else "po"
                stints.append((pid, end, kind, key, gp, pts))
    draft["nhl_gp"] = draft.playerId.map(nhl_gp).fillna(0)
    stints = pd.DataFrame(stints, columns=["playerId", "end", "kind", "league", "gp", "pts"])
    cover = pd.DataFrame(cover, columns=["playerId", "end", "league", "gp"])
    return draft, stints, cover


def model_sample(d, stints, cover, rel, tourn, factors):
    """One row per pick with a usable season: club pts82 (x), reg GP, and per
    tournament its raw points and games."""
    st = stints.merge(d[["playerId", "draftYear"]])
    st = st[st.end - st.draftYear == rel]
    club = st[st.kind.isin(["reg", "po"]) & st.league.isin(factors)]
    agg = club.assign(nhle=club.pts * club.league.map(factors)).groupby("playerId").agg(
        nhle=("nhle", "sum"), gp=("gp", "sum"))
    reg_gp = club[club.kind == "reg"].groupby("playerId").gp.sum().rename("reg_gp")
    cv = cover.merge(d[["playerId", "draftYear"]])
    cv = cv[cv.end - cv.draftYear == rel]
    all_gp = cv.groupby("playerId").gp.sum().rename("all_gp")
    rated_gp = cv[cv.league.isin(factors)].groupby("playerId").gp.sum().rename("rated_gp")
    sample = d.set_index("playerId").join(agg, how="inner").join(reg_gp).join(all_gp).join(rated_gp)
    sample = sample[(sample.reg_gp >= MIN_GP) & (sample.rated_gp >= MIN_COVERAGE * sample.all_gp)].copy()
    sample["x"] = sample.nhle / sample.gp * 82
    for t in sorted(tourn):
        w = st[st.kind == t].groupby("playerId")[["pts", "gp"]].sum()
        sample[f"{t}_pts"] = w.pts.reindex(sample.index).fillna(0)
        sample[f"{t}_gp"] = w.gp.reindex(sample.index).fillna(0)
    return sample, int(cv.playerId.nunique())


def design(sample, tourn, factors):
    """Columns: 1, pts82, then per tournament PLAYED and NHLe pts/82 there."""
    cols = [np.ones(len(sample)), sample.x.to_numpy()]
    for t in sorted(tourn):
        gp = sample[f"{t}_gp"].to_numpy()
        played = (gp > 0).astype(float)
        w82 = np.divide(sample[f"{t}_pts"].to_numpy() * factors[t] * 82, gp, out=np.zeros(len(gp)), where=gp > 0)
        cols += [played, w82]
    return np.column_stack(cols)


def pooled_x(sample, tourn, factors):
    """The old way: World Juniors games and NHLe points pooled into pts/82."""
    pts, gp = sample.nhle.to_numpy(float).copy(), sample.gp.to_numpy(float).copy()
    for t in sorted(tourn):
        if t in factors:
            pts += sample[f"{t}_pts"].to_numpy() * factors[t]
            gp += sample[f"{t}_gp"].to_numpy()
    return pts / gp * 82


# ── Logistic regression ───────────────────────────────────────────────────────

def fit_logistic(X, y, ridge=None, iters=100):
    """IRLS for log-odds = X @ beta, with an L2 penalty per column (a whisper on
    the intercept and pts/82 keeps it stable)."""
    X = np.atleast_2d(X)
    if X.shape[0] != len(y):
        X = X.T
    pen = np.full(X.shape[1], 1e-6) if ridge is None else np.asarray(ridge, float)
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ beta)))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + np.diag(pen)
        step = np.linalg.solve(H, X.T @ (y - p) - pen * beta)
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return beta


def ridge_for(tourn):
    return [1e-6, 1e-6] + [WJC_RIDGE, WJC_RIDGE * WJC_SCALE ** 2] * len(tourn)


def auc(score, y):
    ranks = pd.Series(score).rank().to_numpy()
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return (ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else float("nan")


def log_loss(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cv_predict(X, y, classes, ridge):
    """Leave-one-draft-class-out: each class predicted by a fit on the others."""
    out = np.empty(len(y))
    for c in np.unique(classes):
        held = classes == c
        out[held] = sigmoid(X[held] @ fit_logistic(X[~held], y[~held], ridge))
    return out


def old_predict(old, prefix, sample, tourn):
    """What the script had before this fit, with its own factors."""
    factors = old.LEAGUE_FACTORS
    a, b = getattr(old, f"{prefix}_INTERCEPT"), getattr(old, f"{prefix}_COEF")
    terms = getattr(old, f"{prefix}_WJC", None)
    if terms is None:     # the older pooled scripts (their WJC team-share adjustment needs team totals)
        x = old_x(sample, tourn, factors, pooled=True)
        return sigmoid(a + b * x)
    X = design(sample.assign(x=old_x(sample, tourn, factors, pooled=False)), tourn, factors)
    beta = [a, b] + [v for t in sorted(tourn) for v in terms.get(t, (0.0, 0.0))]
    return sigmoid(X @ np.array(beta))


def old_x(sample, tourn, factors, pooled):
    club = sample.club_stints
    pts = np.array([sum(p * factors.get(lg, 0) for lg, p, g in rows) for rows in club])
    gp = np.array([sum(g for lg, p, g in rows if lg in factors) for rows in club], float)
    if pooled:
        for t in sorted(tourn):
            if t in factors:
                pts = pts + sample[f"{t}_pts"].to_numpy() * factors[t]
                gp = gp + sample[f"{t}_gp"].to_numpy()
    return np.divide(pts * 82, gp, out=np.zeros(len(gp)), where=gp > 0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    write = "--write" in sys.argv
    factors = load_factors()

    seasons = gar_seasons()
    thresholds, career = star_thresholds(seasons)
    draft, stints, cover = player_seasons()
    draft = draft.merge(career[["peak"]], left_on="playerId", right_index=True, how="left")
    draft["nhler"] = (draft.nhl_gp >= 200).astype(int)
    draft["star"] = ((draft.nhler == 1) &
                     (draft.peak >= draft.pos.map(thresholds))).astype(int)

    results = []
    for pos, rel, script, tourn in MODELS:
        d = draft[draft.pos == pos]
        sample, played = model_sample(d, stints, cover, rel, tourn, factors)
        # Each pick's club stints, so the old scripts can be scored with their own factors
        st = stints.merge(d[["playerId", "draftYear"]])
        st = st[(st.end - st.draftYear == rel) & st.kind.isin(["reg", "po"])]
        rows = st.groupby("playerId").apply(lambda g: list(zip(g.league, g.pts, g.gp)), include_groups=False)
        sample["club_stints"] = rows.reindex(sample.index).apply(lambda r: r if isinstance(r, list) else [])
        X = design(sample, tourn, factors)
        Xp = np.column_stack([np.ones(len(sample)), pooled_x(sample, tourn, factors)])
        classes = sample.draftYear.to_numpy()
        ridge = ridge_for(tourn)
        res = {"pos": pos, "rel": rel, "script": script, "n": len(sample), "played": played,
               "tournaments": sorted(tourn),
               "wjc_played": {t: int((sample[f"{t}_gp"] > 0).sum()) for t in sorted(tourn)}}
        old = load_script(script) if (ROOT / script).exists() else None
        for target, prefix in (("nhler", "NHLER"), ("star", "STAR")):
            y = sample[target].to_numpy()
            beta = fit_logistic(X, y, ridge)
            p = sigmoid(X @ beta)
            m = {"intercept": round(float(beta[0]), 6), "coef": round(float(beta[1]), 6),
                 "wjc": {t: [round(float(beta[2 + 2 * i]), 6), round(float(beta[3 + 2 * i]), 6)]
                         for i, t in enumerate(sorted(tourn))},
                 "rate": float(y.mean()), "auc": float(auc(X @ beta, y)),
                 "log_loss": log_loss(p, y), "cv_log_loss": log_loss(cv_predict(X, y, classes, ridge), y)}
            # The same data with the World Juniors pooled into pts/82, as before
            m["pooled_cv_log_loss"] = log_loss(cv_predict(Xp, y, classes, None), y)
            if old is not None:
                m["old_log_loss"] = log_loss(old_predict(old, prefix, sample, tourn), y)
            res[target] = m
        res["pts82_median"] = float(np.median(sample.x))
        results.append(res)
        print(f"{pos} D{rel:+d}  n={res['n']:4d}/{played}  NHLer {res['nhler']['rate']:.1%} "
              f"(AUC {res['nhler']['auc']:.3f}, CV {res['nhler']['cv_log_loss']:.4f} vs pooled "
              f"{res['nhler']['pooled_cv_log_loss']:.4f})  star {res['star']['rate']:.1%} "
              f"(AUC {res['star']['auc']:.3f}, CV {res['star']['cv_log_loss']:.4f} vs pooled "
              f"{res['star']['pooled_cv_log_loss']:.4f})  WJC {res['nhler']['wjc']}")

    examples = star_examples(career, thresholds)
    out = {"classes": [CLASSES.start, CLASSES.stop - 1], "star_thresholds": thresholds,
           "star_examples": examples, "factors": factors, "models": results}
    (HERE / "coefficients.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    write_report(out, draft)
    if write:
        for res in results:
            write_script(res, factors, examples[res["pos"]])
    print("star thresholds:", {k: round(v, 1) for k, v in thresholds.items()})


def star_examples(career, thresholds):
    names = pd.read_csv(DATA / "moneypuck.csv")[["playerId", "name"]].drop_duplicates("playerId")
    names = names.set_index("playerId").name
    out = {}
    for pos, t in thresholds.items():
        c = career[(career.pos == pos) & (career.gp >= 400) & (career["last"] >= 2022)]
        near = c.iloc[(c.peak - t).abs().argsort()[:5]]
        out[pos] = [names.get(pid, str(pid)) for pid in near.index]
    return out


def write_script(res, factors, examples):
    """Render the model script from training/model_template.py.in."""
    pos, rel, tourn = res["pos"], res["rel"], res["tournaments"]
    label = "Draft Year" if rel == 0 else f"D{rel:+d}"
    group = "Forward" if pos == "F" else "Defenseman"
    age = 18 + rel
    title = f"NHLe {group} {label} Model (age ~{age})"
    season = {-1: "the season before the draft year", 0: "the draft year season"}.get(
        rel, f"{rel} season{'s' if rel > 1 else ''} after the draft year")
    known = rel >= 2
    known_doc = ("\nIf the player has ALREADY reached 200 GP or the star threshold, that\n"
                 "outcome is reported as 100%.\n") if known else ""
    if tourn:
        names = " and ".join(t.upper() for t in tourn)
        wjc_doc = (f"The World Juniors ({names}) are their own fitted term, not pooled into\n"
                   "pts/82: making the team and scoring there move the odds on top of the\n"
                   "league season (NHLER_WJC / STAR_WJC), so a big tournament counts for more\n"
                   "than its 7 games would.")
    else:
        wjc_doc = f"No World Juniors: players are {age}, no longer eligible."
    keep = {k: v for k, v in factors.items() if tourn or k not in TOURNAMENTS}
    width = max(len(k) for k in keep) + 3
    factor_lines = "\n".join(f"    {json.dumps(k) + ':':<{width}} {v:.6f}," for k, v in
                             sorted(keep.items(), key=lambda kv: -kv[1]))

    def wjc_dict(m):
        return "{" + ", ".join(f'"{t}": ({a:.6f}, {b:.6f})' for t, (a, b) in m["wjc"].items()) + "}"

    text = Template((HERE / "model_template.py.in").read_text(encoding="utf-8")).substitute(
        group_title=group, label=label, age=age, underline="=" * len(title), season_desc=season,
        group_plural="forwards" if pos == "F" else "defensemen", known_doc=known_doc, wjc_doc=wjc_doc,
        classes=f"{CLASSES.start}-{CLASSES.stop - 1}", script_name=Path(res["script"]).name,
        nhler_intercept=f"{res['nhler']['intercept']: .6f}", nhler_coef=f"{res['nhler']['coef']: .6f}",
        nhler_wjc=wjc_dict(res["nhler"]),
        star_intercept=f"{res['star']['intercept']: .6f}", star_coef=f"{res['star']['coef']: .6f}",
        star_wjc=wjc_dict(res["star"]),
        factors=factor_lines, tournaments=repr(tuple(tourn)), known=known, examples=", ".join(examples))
    (ROOT / res["script"]).write_text(text, encoding="utf-8", newline="\n")
    print("wrote", res["script"])


def write_report(out, draft):
    t = out["star_thresholds"]
    lines = [
        "# NHLe model fit",
        "",
        "Generated by `training/fit.py` — don't edit by hand.",
        "",
        f"Draft classes {out['classes'][0]}–{out['classes'][1]}: {len(draft)} picks with an NHL id "
        f"({(draft.pos == 'F').sum()} forwards, {(draft.pos == 'D').sum()} defensemen). "
        f"League factors: network NHLe, {len(out['factors'])} leagues ([NETWORK.md](NETWORK.md)).",
        f"NHL player = 200+ NHL games. Star = NHL player with peak GAR (Neil Paine's; mean of best 3 seasons) at or above "
        f"**{t['F']:.1f}** for forwards and **{t['D']:.1f}** for defensemen (85th percentile of 200+ GP skaters).",
        f"Star examples near the line — F: {', '.join(out['star_examples']['F'])}; "
        f"D: {', '.join(out['star_examples']['D'])}.",
        "",
        "`n` is the number of picks with a usable season (10+ GP, at least half of it in leagues with an NHLe "
        "factor) out of those who played that season at all. `x` is NHLe pts/82 in league play (regular "
        "season + playoffs). Log loss (lower is better) is on the same sample four ways: the fit; "
        "leave-one-draft-class-out cross-validation; CV with the World Juniors pooled into pts/82 instead "
        "of their own term; and the coefficients and factors the script had before this fit.",
        "",
        "| Model | n / played | NHLer rate | NHLer coefs | AUC | log loss fit / CV / pooled CV / old | Star rate | Star coefs | AUC | log loss fit / CV / pooled CV / old |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in out["models"]:
        cells = [f"{'Fwd' if r['pos'] == 'F' else 'Def'} D{r['rel']:+d}".replace("D+0", "DY"),
                 f"{r['n']} / {r['played']}"]
        for target in ("nhler", "star"):
            m = r[target]
            old = f"{m['old_log_loss']:.3f}" if "old_log_loss" in m else "–"
            cells += [f"{m['rate']:.1%}", f"{m['intercept']:.3f} + {m['coef']:.4f}·x",
                      f"{m['auc']:.3f}",
                      f"{m['log_loss']:.3f} / {m['cv_log_loss']:.3f} / {m['pooled_cv_log_loss']:.3f} / {old}"]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "## World Juniors terms",
        "",
        "Log-odds added for a player who played the tournament: `played + per_pts82 × (his NHLe pts/82 there)`. "
        "`At 0 / 20 / 40` is the total for a tournament at that NHLe pts/82 (WJC points × the WJC factor / games × 82). "
        f"Ridge {WJC_RIDGE} on these terms (a N(0, 1) prior on `played`, and on `per_pts82` per "
        f"{WJC_SCALE:.0f} pts/82), so a term with a handful of players stays near zero.",
        "",
        "| Model | WJC | Played | NHLer: played, per pts/82 | at 0 / 20 / 40 | Star: played, per pts/82 | at 0 / 20 / 40 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in out["models"]:
        for t in r["tournaments"]:
            cells = [f"{'Fwd' if r['pos'] == 'F' else 'Def'} D{r['rel']:+d}".replace("D+0", "DY"), t.upper(),
                     str(r["wjc_played"][t])]
            for target in ("nhler", "star"):
                a, b = r[target]["wjc"][t]
                cells += [f"{a:+.3f}, {b:+.4f}", " / ".join(f"{a + b * x:+.2f}" for x in (0, 20, 40))]
            lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Re-run with `python training/scrape.py`, `python training/network.py`, then "
              "`python training/fit.py --write`.", ""]
    (HERE / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
