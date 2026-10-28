"""
Refit every NHLe model (forwards and defense, D-1 through D+5) from the data
scraped by training/scrape.py and the league factors from training/network.py.

  python training/fit.py           # fit, write training/REPORT.md + coefficients.json
  python training/fit.py --write   # also regenerate the model scripts from
                                   # training/model_template.py.in

Sample:   NHL draft picks from CLASSES, by draft-listed position (C/L/R = F, D = D).
Seasons:  Relative to each player's FIRST ELIGIBLE draft (18 by September 15), not
          the year he was picked: an overage pick's age-20 season is his D+2, not
          his "draft year". About a fifth of picks are overagers.
Factors:  network NHLe (training/factors.json): every league's points per NHL
          point, solved jointly from same-season crossovers (training/NETWORK.md).
Feature:  NHLe pts/84 for the season N years from the draft: regular season +
          playoffs in every league with a factor, each playoff game counting as
          PO_WEIGHT regular-season games (python training/fit.py --po-sweep tests
          the weights and writes training/po_sweep.json). A player-season is used only
          if at least half of its league games are in leagues with a factor
          (and 10+ of them).
WJC:      The World Juniors are not pooled into pts/84, where 7 games would be
          drowned by 60 league games. Each tournament a model uses gets its own
          two terms, fit with the rest: PLAYED (made the team) and PER_PTS84
          (times that tournament's NHLe pts/84). The WJC-20 terms also depend on
          the player's age at the tournament (Jan 1): a 17-year-old making an
          under-20 team means more than a 19-year-old doing it. They're a line in
          age, fit jointly across D-1 to D+2 per position, so the rare 17-year-olds
          borrow from the trend instead of resting on a dozen players. A light
          ridge (WJC_RIDGE) keeps small terms near zero.
Outcomes: NHL player = 200+ NHL regular-season games (through the latest season).
          Star = NHL player with a star season by on-ice expected goals
          (training/xg_star.py, scored with the xG model): his on-ice net xG
          above replacement, at his average 5-on-5 minutes over 84 games,
          inside that season's Pareto group of regulars (40+ games) at his
          position (the split in "What's a point worth?"). training/stars.json
          lists them and when.
Model:    log-odds = INTERCEPT + COEF * shape(pts84) + sum over WJCs (PLAYED + PER_PTS84 * wjc84),
          shape = pts84, log(1 + pts84) or sqrt(pts84), picked per model and
          outcome by cross-validated log loss (a straight line in raw pts84
          under-rates 10-25 NHLe seasons),
          with WJC-20's PLAYED and PER_PTS84 each + slope * (age - 19), for each
          outcome separately.
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

CLASSES = range(2006, 2019)    # 2019+ haven't had time to reach 200 GP
MIN_GP = 10
MIN_COVERAGE = 0.5
WJC_RIDGE = 1.0                # L2 penalty on the WJC terms: a N(0, 1) prior on PLAYED and
WJC_SCALE = 20.0               # on PER_PTS84 per 20 NHLe pts/84 (about the spread of WJC scoring)
WJC_AGE_CENTER = 19.0          # WJC-20 terms are a line in age at the tournament, centred here
# A playoff game's weight in pts/84, per position: best of PO_SWEEP (see po_sweep.json)
PO_WEIGHT = {"F": 1.5, "D": 1.25}
PO_SWEEP = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
SHAPES = ("linear", "log", "sqrt")
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
    "GOHL": "gohl", "GOJHL": "gohl", "OHA-B": "gohl", "CSSHL U18": "csshl", "CSSHL": "csshl",
    "GTHL": "gthl", "GTHL U16": "gthl",
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

def xg_stars():
    """Star seasons by on-ice xG (training/xg_star.py): {playerId: first star season},
    the player-seasons, and the Pareto group's share of skaters per position."""
    stars = {int(k): v for k, v in json.loads((HERE / "stars.json").read_text(encoding="utf-8"))["stars"].items()}
    seasons = pd.read_csv(HERE / "xg_seasons.csv")
    share = seasons.groupby("pos").pareto_share.mean().round(3).to_dict()
    return stars, seasons, share


def star_examples(seasons):
    """Star seasons nearest the line in the latest full season, per position."""
    names = {}
    with (DATA / "landings.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            names[r["playerId"]] = r.get("name", str(r["playerId"]))
    last = seasons[seasons.season == seasons.season.max()]
    out = {}
    for pos in ("F", "D"):
        c = last[(last.pos == pos) & last.star_season].sort_values("value").head(5)
        out[pos] = [names.get(int(i), str(i)) for i in c.player_id]
    return out


# ── Prospect features ─────────────────────────────────────────────────────────

def first_eligible(birth):
    """First NHL draft a player is eligible for: 18 by September 15 of the draft year."""
    b = pd.to_datetime(birth, errors="coerce")
    late = (b.dt.month > 9) | ((b.dt.month == 9) & (b.dt.day > 15))
    return b.dt.year + 18 + late.astype(float)


def shape(x, kind):
    x = np.maximum(np.asarray(x, float), 0)
    return {"linear": x, "log": np.log1p(x), "sqrt": np.sqrt(x)}[kind]


def player_seasons():
    """Draft picks, their stints (playerId, season end year, kind, league key, gp,
    pts) in every league with a key, and their league games for the coverage check."""
    draft = pd.read_csv(DATA / "draft.csv").dropna(subset=["playerId"])
    draft["playerId"] = draft.playerId.astype(int)
    draft = draft[draft.draftYear.isin(CLASSES)].copy()
    draft["pos"] = np.where(draft.position == "D", "D", "F")
    draft["baseYear"] = first_eligible(draft.birthDate).fillna(draft.draftYear).astype(int)
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


def model_sample(d, stints, cover, rel, tourn, factors, po_weight=1.0):
    """One row per pick with a usable season: club pts84 (x), reg GP, and per
    tournament its raw points and games."""
    st = stints.merge(d[["playerId", "baseYear"]])
    st = st[st.end - st.baseYear == rel]
    club = st[st.kind.isin(["reg", "po"]) & st.league.isin(factors)]
    w = np.where(club.kind == "po", po_weight, 1.0)
    agg = club.assign(nhle=club.pts * club.league.map(factors) * w, gp=club.gp * w).groupby("playerId").agg(
        nhle=("nhle", "sum"), gp=("gp", "sum"))
    reg_gp = club[club.kind == "reg"].groupby("playerId").gp.sum().rename("reg_gp")
    cv = cover.merge(d[["playerId", "baseYear"]])
    cv = cv[cv.end - cv.baseYear == rel]
    all_gp = cv.groupby("playerId").gp.sum().rename("all_gp")
    rated_gp = cv[cv.league.isin(factors)].groupby("playerId").gp.sum().rename("rated_gp")
    sample = d.set_index("playerId").join(agg, how="inner").join(reg_gp).join(all_gp).join(rated_gp)
    sample = sample[(sample.reg_gp >= MIN_GP) & (sample.rated_gp >= MIN_COVERAGE * sample.all_gp)].copy()
    sample["x"] = sample.nhle / sample.gp * 84
    # Age on January 1 of the season, i.e. at the World Juniors
    born = pd.to_datetime(sample.birthDate, errors="coerce")
    jan1 = pd.to_datetime((sample.baseYear + rel).astype(int).astype(str) + "-01-01")
    sample["wjc_age"] = ((jan1 - born).dt.days / 365.25).fillna(18.0 + rel - 0.5)
    for t in sorted(tourn):
        w = st[st.kind == t].groupby("playerId")[["pts", "gp"]].sum()
        sample[f"{t}_pts"] = w.pts.reindex(sample.index).fillna(0)
        sample[f"{t}_gp"] = w.gp.reindex(sample.index).fillna(0)
    return sample, int(cv.playerId.nunique())


def design(sample, tourn, factors, kind="linear"):
    """Columns: 1, shape(pts84), then per tournament PLAYED and NHLe pts/84 there."""
    cols = [np.ones(len(sample)), shape(sample.x, kind)]
    for t in sorted(tourn):
        gp = sample[f"{t}_gp"].to_numpy()
        played = (gp > 0).astype(float)
        w84 = np.divide(sample[f"{t}_pts"].to_numpy() * factors[t] * 84, gp, out=np.zeros(len(gp)), where=gp > 0)
        cols += [played, w84]
    return np.column_stack(cols)


def wjc20_cols(sample, factors):
    """The WJC-20 columns shared across D-1..D+2: PLAYED, NHLe pts/84, and each
    times (age - WJC_AGE_CENTER)."""
    gp = sample["wjc-20_gp"].to_numpy()
    played = (gp > 0).astype(float)
    w84 = np.divide(sample["wjc-20_pts"].to_numpy() * factors["wjc-20"] * 84, gp, out=np.zeros(len(gp)), where=gp > 0)
    a = sample.wjc_age.to_numpy() - WJC_AGE_CENTER
    return np.column_stack([played, w84, played * a, w84 * a])


WJC20_RIDGE = [WJC_RIDGE, WJC_RIDGE * WJC_SCALE ** 2, WJC_RIDGE, WJC_RIDGE * WJC_SCALE ** 2]


def joint_design(group, factors, shapes):
    """Stack the models in ``group`` (one position, every model with the WJC-20):
    each keeps its own intercept, pts/84 (in its own shape) and WJC-18 columns,
    and all share the four WJC-20 columns at the end."""
    owns = [design(m["sample"], [t for t in m["tourn"] if t != "wjc-20"], factors, shapes[m["rel"]]) for m in group]
    n = sum(len(o) for o in owns)
    X = np.zeros((n, sum(o.shape[1] for o in owns) + 4))
    rows, col, r0 = [], 0, 0
    for m, o in zip(group, owns):
        r1 = r0 + len(o)
        X[r0:r1, col:col + o.shape[1]] = o
        X[r0:r1, -4:] = wjc20_cols(m["sample"], factors)
        rows.append((r0, r1, col, col + o.shape[1]))
        col, r0 = col + o.shape[1], r1
    ridge = np.concatenate([ridge_for([t for t in m["tourn"] if t != "wjc-20"]) for m in group] + [WJC20_RIDGE])
    return X, ridge, rows


def pooled_x(sample, tourn, factors):
    """The old way: World Juniors games and NHLe points pooled into pts/84."""
    pts, gp = sample.nhle.to_numpy(float).copy(), sample.gp.to_numpy(float).copy()
    for t in sorted(tourn):
        if t in factors:
            pts += sample[f"{t}_pts"].to_numpy() * factors[t]
            gp += sample[f"{t}_gp"].to_numpy()
    return pts / gp * 84


# ── Logistic regression ───────────────────────────────────────────────────────

def fit_logistic(X, y, ridge=None, iters=100):
    """IRLS for log-odds = X @ beta, with an L2 penalty per column (a whisper on
    the intercept and pts/84 keeps it stable)."""
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
    kind = getattr(old, f"{prefix}_SHAPE", "linear")
    terms = getattr(old, f"{prefix}_WJC", None)
    if terms is None:     # the older pooled scripts (their WJC team-share adjustment needs team totals)
        x = old_x(sample, tourn, factors, pooled=True)
        return sigmoid(a + b * x)
    X = design(sample.assign(x=old_x(sample, tourn, factors, pooled=False)), tourn, factors, kind)
    beta = [a, b] + [v for t in sorted(tourn) for v in terms.get(t, (0.0, 0.0))]
    z = X @ np.array(beta)
    age_terms = getattr(old, f"{prefix}_WJC_AGE", {})
    if "wjc-20" in tourn and "wjc-20" in age_terms:
        cols = wjc20_cols(sample, factors)
        z += cols[:, 2] * age_terms["wjc-20"][0] + cols[:, 3] * age_terms["wjc-20"][1]
    return sigmoid(z)


def old_x(sample, tourn, factors, pooled):
    club = sample.club_stints
    pts = np.array([sum(p * factors.get(lg, 0) for lg, p, g in rows) for rows in club])
    gp = np.array([sum(g for lg, p, g in rows if lg in factors) for rows in club], float)
    if pooled:
        for t in sorted(tourn):
            if t in factors:
                pts = pts + sample[f"{t}_pts"].to_numpy() * factors[t]
                gp = gp + sample[f"{t}_gp"].to_numpy()
    return np.divide(pts * 84, gp, out=np.zeros(len(gp)), where=gp > 0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    write = "--write" in sys.argv
    factors = load_factors()

    stars, xg_seasons, star_share = xg_stars()
    draft, stints, cover = player_seasons()
    draft["nhler"] = (draft.nhl_gp >= 200).astype(int)
    # A star is an NHLer with a star season (so star stays inside NHLer, as the curves require)
    draft["star"] = ((draft.nhler == 1) & draft.playerId.isin(stars)).astype(int)

    po_weight = po_sweep(draft, stints, cover, factors) if "--po-sweep" in sys.argv else PO_WEIGHT
    results = fit_models(draft, stints, cover, factors, po_weight)

    examples = star_examples(xg_seasons)
    out = {"classes": [CLASSES.start, CLASSES.stop - 1], "star_share": star_share,
           "star_examples": examples, "factors": factors, "po_weight": po_weight, "models": results}
    (HERE / "coefficients.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    write_report(out, draft)
    if write:
        for res in results:
            write_script(res, factors, examples[res["pos"]], po_weight[res["pos"]])
    print("Pareto star group, share of skaters:", star_share, "| stars among picks:", int(draft.star.sum()))


def po_sweep(draft, stints, cover, factors):
    """CV log loss of every model at each playoff weight -> training/po_sweep.json;
    returns the best weight per position (NHLer + star log loss summed over its 7 models)."""
    sweep, best = {}, {}
    for w in PO_SWEEP:
        res = fit_models(draft, stints, cover, factors, {"F": w, "D": w}, quiet=True)
        sweep[str(w)] = {f"{r['pos']}{r['rel']:+d}": [r["nhler"]["cv_log_loss"], r["star"]["cv_log_loss"]] for r in res}
        for pos in ("F", "D"):
            tot = sum(sum(v) for k, v in sweep[str(w)].items() if k[0] == pos)
            print(f"playoff weight {w} {pos}: summed CV log loss {tot:.4f}")
            if pos not in best or tot < best[pos][1]:
                best[pos] = (w, tot)
    (HERE / "po_sweep.json").write_text(json.dumps(sweep, indent=1), encoding="utf-8")
    print("best playoff weights:", {p: b[0] for p, b in best.items()})
    return {p: b[0] for p, b in best.items()}


def fit_models(draft, stints, cover, factors, po_weight, quiet=False):
    prepared = []
    for pos, rel, script, tourn in MODELS:
        d = draft[draft.pos == pos]
        sample, played = model_sample(d, stints, cover, rel, tourn, factors, po_weight[pos])
        # Each pick's club stints, so the old scripts can be scored with their own factors
        st = stints.merge(d[["playerId", "baseYear"]])
        st = st[(st.end - st.baseYear == rel) & st.kind.isin(["reg", "po"])]
        rows = st.groupby("playerId").apply(lambda g: list(zip(g.league, g.pts, g.gp)), include_groups=False)
        sample["club_stints"] = rows.reindex(sample.index).apply(lambda r: r if isinstance(r, list) else [])
        m = {"pos": pos, "rel": rel, "script": script, "tourn": sorted(tourn), "sample": sample, "played": played}
        # The curve's shape, per model and outcome: the one with the best CV log loss
        classes = sample.draftYear.to_numpy()
        m["shape_cv"] = {}
        for target in ("nhler", "star"):
            y = sample[target].to_numpy()
            m["shape_cv"][target] = {k: log_loss(cv_predict(design(sample, tourn, factors, k), y, classes,
                                                            ridge_for(tourn)), y) for k in SHAPES}
        m["shape"] = {t: min(c, key=c.get) for t, c in m["shape_cv"].items()}
        prepared.append(m)

    # The models with the WJC-20 are fit jointly per position, sharing its age line.
    joint = {}
    for pos in ("F", "D"):
        group = [m for m in prepared if m["pos"] == pos and "wjc-20" in m["tourn"]]
        classes = np.concatenate([m["sample"].draftYear.to_numpy() for m in group])
        for target in ("nhler", "star"):
            X, ridge, rows = joint_design(group, factors, {m["rel"]: m["shape"][target] for m in group})
            shared = list(range(X.shape[1] - 4, X.shape[1]))
            y = np.concatenate([m["sample"][target].to_numpy() for m in group])
            beta = fit_logistic(X, y, ridge)
            cvp = cv_predict(X, y, classes, ridge)
            for m, (r0, r1, c0, c1) in zip(group, rows):
                cols = list(range(c0, c1)) + shared
                joint[(pos, m["rel"], target)] = (beta[cols], X[r0:r1][:, cols], cvp[r0:r1])

    results = []
    for m in prepared:
        pos, rel, script, tourn, sample, played = (m[k] for k in ("pos", "rel", "script", "tourn", "sample", "played"))
        Xp = np.column_stack([np.ones(len(sample)), pooled_x(sample, tourn, factors)])
        classes = sample.draftYear.to_numpy()
        ridge = ridge_for(tourn)
        others = [t for t in tourn if t != "wjc-20"]
        res = {"pos": pos, "rel": rel, "script": script, "n": len(sample), "played": played,
               "tournaments": tourn,
               "wjc_played": {t: int((sample[f"{t}_gp"] > 0).sum()) for t in tourn}}
        if "wjc-20" in tourn:
            res["wjc_age_median"] = round(float(sample.wjc_age[sample["wjc-20_gp"] > 0].median()), 1)
        old = load_script(script) if (ROOT / script).exists() and not quiet else None
        for target, prefix in (("nhler", "NHLER"), ("star", "STAR")):
            y = sample[target].to_numpy()
            kind = m["shape"][target]
            X = design(sample, tourn, factors, kind)
            per_model_cv = log_loss(cv_predict(X, y, classes, ridge), y) if not quiet else float("nan")
            if "wjc-20" in tourn:
                beta, Xj, cvp = joint[(pos, rel, target)]
                score = Xj @ beta
                wjc = {t: [round(float(beta[2 + 2 * i]), 6), round(float(beta[3 + 2 * i]), 6)]
                       for i, t in enumerate(others)}
                wjc["wjc-20"] = [round(float(beta[-4]), 6), round(float(beta[-3]), 6)]
                wjc_age = {"wjc-20": [round(float(beta[-2]), 6), round(float(beta[-1]), 6)]}
            else:
                beta = fit_logistic(X, y, ridge)
                score, cvp = X @ beta, cv_predict(X, y, classes, ridge)
                wjc = {t: [round(float(beta[2 + 2 * i]), 6), round(float(beta[3 + 2 * i]), 6)]
                       for i, t in enumerate(others)}
                wjc_age = {}
            mres = {"intercept": round(float(beta[0]), 6), "coef": round(float(beta[1]), 6),
                    "shape": kind, "shape_cv": {k: round(v, 5) for k, v in m["shape_cv"][target].items()},
                    # CV log loss of the straight line in raw pts84 (the old shape), for comparison
                    "linear_cv_log_loss": m["shape_cv"][target]["linear"],
                    "wjc": wjc, "wjc_age": wjc_age,
                    "rate": float(y.mean()), "auc": float(auc(score, y)),
                    "log_loss": log_loss(sigmoid(score), y), "cv_log_loss": log_loss(cvp, y),
                    # Each model's World Juniors fit on its own, without the shared age line
                    "per_model_cv_log_loss": per_model_cv,
                    # The same data with the World Juniors pooled into pts/84, as before
                    "pooled_cv_log_loss": log_loss(cv_predict(Xp, y, classes, None), y) if not quiet else float("nan")}
            if old is not None:
                mres["old_log_loss"] = log_loss(old_predict(old, prefix, sample, tourn), y)
            res[target] = mres
        res["pts84_median"] = float(np.median(sample.x))
        res["po_weight"] = po_weight[pos]
        results.append(res)
        if quiet:
            continue
        print(f"{pos} D{rel:+d}  shapes {res['nhler']['shape']}/{res['star']['shape']}  "
              f"n={res['n']:4d}/{played}  NHLer {res['nhler']['rate']:.1%} "
              f"(AUC {res['nhler']['auc']:.3f}, CV {res['nhler']['cv_log_loss']:.4f} / per-model "
              f"{res['nhler']['per_model_cv_log_loss']:.4f} / pooled {res['nhler']['pooled_cv_log_loss']:.4f})"
              f"  star {res['star']['rate']:.1%} (AUC {res['star']['auc']:.3f}, CV {res['star']['cv_log_loss']:.4f}"
              f" / per-model {res['star']['per_model_cv_log_loss']:.4f} / pooled {res['star']['pooled_cv_log_loss']:.4f})")
    return results


def write_script(res, factors, examples, po_weight):
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
                   "pts/84: making the team and scoring there move the odds on top of the\n"
                   "league season (NHLER_WJC / STAR_WJC), so a big tournament counts for more\n"
                   "than its 7 games would." + (" The WJC-20 terms also depend on age at the\n"
                   "tournament (NHLER_WJC_AGE / STAR_WJC_AGE)." if "wjc-20" in tourn else ""))
    else:
        wjc_doc = f"No World Juniors: players are {age}, no longer eligible."
    keep = {k: v for k, v in factors.items() if tourn or k not in TOURNAMENTS}
    width = max(len(k) for k in keep) + 3
    factor_lines = "\n".join(f"    {json.dumps(k) + ':':<{width}} {v:.6f}," for k, v in
                             sorted(keep.items(), key=lambda kv: -kv[1]))

    def wjc_dict(terms):
        return "{" + ", ".join(f'"{t}": ({a:.6f}, {b:.6f})' for t, (a, b) in terms.items()) + "}"

    text = Template((HERE / "model_template.py.in").read_text(encoding="utf-8")).substitute(
        group_title=group, label=label, age=age, underline="=" * len(title), season_desc=season,
        group_plural="forwards" if pos == "F" else "defensemen", known_doc=known_doc, wjc_doc=wjc_doc,
        classes=f"{CLASSES.start}-{CLASSES.stop - 1}", script_name=Path(res["script"]).name,
        nhler_intercept=f"{res['nhler']['intercept']: .6f}", nhler_coef=f"{res['nhler']['coef']: .6f}",
        po_weight=f"{po_weight}",
        nhler_shape=repr(res["nhler"]["shape"]), star_shape=repr(res["star"]["shape"]),
        nhler_wjc=wjc_dict(res["nhler"]["wjc"]), nhler_wjc_age=wjc_dict(res["nhler"]["wjc_age"]),
        star_intercept=f"{res['star']['intercept']: .6f}", star_coef=f"{res['star']['coef']: .6f}",
        star_wjc=wjc_dict(res["star"]["wjc"]), star_wjc_age=wjc_dict(res["star"]["wjc_age"]),
        wjc_age_typical=f"{res.get('wjc_age_median') or 18.0 + rel - 0.5:.1f}",

        factors=factor_lines, tournaments=repr(tuple(tourn)), known=known, examples=", ".join(examples))
    (ROOT / res["script"]).write_text(text, encoding="utf-8", newline="\n")
    print("wrote", res["script"])


def write_report(out, draft):
    t = out["star_share"]
    lines = [
        "# NHLe model fit",
        "",
        "Generated by `training/fit.py` — don't edit by hand.",
        "",
        f"Draft classes {out['classes'][0]}–{out['classes'][1]}: {len(draft)} picks with an NHL id "
        f"({(draft.pos == 'F').sum()} forwards, {(draft.pos == 'D').sum()} defensemen). "
        f"League factors: network NHLe, {len(out['factors'])} leagues ([NETWORK.md](NETWORK.md)).",
        f"NHL player = 200+ NHL games. Star = NHL player with a star season by on-ice xG "
        "([xg_star.py](xg_star.py), scored with the xG model): on-ice net xG above replacement at his average 5-on-5 "
        "minutes over 84 games, inside that season's Pareto group of regulars (40+ games) at his position (on average "
        f"the top {t['F']:.0%} of forwards and {t['D']:.0%} of defensemen).",
        f"Star examples near the line — F: {', '.join(out['star_examples']['F'])}; "
        f"D: {', '.join(out['star_examples']['D'])}.",
        "",
        "`n` is the number of picks with a usable season (10+ GP, at least half of it in leagues with an NHLe "
        "factor) out of those who played that season at all. `x` is NHLe pts/84 in league play (regular "
        f"season + playoffs, each playoff game counting as {out['po_weight']['F']} games for forwards and "
        f"{out['po_weight']['D']} for defensemen; see [po_sweep.json](po_sweep.json)). Seasons are counted from each "
        "player's first eligible draft. `shape` is the curve in pts84 each fit uses (linear, log(1+x) or sqrt), the one "
        "with the best CV log loss; `linear` is the CV log loss a straight line would have. "
        "Log loss (lower is better) is on the same sample five ways: the fit; "
        "leave-one-draft-class-out cross-validation; CV with each model's World Juniors terms fit on their "
        "own, without the shared WJC-20 age line (`per-season`); CV with the World Juniors pooled into "
        "pts/84 instead of their own term (`pooled`); and the coefficients and factors the script had before this fit.",
        "",
        "| Model | n / played | NHLer rate | NHLer coefs | AUC | log loss fit / CV / linear / per-season / pooled / old | Star rate | Star coefs | AUC | log loss fit / CV / linear / per-season / pooled / old |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in out["models"]:
        cells = [f"{'Fwd' if r['pos'] == 'F' else 'Def'} D{r['rel']:+d}".replace("D+0", "DY"),
                 f"{r['n']} / {r['played']}"]
        for target in ("nhler", "star"):
            m = r[target]
            old = f"{m['old_log_loss']:.3f}" if "old_log_loss" in m else "–"
            xs = {"linear": "x", "log": "log(1+x)", "sqrt": "√x"}[m["shape"]]
            cells += [f"{m['rate']:.1%}", f"{m['intercept']:.3f} + {m['coef']:.4f}·{xs}",
                      f"{m['auc']:.3f}",
                      f"{m['log_loss']:.3f} / {m['cv_log_loss']:.3f} / {m['linear_cv_log_loss']:.3f} / "
                      f"{m['per_model_cv_log_loss']:.3f} / "
                      f"{m['pooled_cv_log_loss']:.3f} / {old}"]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "## World Juniors terms",
        "",
        "Log-odds added for a player who played the tournament: `played + per_pts84 × (his NHLe pts/84 there)`, "
        "where NHLe pts/84 = WJC points × the WJC factor / games × 84. "
        f"Ridge {WJC_RIDGE} on these terms (a N(0, 1) prior on `played`, and on `per_pts84` per "
        f"{WJC_SCALE:.0f} pts/84), so a term with a handful of players stays near zero.",
        "",
        "### WJC-20, by age at the tournament",
        "",
        f"One line in age (on January 1) per position, fit jointly across D-1 to D+2: `played` and `per_pts84` "
        f"each move by a slope per year away from {WJC_AGE_CENTER:.0f}. Cells are the total log-odds at "
        "0 / 20 / 40 NHLe pts/84.",
        "",
        "| Position | Outcome | played (at 19) | per year | per_pts84 (at 19) | per year | age 17 | age 18 | age 19 | age 20 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for pos in ("F", "D"):
        r = next(m for m in out["models"] if m["pos"] == pos and "wjc-20" in m["tournaments"])
        for target in ("nhler", "star"):
            a, b = r[target]["wjc"]["wjc-20"]
            sa, sb = r[target]["wjc_age"]["wjc-20"]
            ages = [" / ".join(f"{a + sa * (age - WJC_AGE_CENTER) + (b + sb * (age - WJC_AGE_CENTER)) * x:+.2f}"
                               for x in (0, 20, 40)) for age in (17, 18, 19, 20)]
            lines.append(f"| {'Forwards' if pos == 'F' else 'Defense'} | {'NHLer' if target == 'nhler' else 'Star'} | "
                         f"{a:+.3f} | {sa:+.3f} | {b:+.4f} | {sb:+.4f} | " + " | ".join(ages) + " |")
    lines += ["", "Players at the WJC-20 in each model's sample, and their median age:", ""]
    lines += [f"- {'Fwd' if r['pos'] == 'F' else 'Def'} D{r['rel']:+d}".replace("D+0", "DY") +
              f": {r['wjc_played']['wjc-20']} (median {r['wjc_age_median']})"
              for r in out["models"] if "wjc-20" in r["tournaments"]]
    lines += [
        "",
        "### WJC-18",
        "",
        "| Model | Played | NHLer: played, per pts/84 | at 0 / 20 / 40 | Star: played, per pts/84 | at 0 / 20 / 40 |",
        "|---|---|---|---|---|---|",
    ]
    for r in out["models"]:
        if "wjc-18" not in r["tournaments"]:
            continue
        cells = [f"{'Fwd' if r['pos'] == 'F' else 'Def'} D{r['rel']:+d}".replace("D+0", "DY"),
                 str(r["wjc_played"]["wjc-18"])]
        for target in ("nhler", "star"):
            a, b = r[target]["wjc"]["wjc-18"]
            cells += [f"{a:+.3f}, {b:+.4f}", " / ".join(f"{a + b * x:+.2f}" for x in (0, 20, 40))]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Re-run with `python training/scrape.py`, `python training/network.py`, then "
              "`python training/fit.py --write`.", ""]
    (HERE / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
