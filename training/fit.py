"""
Refit every NHLe model (forwards and defense, D-1 through D+5) from the data
scraped by training/scrape.py.

  python training/fit.py           # fit, write training/REPORT.md + coefficients.json
  python training/fit.py --write   # also write the coefficients into the model scripts

Sample:   NHL draft picks from CLASSES, by draft-listed position (C/L/R = F, D = D).
Feature:  NHLe pts/82 for the season N years from the draft, pooled the same way
          the scripts pool it: regular season + playoffs in every league with a
          factor, plus the tournaments that model asks for (WJC-20 / WJC-18).
          A player-season is used only if at least half of its league games are
          in leagues with a factor (and 10+ of them).
Outcomes: NHL player = 200+ NHL regular-season games (through the latest season).
          Star = NHL player whose peak GAR (mean of best 3 seasons) is in the top
          15% of NHL skaters at the position with 200+ GP and a season since
          2008-09. GAR is Hockey-Reference's through 2020-21 and, after that,
          estimated from MoneyPuck season stats (see gar_seasons).
Model:    log-odds = INTERCEPT + COEF * pts82, for each outcome separately.
          The WJC team-share adjustment in the scripts is applied on top and is
          not refit: it needs team totals the NHL data doesn't have.
"""

import importlib.util
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "data"
GAR_CSV = ROOT / "pareto" / "NHL-GAR-data-v2.5-1956-2022.csv"
HR_LAST_SEASON = 2020          # start year of the last complete HR season (2020-21)

CLASSES = range(2006, 2019)    # 2019+ haven't had time to reach 200 GP / a GAR peak
MIN_GP = 10
MIN_COVERAGE = 0.5
STAR_PCT = 0.85

# Scripts per (position, season relative to draft year) and the tournaments
# each one pools in.
MODELS = [
    ("F", -1, "Forwards/fwd-d-1.py", {"wjc-18"}),
    ("F",  0, "Forwards/fwd-d0.py",  {"wjc-18", "wjc-20"}),
    ("F",  1, "Forwards/fwd-d1.py",  {"wjc-20"}),
    ("F",  2, "Forwards/fwd-d2.py",  {"wjc-20"}),
    ("F",  3, "Forwards/fwd-d3.py",  set()),
    ("F",  4, "Forwards/fwd-d4.py",  set()),
    ("F",  5, "Forwards/fwd-d5.py",  set()),
    ("D", -1, "Defense/d-d-1.py",    {"wjc-18"}),
    ("D",  0, "Defense/d-d0.py",     {"wjc-18", "wjc-20"}),
    ("D",  1, "Defense/d-d1.py",     {"wjc-20"}),
    ("D",  2, "Defense/d-d2.py",     {"wjc-20"}),
    ("D",  3, "Defense/d-d3.py",     set()),
    ("D",  4, "Defense/d-d4.py",     set()),
    ("D",  5, "Defense/d-d5.py",     set()),
]
TOURNAMENTS = {"wjc-20", "wjc-18"}

# NHL API league names -> LEAGUE_FACTORS keys. Names changed over the years
# (Sweden -> SHL, CzRep -> Czechia...) and NCAA is listed by conference.
LEAGUE_ALIASES = {
    "NHL": "nhl", "AHL": "ahl", "OHL": "ohl", "WHL": "whl", "QMJHL": "qmjhl", "USHL": "ushl",
    "NCAA": "ncaa", "WCHA": "ncaa", "H-East": "ncaa", "CCHA": "ncaa", "ECAC": "ncaa",
    "NCHC": "ncaa", "Big Ten": "ncaa", "Atlantic": "ncaa", "AHA": "ncaa", "CHA": "ncaa",
    "SHL": "shl", "Sweden": "shl",
    "Liiga": "liiga", "Finland": "liiga", "SM-liiga": "liiga",
    "KHL": "khl", "Rus-KHL": "khl", "Russia": "khl",
    "VHL": "vhl", "Russia-2": "vhl",
    "MHL": "mhl", "Russia-Jr.": "mhl",
    "HockeyAllsvenskan": "allsvenskan", "Allsvenskan": "allsvenskan", "Sweden-2": "allsvenskan",
    "Swe-Jr.": "u20-nationell", "J20 SuperElit": "u20-nationell", "J20 Nationell": "u20-nationell",
    "U20 SM-liiga": "u20-sm-sarja", "U20 SM-sarja": "u20-sm-sarja", "Fin-Jr.": "u20-sm-sarja",
    "Czechia": "czechia", "CzRep": "czechia", "Czech": "czechia",
    "Czechia2": "czechia2", "CzRep-2": "czechia2", "Czech2": "czechia2",
    "CzRep-Jr.": "czechia-u20", "Czechia U20": "czechia-u20",
    "NL": "nl", "NLA": "nl", "Swiss": "nl",
    "SL": "sl", "NLB": "sl", "Swiss-2": "sl",
    "DEL": "del", "Germany": "del",
    "DEL2": "del2", "German-2": "del2",
    "Ger-Jr.": "dnl", "DNL": "dnl",
    "Mestis": "mestis", "Finland-2": "mestis",
    "Slovakia": "slovakia", "Slovakia2": "slovakia2", "Slovak-2": "slovakia2",
    "Latvia": "latvia", "BCHL": "bchl",
    "WJC-A": "wjc-20", "WJC-20": "wjc-20",
    "WJ18-A": "wjc-18", "WJC-18": "wjc-18",
}

# Short events that aren't a league season; left out of the coverage check.
EVENTS = {"WC", "WC-A", "WCup", "Olympics", "OG", "OGQ", "Champions HL", "Spengler Cup",
          "M-Cup", "Exhib.", "International", "Ivan Hlinka Memorial", "Hlinka Gretzky Cup",
          "WHC-17", "WJAC-19", "JCWC", "Tel-Cup", "Continental Cup", "WJC-B", "WJ18-B",
          "EHT", "TV-Pucken", "U17-Dev"}


def load_script(rel_path):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def norm_name(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return s.lower().replace(".", "").replace("-", " ").strip()


# ── NHL outcomes ──────────────────────────────────────────────────────────────

GAR_FEATURES = ["I_F_goals", "I_F_primaryAssists", "I_F_secondaryAssists", "xgd", "gd",
                "toi", "games_played", "I_F_xGoals", "penalties", "I_F_takeaways",
                "I_F_giveaways", "shotsBlockedByPlayer"]


def gar_seasons():
    """One GAR value per (NHL playerId, season start year), plus career GP.

    Hockey-Reference GAR where it exists (through 2020-21). Later seasons use a
    per-position linear fit of HR GAR on MoneyPuck season stats; on held-out
    2017-18 to 2020-21 it explains 97.5% (F) / 95.9% (D) of the variance.
    Estimated seasons are then rescaled to HR's league-wide GAR per team-game.
    """
    hr = pd.read_csv(GAR_CSV, encoding="latin-1")
    hr = hr[hr.maj_pos.isin(["F", "D"])]
    hr = hr.groupby(["player_ID", "name_common", "year_ID", "maj_pos"], as_index=False).agg(
        gar=("tot_gar", "sum"), gp=("gp", "sum"))
    hr["season"] = hr.year_ID - 1
    hr = hr[hr.season <= HR_LAST_SEASON]
    hr["key"] = hr.name_common.map(norm_name)

    mp = pd.read_csv(DATA / "moneypuck.csv")
    mp = mp[mp.situation == "all"].copy()
    mp["key"] = mp.name.map(norm_name)
    mp["xgd"] = mp.OnIce_F_xGoals - mp.OnIce_A_xGoals
    mp["gd"] = mp.OnIce_F_goals - mp.OnIce_A_goals
    mp["toi"] = mp.icetime / 3600
    mp["pos"] = np.where(mp.position == "D", "D", "F")

    both = hr.merge(mp, on=["key", "season"])
    both = both[(both.gp - both.games_played).abs() <= 3]

    mp["est_gar"] = np.nan
    for pos in ("F", "D"):
        train = both[both.maj_pos == pos]
        X = np.column_stack([np.ones(len(train))] + [train[f].to_numpy(float) for f in GAR_FEATURES])
        beta, *_ = np.linalg.lstsq(X, train.gar.to_numpy(float), rcond=None)
        rows = mp.pos == pos
        Xp = np.column_stack([np.ones(rows.sum())] + [mp.loc[rows, f].to_numpy(float) for f in GAR_FEATURES])
        mp.loc[rows, "est_gar"] = Xp @ beta

    # HR player -> NHL id, from the seasons that matched
    xwalk = both.groupby("player_ID").playerId.agg(lambda s: s.mode().iloc[0])
    hr["playerId"] = hr.player_ID.map(xwalk)
    hr_rows = hr.dropna(subset=["playerId"])[["playerId", "season", "gar", "gp", "maj_pos"]]
    hr_rows = hr_rows.rename(columns={"maj_pos": "pos"})
    # HR GAR is measured against replacement level, so the league total per
    # team-game barely moves (2013-14 to 2018-19). Raw stats rose after 2021,
    # which pushed the estimate ~25% high; rescale each estimated season back
    # to HR's level.
    def team_games(season):
        teams = mp[(mp.season == season) & (mp.team.str.len() == 3)].team.nunique()
        return teams * 82
    ref = range(2013, 2019)
    hr_rate = hr[hr.season.isin(ref)].gar.sum() / sum(team_games(s) for s in ref)
    mp_rows = mp[mp.season > HR_LAST_SEASON][["playerId", "season", "est_gar", "games_played", "pos"]]
    mp_rows = mp_rows.rename(columns={"est_gar": "gar", "games_played": "gp"})
    for s in mp_rows.season.unique():
        rows = mp_rows.season == s
        mp_rows.loc[rows, "gar"] *= hr_rate * team_games(s) / mp_rows.loc[rows, "gar"].sum()
    # Matched HR seasons win; unmatched MP seasons up to 2020-21 fill the gaps
    gap = mp[mp.season <= HR_LAST_SEASON][["playerId", "season", "est_gar", "games_played", "pos"]]
    gap = gap.rename(columns={"est_gar": "gar", "games_played": "gp"})
    gap = gap.merge(hr_rows[["playerId", "season"]], how="left", indicator=True)
    gap = gap[gap._merge == "left_only"].drop(columns="_merge")

    seasons = pd.concat([hr_rows, gap, mp_rows], ignore_index=True)
    seasons["playerId"] = seasons.playerId.astype(int)
    return seasons


def star_thresholds(seasons):
    career = seasons.groupby("playerId").agg(
        gp=("gp", "sum"), last=("season", "max"), pos=("pos", lambda s: s.mode().iloc[0]),
        peak=("gar", lambda g: g.nlargest(3).sum() / 3))
    pool = career[(career.gp >= 200) & (career["last"] >= 2008)]
    return {pos: float(pool[pool.pos == pos].peak.quantile(STAR_PCT)) for pos in ("F", "D")}, career


# ── Prospect features ─────────────────────────────────────────────────────────

def player_seasons(factors):
    """Rows of (playerId, season end year, kind, league key, gp, pts) + coverage."""
    draft = pd.read_csv(DATA / "draft.csv").dropna(subset=["playerId"])
    draft["playerId"] = draft.playerId.astype(int)
    draft = draft[draft.draftYear.isin(CLASSES)]
    draft["pos"] = np.where(draft.position == "D", "D", "F")
    ids = set(draft.playerId)

    stints, cover = [], []
    with (DATA / "landings.jsonl").open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pid = rec["playerId"]
            if pid not in ids:
                continue
            nhl_gp = 0
            for s in rec["seasonTotals"]:
                end = int(str(s["season"])[4:])
                league = s.get("leagueAbbrev")
                gp, pts = s.get("gamesPlayed") or 0, s.get("points") or 0
                key = LEAGUE_ALIASES.get(league)
                if league == "NHL" and s["gameTypeId"] == 2:
                    nhl_gp += gp
                if s["gameTypeId"] == 2 and league not in EVENTS and key not in TOURNAMENTS:
                    cover.append((pid, end, gp, gp if key in factors else 0))
                if key not in factors or gp <= 0:
                    continue
                if key in TOURNAMENTS:
                    kind = key if s["gameTypeId"] == 2 else None
                else:
                    kind = "reg" if s["gameTypeId"] == 2 else "po"
                if kind:
                    stints.append((pid, end, kind, key, gp, pts))
            draft.loc[draft.playerId == pid, "nhl_gp"] = nhl_gp

    stints = pd.DataFrame(stints, columns=["playerId", "end", "kind", "league", "gp", "pts"])
    stints["nhle"] = stints.pts * stints.league.map(factors)
    cover = pd.DataFrame(cover, columns=["playerId", "end", "gp", "mapped_gp"]).groupby(
        ["playerId", "end"], as_index=False).sum()
    return draft, stints, cover


# ── Logistic regression ───────────────────────────────────────────────────────

def fit_logistic(x, y, iters=50):
    """IRLS for log-odds = a + b*x (a whisper of ridge keeps it stable)."""
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ beta)))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, X.T @ (y - p))
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return beta


def auc(score, y):
    ranks = pd.Series(score).rank().to_numpy()
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return (ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else float("nan")


def log_loss(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    write = "--write" in sys.argv
    factors = load_script("Forwards/fwd-d0.py").LEAGUE_FACTORS   # the full table, WJC included

    seasons = gar_seasons()
    thresholds, career = star_thresholds(seasons)
    draft, stints, cover = player_seasons(factors)
    draft["nhl_gp"] = draft.nhl_gp.fillna(0)
    draft = draft.merge(career[["peak"]], left_on="playerId", right_index=True, how="left")
    draft["nhler"] = (draft.nhl_gp >= 200).astype(int)
    draft["star"] = ((draft.nhler == 1) &
                     (draft.peak >= draft.pos.map(thresholds))).astype(int)

    results = []
    for pos, rel, script, tourn in MODELS:
        d = draft[draft.pos == pos]
        st = stints[stints.playerId.isin(d.playerId)].merge(d[["playerId", "draftYear"]])
        st = st[(st.end - st.draftYear == rel) & st.kind.isin({"reg", "po"} | tourn)]
        agg = st.groupby("playerId").agg(nhle=("nhle", "sum"), gp=("gp", "sum"))
        reg_gp = st[st.kind == "reg"].groupby("playerId").gp.sum().rename("reg_gp")
        cv = cover.merge(d[["playerId", "draftYear"]])
        cv = cv[cv.end - cv.draftYear == rel].set_index("playerId")
        sample = d.set_index("playerId").join(agg, how="inner").join(reg_gp).join(
            cv[["gp", "mapped_gp"]].rename(columns={"gp": "all_gp"}))
        eligible = (sample.reg_gp >= MIN_GP) & (sample.mapped_gp >= MIN_COVERAGE * sample.all_gp)
        played = cv.index.intersection(d.playerId).size
        sample = sample[eligible]
        x = (sample.nhle / sample.gp * 82).to_numpy()
        res = {"pos": pos, "rel": rel, "script": script, "n": len(sample), "played": int(played)}
        old = load_script(script) if (ROOT / script).exists() else None
        classes = sample.draftYear.to_numpy()
        for target in ("nhler", "star"):
            y = sample[target].to_numpy()
            a, b = fit_logistic(x, y)
            p = 1 / (1 + np.exp(-(a + b * x)))
            # Leave-one-draft-class-out: each class predicted by a fit on the others
            cv = np.empty_like(p)
            for c in np.unique(classes):
                held = classes == c
                ca, cb = fit_logistic(x[~held], y[~held])
                cv[held] = 1 / (1 + np.exp(-(ca + cb * x[held])))
            res[target] = {"intercept": round(float(a), 6), "coef": round(float(b), 6),
                           "rate": float(y.mean()), "auc": float(auc(x, y)),
                           "log_loss": log_loss(p, y), "cv_log_loss": log_loss(cv, y)}
            if old is not None:
                prefix = "NHLER" if target == "nhler" else "STAR"
                oa, ob = getattr(old, f"{prefix}_INTERCEPT"), getattr(old, f"{prefix}_COEF")
                res[target]["old_log_loss"] = log_loss(1 / (1 + np.exp(-(oa + ob * x))), y)
                res[target]["old"] = [oa, ob]
        res["pts82_median"] = float(np.median(x))
        results.append(res)
        print(f"{pos} D{rel:+d}  n={res['n']:4d}  NHLer {res['nhler']['rate']:.1%} "
              f"(AUC {res['nhler']['auc']:.3f})  star {res['star']['rate']:.1%} "
              f"(AUC {res['star']['auc']:.3f})")

    examples = star_examples(career, thresholds)
    out = {"classes": [CLASSES.start, CLASSES.stop - 1], "star_thresholds": thresholds,
           "star_examples": examples, "models": results}
    (HERE / "coefficients.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    write_report(out, draft)
    if write:
        for res in results:
            write_coefficients(res, examples[res["pos"]])
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


def write_coefficients(res, examples):
    path = ROOT / res["script"]
    text = path.read_text(encoding="utf-8")
    for prefix, target in (("NHLER", "nhler"), ("STAR", "star")):
        for name, value in ((f"{prefix}_INTERCEPT", res[target]["intercept"]),
                            (f"{prefix}_COEF", res[target]["coef"])):
            # Keeps the scripts' aligned style: "NHLER_COEF      =  0.238853"
            text, n = re.subn(rf"^{name}\s*=.*$", f"{name.ljust(16)}= {value: .6f}", text, flags=re.M)
            assert n == 1, (path, name)
    text, n = re.subn(r'"  Star ≈ [^"]*"', f'"  Star ≈ {", ".join(examples)}"', text)
    assert n == 1, path
    path.write_text(text, encoding="utf-8", newline="\n")
    print("wrote", res["script"])


def write_report(out, draft):
    t = out["star_thresholds"]
    lines = [
        "# NHLe model fit",
        "",
        "Generated by `training/fit.py` — don't edit by hand.",
        "",
        f"Draft classes {out['classes'][0]}–{out['classes'][1]}: {len(draft)} picks with an NHL id "
        f"({(draft.pos == 'F').sum()} forwards, {(draft.pos == 'D').sum()} defensemen).",
        f"NHL player = 200+ NHL games. Star = NHL player with peak GAR (mean of best 3 seasons) at or above "
        f"**{t['F']:.1f}** for forwards and **{t['D']:.1f}** for defensemen (85th percentile of 200+ GP skaters).",
        f"Star examples near the line — F: {', '.join(out['star_examples']['F'])}; "
        f"D: {', '.join(out['star_examples']['D'])}.",
        "",
        "`n` is the number of picks with a usable season (10+ GP, at least half of it in leagues with an NHLe "
        "factor) out of those who played that season at all. Log loss (lower is better) is shown three "
        "ways on the same sample: the fit, leave-one-draft-class-out cross-validation, and the "
        "coefficients the script had before this fit.",
        "",
        "| Model | n / played | NHLer rate | NHLer coefs | AUC | log loss fit / CV / old | Star rate | Star coefs | AUC | log loss fit / CV / old |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in out["models"]:
        cells = [f"{'Fwd' if r['pos'] == 'F' else 'Def'} D{r['rel']:+d}".replace("D+0", "DY"),
                 f"{r['n']} / {r['played']}"]
        for target in ("nhler", "star"):
            m = r[target]
            old = f"{m['old_log_loss']:.3f}" if "old_log_loss" in m else "–"
            cells += [f"{m['rate']:.1%}", f"{m['intercept']:.3f} + {m['coef']:.4f}·x",
                      f"{m['auc']:.3f}", f"{m['log_loss']:.3f} / {m['cv_log_loss']:.3f} / {old}"]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Re-run with `python training/scrape.py` then `python training/fit.py --write`.", ""]
    (HERE / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
