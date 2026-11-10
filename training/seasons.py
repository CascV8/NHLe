"""
Season-by-season NHLe factors: the network factors (training/network.py) allowed to
move from season to season, and to step at the 2025-26 rule change that let CHL
players keep NCAA eligibility.

  python training/seasons.py [--data DIR]   # writes training/factors_by_season.csv + SEASONS.md

The same crossovers as the network solve, pooled per league pair and season
instead of over every season, are fit in log space:

  log PPG_B - log PPG_A = log f_A,s - log f_B,s           (one row per pair and season)
  log f_L,s             = theta_L + delta_L,s + step_L * [s >= RULE_CHANGE]

weighted by the players in the pair that season (variance sigma^2 / n). The NHL is
fixed at 1 every season. delta_L,s is a season's departure from the league's level,
with a N(0, tau^2) prior: tau and sigma are fit by maximum likelihood, so a league
only moves in a season when its crossovers that season say more than noise would.
step_L (NCAA, OHL, WHL, QMJHL only) has a N(0, STEP_SD^2) prior: until there
are enough seasons after the change, it stays near 0.

The published factor is the network factor times exp(delta + step), so the
seasons stay centred on the factors the models use:

  factor_L,s = factors.json[L] * exp(delta_L,s + step_L * [s >= RULE_CHANGE])

`log_se` is the posterior standard deviation of delta + step (it doesn't include
the network factor's own uncertainty, which is in factors.json).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import network  # noqa: E402

HERE = Path(__file__).resolve().parent
RULE_CHANGE = 2025               # 2025-26: CHL players can play NCAA hockey
STEP_LEAGUES = ("ncaa", "ohl", "whl", "qmjhl")
STEP_SD = 0.15                   # prior sd of the log step (a rule change moving a league ~±15%)
RHO_GRID = np.concatenate([[0.0], np.geomspace(1e-4, 1.0, 41)])   # tau^2 / sigma^2


def season_pairs(pairs, leagues):
    """Crossovers pooled per (season, league A, league B), among leagues with a network factor."""
    p = pairs[pairs.league_a.isin(leagues) & pairs.league_b.isin(leagues)]
    g = p.groupby(["season", "league_a", "league_b"]).agg(
        n=("player", "size"), gp_a=("gp_a", "sum"), pts_a=("pts_a", "sum"),
        gp_b=("gp_b", "sum"), pts_b=("pts_b", "sum")).reset_index()
    g = g[(g.pts_a > 0) & (g.pts_b > 0)].reset_index(drop=True)
    g["y"] = np.log((g.pts_b / g.gp_b) / (g.pts_a / g.gp_a))
    return g


def designs(g, leagues, seasons):
    """Level (theta), season (delta) and step design matrices; the NHL has no columns."""
    L = {lg: i for i, lg in enumerate(leagues)}
    S = {s: i for i, s in enumerate(seasons)}
    steps = [lg for lg in STEP_LEAGUES if lg in L]
    X = np.zeros((len(g), len(leagues)))
    Z = np.zeros((len(g), len(leagues) * len(seasons)))
    B = np.zeros((len(g), len(steps)))
    for r, (a, b, s) in enumerate(zip(g.league_a, g.league_b, g.season)):
        for lg, sign in ((a, 1.0), (b, -1.0)):
            if lg == "nhl":
                continue
            X[r, L[lg]] += sign
            Z[r, L[lg] * len(seasons) + S[s]] += sign
            if lg in steps and s >= RULE_CHANGE:
                B[r, steps.index(lg)] += sign
    return X, Z, B, steps


def log_likelihood(y, X, Z, w, rho):
    """Profile log-likelihood of y ~ N(X theta, sigma^2 V), V = W^-1 + rho Z Z', with sigma^2
    profiled out. Woodbury keeps it in the parameter space (Z has a few hundred columns)."""
    ZW = Z.T * w                                  # Z' W
    K = np.eye(Z.shape[1]) + rho * (ZW @ Z)       # I + rho Z' W Z
    Kc = np.linalg.cholesky(K)

    def vinv(u):                                  # V^-1 u = W u - rho W Z K^-1 Z' W u
        wu = u * (w[:, None] if u.ndim == 2 else w)
        return wu - rho * (Z * w[:, None]) @ np.linalg.solve(K, Z.T @ wu)

    Vi_X = vinv(X)
    theta = np.linalg.lstsq(X.T @ Vi_X, Vi_X.T @ y, rcond=None)[0]
    r = y - X @ theta
    n = len(y)
    sigma2 = float(r @ vinv(r)) / n
    logdet = -np.sum(np.log(w)) + 2 * np.sum(np.log(np.diag(Kc)))
    return -0.5 * (n * np.log(sigma2) + logdet + n), sigma2


def fit(g, leagues, seasons):
    X, Z, B, steps = designs(g, leagues, seasons)
    y, w = g.y.to_numpy(), g.n.to_numpy(float)
    # rho = tau^2 / sigma^2 by maximum likelihood (sigma^2 profiled out); the step's prior
    # is fixed, so it's left out here
    best = None
    for rho in RHO_GRID:
        ll, s2 = log_likelihood(y, X, Z, w, rho)
        if best is None or ll > best[0]:
            best = (ll, rho, s2)
    _, rho, sigma2 = best
    tau = float(np.sqrt(rho * sigma2))

    # Posterior mode and covariance of [theta, delta, step]
    D = np.hstack([X, Z, B])
    prior = np.concatenate([np.zeros(X.shape[1]),
                            np.full(Z.shape[1], sigma2 / max(tau, 1e-3) ** 2),
                            np.full(B.shape[1], sigma2 / STEP_SD ** 2)])
    M = D.T @ (D * w[:, None]) + np.diag(prior) + 1e-9 * np.eye(D.shape[1])
    est = np.linalg.solve(M, D.T @ (w * y))
    cov = np.linalg.inv(M) * sigma2
    return dict(est=est, cov=cov, nX=X.shape[1], nZ=Z.shape[1], steps=steps, tau=tau, sigma2=sigma2)


def table(g, leagues, seasons, pooled, f):
    nX, nS = f["nX"], len(seasons)
    players = pd.concat([g[["season", "league_a", "n"]].rename(columns={"league_a": "league"}),
                         g[["season", "league_b", "n"]].rename(columns={"league_b": "league"})]
                        ).groupby(["league", "season"]).n.sum()
    rows = []
    for i, lg in enumerate(leagues):
        for j, s in enumerate(seasons):
            v = np.zeros(len(f["est"]))
            v[nX + i * nS + j] = 1.0
            if lg in f["steps"] and s >= RULE_CHANGE:
                v[nX + f["nZ"] + f["steps"].index(lg)] = 1.0
            dev = float(v @ f["est"])
            se = float(np.sqrt(v @ f["cov"] @ v))
            rows.append({"league": lg, "season": s, "factor": pooled[lg] * np.exp(dev), "log_se": se,
                         "network_factor": pooled[lg], "players": int(players.get((lg, s), 0))})
    rows += [{"league": "nhl", "season": s, "factor": 1.0, "log_se": 0.0, "network_factor": 1.0,
              "players": int(players.get(("nhl", s), 0))} for s in seasons]
    return pd.DataFrame(rows).sort_values(["league", "season"]).reset_index(drop=True)


def write_report(df, f, seasons):
    at = f["nX"] + f["nZ"]
    steps = {lg: (f["est"][at + k], np.sqrt(f["cov"][at + k, at + k])) for k, lg in enumerate(f["steps"])}
    lines = ["# Season-by-season NHLe factors", "",
             "Generated by `training/seasons.py` — don't edit by hand.", "",
             f"The network factors ([NETWORK.md](NETWORK.md)), allowed to move by season: each league-season's crossovers, "
             f"pooled per league pair, shift it from the league's network level, with a N(0, τ²) prior on the shift. "
             f"τ is fit by maximum likelihood: **τ = {f['tau']:.2f}** (σ = {np.sqrt(f['sigma2']):.2f} per player). "
             f"A τ near 0 means seasons don't differ by more than their crossovers' noise.", "",
             f"Rule change: from {RULE_CHANGE}-{(RULE_CHANGE + 1) % 100:02d} CHL players keep NCAA eligibility. "
             f"The NCAA and the CHL leagues get a step from that season on, with a N(0, {STEP_SD}²) prior on its log, "
             f"so it stays near 1 until seasons after the change say otherwise.", "",
             "| League | Step from " + f"{RULE_CHANGE}-{(RULE_CHANGE + 1) % 100:02d}" + " | ± (log) |", "|---|---|---|"]
    lines += [f"| {lg} | ×{np.exp(b):.3f} | {se:.3f} |" for lg, (b, se) in steps.items()]
    lines += ["", "Factors by season (`*` = the step applies):", "",
              "| League | Network | " + " | ".join(f"{s % 100:02d}-{(s + 1) % 100:02d}" for s in seasons) + " |",
              "|---|---|" + "---|" * len(seasons)]
    for lg, d in df[df.league != "nhl"].groupby("league", sort=False):
        d = d.set_index("season")
        cells = [f"{d.factor[s]:.3f}" + ("*" if lg in steps and s >= RULE_CHANGE else "") for s in seasons]
        lines.append(f"| {lg} | {d.network_factor.iloc[0]:.3f} | " + " | ".join(cells) + " |")
    lines += ["", "Re-run with `python training/network.py`, then `python training/seasons.py`.", ""]
    (HERE / "SEASONS.md").write_text("\n".join(lines), encoding="utf-8")
    return steps


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, help="training data folder (default: training/data)")
    args = ap.parse_args(argv)
    if args.data:
        network.DATA = args.data

    pooled = {lg: r["factor"] for lg, r in
              json.loads((HERE / "factors.json").read_text(encoding="utf-8"))["factors"].items()}
    pairs = network.network_pairs()
    g = season_pairs(pairs, set(pooled))
    leagues = sorted(set(pooled) - {"nhl"})
    seasons = sorted(g.season.unique())
    f = fit(g, leagues, seasons)
    df = table(g, leagues, seasons, pooled, f)
    df.to_csv(HERE / "factors_by_season.csv", index=False, float_format="%.6g")
    steps = write_report(df, f, seasons)
    print(f"tau {f['tau']:.3f}, sigma {np.sqrt(f['sigma2']):.3f}, {len(g)} pair-seasons")
    for lg, (b, se) in steps.items():
        print(f"step {lg:6} x{np.exp(b):.3f} (log se {se:.3f})")


if __name__ == "__main__":
    main()
