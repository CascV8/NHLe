"""
NHL Pareto Principle Analysis
==============================
For each season, finds the "optimal" Pareto split among skaters based on GAR:
the top-N skaters (by GAR) for which (percent of all skaters) + (percent of
total GAR they contributed) sums closest to exactly 1.

Skaters = forwards + defensemen (goalies excluded). Players traded mid-season
have their games and GAR merged into a single player-season row before ranking.
"""

from pathlib import Path

import pandas as pd
import numpy as np

# ── Load data ─────────────────────────────────────────────────────────────────
# Paths are relative to this file so the script runs from any directory
HERE        = Path(__file__).resolve().parent
DATA_PATH   = HERE / "NHL-GAR-data-v2.5-1956-2022.csv"
OUTPUT_PATH = HERE / "pareto_results_by_season.csv"
df = pd.read_csv(DATA_PATH, encoding='latin-1')

# ── Parameters ────────────────────────────────────────────────────────────────
POSITIONS = ['F', 'D']   # forwards + defensemen = "skaters" (goalies excluded)
MIN_GP    = None         # None = no games-played floor

# ── Confirmed column names ─────────────────────────────────────────────────────
COL_SEASON   = 'year_ID'
COL_PLAYER   = 'player_ID'
COL_GP       = 'gp'
COL_POSITION = 'maj_pos'
COL_GAR      = 'tot_gar'   # swap to 'adj_gar' for the schedule/roster-adjusted version

skaters = df[df[COL_POSITION].isin(POSITIONS)].copy()

# Merge mid-season trades into one row per player per season (sum GP and GAR
# across all teams played for that year) -- without this, traded players get
# double-counted and inflate the total skater count for that season.
skaters = skaters.groupby([COL_PLAYER, COL_SEASON], as_index=False).agg(
    gp=(COL_GP, 'sum'),
    tot_gar=(COL_GAR, 'sum')
)

if MIN_GP is not None:
    skaters = skaters[skaters['gp'] >= MIN_GP]


def pareto_table(season_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full Top_n_Players / Percent_Of_Players / Percent_of_GAR /
    Distance_From_1 table for one season, ranked by GAR descending.
    """
    ranked = season_df.sort_values('tot_gar', ascending=False).reset_index(drop=True)
    n_total = len(ranked)
    total_gar = ranked['tot_gar'].sum()

    top_n = np.arange(1, n_total + 1)
    cum_gar = ranked['tot_gar'].cumsum().to_numpy()

    percent_of_players = top_n / n_total
    percent_of_gar = cum_gar / total_gar
    distance_from_1 = np.abs(percent_of_players + percent_of_gar - 1)

    return pd.DataFrame({
        'Top_n_Players': top_n,
        'Percent_Of_Players': percent_of_players.round(4),
        'Percent_of_GAR': percent_of_gar.round(4),
        'Distance_From_1': distance_from_1.round(4),
    })


def optimal_pareto_value(season_df: pd.DataFrame) -> pd.Series:
    """Return the single row (top_n) that minimizes Distance_From_1."""
    table = pareto_table(season_df)
    return table.loc[table['Distance_From_1'].idxmin()]


def print_pareto_window(season_df: pd.DataFrame, season: int, window: int = 8) -> None:
    """
    Print the rows leading up to AND past the optimal Pareto point, so the
    distance can be seen shrinking to a minimum and then growing again --
    proof that the optimal point really is a minimum, not just a low spot.
    """
    table = pareto_table(season_df)
    best_idx = table['Distance_From_1'].idxmin()
    lo = max(0, best_idx - window)
    hi = min(len(table), best_idx + window + 1)
    view = table.loc[lo:hi - 1, ['Top_n_Players', 'Percent_Of_Players', 'Percent_of_GAR', 'Distance_From_1']].copy()
    view['Percent_Of_Players'] = (view['Percent_Of_Players'] * 100).round(2)
    view['Percent_of_GAR'] = (view['Percent_of_GAR'] * 100).round(2)
    view = view.rename(columns={
        'Top_n_Players': 'Players',
        'Percent_Of_Players': 'Pct of Players',
        'Percent_of_GAR': 'Pct of GAR',
        'Distance_From_1': 'Distance From 1'
    })

    print(f"\nPareto point detail for {season} (players leading up to, and overshooting, the optimum):")
    print(view.to_string(index=False))
    print(f"Optimal: {int(table.loc[best_idx, 'Top_n_Players'])} players "
          f"({table.loc[best_idx, 'Percent_Of_Players']*100:.2f}% of players -> "
          f"{table.loc[best_idx, 'Percent_of_GAR']*100:.2f}% of GAR)")


# ── Run for every season present in the data ──────────────────────────────────
EXCLUDE_SEASONS = [2022]   # this file only has partial-season data for 2022 -- too small a sample

results = []
for season in sorted(skaters[COL_SEASON].dropna().unique()):
    if season in EXCLUDE_SEASONS:
        continue

    season_df = skaters[skaters[COL_SEASON] == season]

    if len(season_df) < 5:   # skip incomplete seasons
        continue

    best = optimal_pareto_value(season_df)
    results.append({
        'Season': int(season),
        'Total Skaters': len(season_df),
        'Optimal Top N': int(best['Top_n_Players']),
        'Percent of Players': best['Percent_Of_Players'],
        'Percent of GAR': best['Percent_of_GAR'],
        'Distance From 1': best['Distance_From_1'],
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

results_df.to_csv(OUTPUT_PATH, index=False)

# ── Print player-by-player detail around the optimal point for one example season ──
WINDOW_SEASON = 2021   # change to inspect a different season
WINDOW_SIZE   = 8      # players shown on each side of the optimal point

example_df = skaters[skaters[COL_SEASON] == WINDOW_SEASON]
print_pareto_window(example_df, WINDOW_SEASON, window=WINDOW_SIZE)