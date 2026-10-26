# NHLe

NHL equivalency (NHLe) prospect models. Each model turns a player's season into NHL-equivalent
points per 82 games, then gives two logistic-regression probabilities:

- **NHL player**: reaches 200 NHL games
- **Star**: peak GAR (average of best 3 seasons) in the top 24% of NHL skaters at the position
  (the Pareto cut): 12.7+ for forwards, 11.0+ for defensemen ([Neil Paine's GAR](#credits))

There is one model per position and season relative to the draft year, D-1 (age ~17) through
D+5 (age ~23), each fit on the 2006–2018 draft classes (see [Training](#training)).

```
  [★★★★★★★★★★★★★★★███████████░░░░░░░░░░░░░░]
  ★★ Star (top 24% GAR) : 38.3%
  ██ NHL player (200 GP+): 66.2%  (includes star %)
  ░░ Bust (<200 GP)      : 33.8%
```

## Install

The models only need Python 3.8+ and the standard library. The card generator and Pareto analysis
need a few packages:

```
git clone https://github.com/CascV8/NHLe.git
cd NHLe
pip install -r requirements.txt
```

## Use

### Prospect models

| Season | Forwards | Defense | Tournaments |
|---|---|---|---|
| D-1 (~17) | `Forwards/fwd-d-1.py` | `Defense/d-d-1.py` | WJC-18, WJC-20 |
| Draft year (~18) | `Forwards/fwd-d0.py` | `Defense/d-d0.py` | WJC-18, WJC-20 |
| D+1 (~19) | `Forwards/fwd-d1.py` | `Defense/d-d1.py` | WJC-20 |
| D+2 (~20) | `Forwards/fwd-d2.py` | `Defense/d-d2.py` | WJC-20 |
| D+3 to D+5 | `Forwards/fwd-d3.py` … `fwd-d5.py` | `Defense/d-d3.py` … `d-d5.py` | none |

Run one interactively and enter each league stint. League splits and playoffs are pooled into a
single pts/82; the World Juniors are asked for separately:

```
python Forwards/fwd-d0.py
```

Or skip the prompts if you already have an NHLe pts/82 (no World Juniors):

```
python Forwards/fwd-d0.py --pts82 25
```

Type `leagues` at any league prompt to list the built-in NHLe factors (40 leagues, from
[`training/NETWORK.md`](training/NETWORK.md)). You can also enter a custom factor between 0 and 1
in place of a league name.

The World Juniors aren't pooled into pts/82, where 7 games would barely move 60 games of league
play. Each tournament a model uses is its own fitted term: making the team adds `PLAYED` to the
log-odds, and scoring there adds `PER_PTS82` per NHLe pts/82 at the tournament. The WJC-20 terms
also depend on the player's age at the tournament: one line in age per position, fit across D-1
to D+2, so making an under-20 team at 17 counts for more than at 19
([`training/REPORT.md`](training/REPORT.md) lists them all). D+2 to D+5 also ask whether the player has
already reached 200 GP or star GAR, and if so report that outcome as 100%.

The scripts are generated from `training/model_template.py.in` by `training/fit.py --write`; edit
the template, not the scripts.

### Player card

Edit the values at the top of `graphic/graphic.py` (name, draft info, and the NHLe / probability
values per season), then run:

```
python graphic/graphic.py
```

The PNG is written to `graphic/output/`. League logos are optional: drop them in
`graphic/logos/leagues/` named after the league (for example `WHL.png`). Grey placeholders are
used otherwise.

### On the web

The same card, built live for any player you search, is on the microstats site (`nhle.html`).
The code is in [`web/`](web/README.md) and reads its coefficients and league factors from this repo.

### Pareto analysis

`pareto/pareto.py` finds, for each NHL season through 2024-25, the top-N skaters by GAR where
(share of skaters) + (share of total GAR) comes closest to 1. It uses Neil Paine's GAR in
`data/neil-paine/` and writes `pareto/pareto_results_by_season.csv`.

```
python pareto/pareto.py
```

## Training

The coefficients come from `training/`, using [puckscrape](https://github.com/CascV8/Puckscrape):

```
pip install git+https://github.com/CascV8/Puckscrape.git
python training/scrape.py      # ~2 hours the first time; writes training/data/ (not committed)
python training/network.py     # network NHLe factors -> training/factors.json, NETWORK.md
python training/fit.py --write # refits all 14 models and regenerates the scripts
```

- **League factors:** network NHLe, the method from
  [What's a point worth?](https://beyondtheweiwu.substack.com/p/whats-a-point-worth). A league's
  NHLe is the NHL points per game of the players in both it and the NHL that season, over their
  points per game in the league; leagues with few NHL crossovers go through connector leagues
  (PPG in the connector / PPG in the league × the connector's NHLe), each path weighted by its
  share of players. Leagues are solved outward from the NHL, each from the connectors solved
  before it. The crossovers are every same-season pair (10+ games in each league) from 2010-11 to
  2025-26: NHL career histories of every pick 2000-2025 and every NHL skater since 2000-01, plus
  league-website tables from puckscrape (AHL, ECHL, CHL, junior A, Sweden, Finland, Switzerland,
  Germany, Czechia, Austria, the UK...) linked by name within a country. Leagues whose factor has a
  bootstrap error above ~28% are left out of the scripts.
- **Players:** every NHL draft pick from 2006 to 2018, split by draft-listed position.
- **Stats:** each pick's season-by-season totals in every league, from the NHL's player pages.
  (Elite Prospects blocks scraping.) League names are mapped onto the factor table. A season is
  used only if at least half its games are in leagues with a factor; high school, U18 AAA and
  similar seasons are left out, just as they can't be entered in the scripts. The NTDP's games
  outside the USHL are treated like exhibitions, so its USHL games rate the season.
- **Outcomes:** 200 NHL games, and peak GAR from Neil Paine's data through 2024-25. Later seasons
  use an estimate of his GAR from MoneyPuck stats (R² 0.94–0.95 on held-out seasons), rescaled
  to his league-wide level.
- **Fit:** one logistic regression per model and outcome on NHLe pts/82, plus the World Juniors
  terms (lightly ridged, so a tournament with a handful of players in the sample stays near zero).
  [`training/REPORT.md`](training/REPORT.md) lists sample sizes, AUC, and log loss in-sample,
  leave-one-draft-class-out, with the World Juniors pooled into pts/82 instead, and against the
  previous coefficients and factors.

## Credits

- GAR data: [Neil Paine](https://github.com/Neil-Paine-1/NHL-Player-And-Team-Ratings),
  `data/neil-paine/nhl_gar_historical.csv`, MIT license, Copyright (c) 2024 Neil Paine
  (full text in [`data/neil-paine/LICENSE.txt`](data/neil-paine/LICENSE.txt)).
- Draft picks and player career stats: the NHL's public APIs. Advanced stats:
  [MoneyPuck](https://moneypuck.com). Both are fetched by [puckscrape](https://github.com/CascV8/Puckscrape)
  when you retrain and are not included in this repo.

## Tests

```
python -m unittest discover -s tests
```

The tests check that every model gives valid, monotonic probabilities, that the league-factor
tables agree across models, that bad input is rejected, and that `--pts82` runs end to end.
`tests/test_web.py` and `node --test web/tests/nhle.test.mjs` check the website against the scripts.
