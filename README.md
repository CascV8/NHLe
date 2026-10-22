# NHLe

NHL equivalency (NHLe) prospect models. Each model turns a player's season into NHL-equivalent
points per 82 games, then gives two logistic-regression probabilities:

- **NHL player**: reaches 200 NHL games
- **Star**: peak GAR (average of best 3 seasons) in the top 15% of NHL skaters at the position:
  15.2+ for forwards, 13.5+ for defensemen ([Neil Paine's GAR](#credits))

There is one model per position and season relative to the draft year, D-1 (age ~17) through
D+5 (age ~23), each fit on the 2006–2018 draft classes (see [Training](#training)).

```
  [★★★★★★★★★██████████████████████░░░░░░░░░]
  ★★ Star (top 15% GAR) : 22.6%
  ██ NHL player (200 GP+): 77.5%  (includes star %)
  ░░ Bust (<200 GP)      : 22.5%
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
| D-1 (~17) | `Forwards/fwd-d-1.py` | `Defense/d-d-1.py` | WJC-18 |
| Draft year (~18) | `Forwards/fwd-d0.py` | `Defense/d-d0.py` | WJC-18, WJC-20 |
| D+1 (~19) | `Forwards/fwd-d1.py` | `Defense/d-d1.py` | WJC-20 |
| D+2 (~20) | `Forwards/fwd-d2.py` | `Defense/d-d2.py` | WJC-20 |
| D+3 to D+5 | `Forwards/fwd-d3.py` … `fwd-d5.py` | `Defense/d-d3.py` … `d-d5.py` | none |

Run one interactively and enter each league stint. Splits, playoffs and tournaments are pooled
into a single pts/82:

```
python Forwards/fwd-d0.py
```

Or skip the prompts if you already have an NHLe pts/82:

```
python Forwards/fwd-d0.py --pts82 25
```

Type `leagues` at any league prompt to list the built-in NHLe factors. You can also enter a custom
factor between 0 and 1 in place of a league name.

WJC stats add a log-odds adjustment based on the player's share of their team's points, counting
forwards only in the forward models (neutral at 7.7%, 1 of 13 forwards) and defensemen only in the
defense models (neutral at 12.5%, 1 of 8). Its size is hand-set rather than fit. D+2 to D+5 also
ask whether the player has already reached 200 GP or star GAR, and if so report that outcome as 100%.

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
python training/scrape.py 2006 2019   # ~10 minutes; writes training/data/ (not committed)
python training/fit.py --write        # refits all 14 models and updates the scripts
```

- **Players:** every NHL draft pick from 2006 to 2018, split by draft-listed position.
- **Stats:** each pick's season-by-season totals in every league, from the NHL's player pages.
  (Elite Prospects blocks scraping.) League names are mapped onto the NHLe factor table. A season is
  used only if at least half its games are in leagues with a factor, so ECHL, NTDP, high school
  and similar seasons are left out, just as they can't be entered in the scripts.
- **Outcomes:** 200 NHL games, and peak GAR from Neil Paine's data through 2024-25. Later seasons
  use an estimate of his GAR from MoneyPuck stats (R² 0.94–0.95 on held-out seasons), rescaled
  to his league-wide level.
- **Fit:** one logistic regression per model and outcome on NHLe pts/82.
  [`training/REPORT.md`](training/REPORT.md) lists sample sizes, AUC, and log loss in-sample,
  leave-one-draft-class-out, and against the previous hand-set coefficients.

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
