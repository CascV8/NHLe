# NHLe

NHL equivalency (NHLe) prospect models. Each model turns a player's season into NHL-equivalent
points per 82 games, then gives two logistic-regression probabilities:

- **NHL player**: reaches 200 NHL games
- **Star**: peak GAR (average of best 3 seasons) of about 14.7 or more, the top 15% of NHL skaters

There is one model per season relative to the draft year: D-1 (age ~17) through D+5 (age ~23).

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
| D+1 (~19) | `Forwards/fwd-d1.py` | none yet | WJC-20 |
| D+2 (~20) | `Forwards/fwd-d2.py` | none yet | WJC-20 |
| D+3 to D+5 | `Forwards/fwd-d3.py` … `fwd-d5.py` | none yet | none |

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
forwards only (defensemen only in the defense models). The adjustment is neutral at about 7.7%
(1 of 13 forwards). D+2 to D+5 also ask whether the player has already reached 200 GP or star GAR,
and if so report that outcome as 100%.

### Player card

Edit the values at the top of `graphic/graphic.py` (name, draft info, and the NHLe / probability
values per season), then run:

```
python graphic/graphic.py
```

The PNG is written to `graphic/output/`. League logos are optional: drop them in
`graphic/logos/leagues/` named after the league (for example `WHL.png`). Grey placeholders are
used otherwise.

### Pareto analysis

`pareto/pareto.py` finds, for each season from 1956 to 2021, the top-N skaters by GAR where
(share of skaters) + (share of total GAR) comes closest to 1. It uses the bundled
`NHL-GAR-data-v2.5-1956-2022.csv` and writes `pareto/pareto_results_by_season.csv`.

```
python pareto/pareto.py
```

## Tests

```
python -m unittest discover -s tests
```

The tests check that every model gives valid, monotonic probabilities, that the league-factor
tables agree across models, that bad input is rejected, and that `--pts82` runs end to end.
