# NHLe prospect cards on the web

The card from `graphic/graphic.py`, live on the microstats site (`nhle.html`): search any player, or
browse a draft class (or a league, with Elite Prospects), and the card is built when you pick one.
Nothing is generated ahead of time.

| File | What it is |
|---|---|
| `nhle.html` | the page: search, browse, the card |
| `nhle.js` | the models in the browser, driven by `nhle-model.json` |
| `build.py` | writes the page, `nhle.js` and `nhle-model.json` into a site, and copies the Pages Function |
| `functions/api/[[path]].js` | Cloudflare Pages Function: player search, careers, draft classes, league leaderboards |

```
python web/build.py path/to/site path/to/functions
```

The microstats site's deploy workflow checks this repo out and runs that, so a push here reaches the
site on its next run (within about 5 minutes).

## The numbers come from this repo

`nhle-model.json` is read, not copied by hand: league factors from the scripts' `LEAGUE_FACTORS`,
coefficients from `training/coefficients.json`, and from `training/fit.py` the tournaments each model
pools, the NHL league-name aliases, the events left out, and the 10-game / half-coverage rule. Refit
with `training/fit.py --write`, push, and the site follows. A season gets probabilities only when it
passes the same rule the fit used; otherwise the card shows its NHLe and marks it "not rated".

Not applied on the web: the WJC team-share adjustment (it needs team scoring totals) and the star
known outcome (it needs GAR). Reaching 200 NHL games does turn NHLer to 100% from D+2 on.

## Logos and photos

- **League logos** under the NHLe line come from `web/logos/` (`python web/logos/fetch.py` re-downloads
  them), one for every league the model rates: the NHL's own shield, NCAA and the IIHF (for the World
  Juniors) from Wikimedia Commons, 16 from TheSportsDB, and the rest from the league's official site, or,
  where a junior league has no logo of its own, the federation that runs it (as the league itself does:
  Sweden's U20 Nationell, Finland's U20 SM-sarja, Germany's DNL, Latvia, Switzerland's U20-Elit). They're
  drawn transparent; `logos.json` flags the dark single-colour ones, which are inverted (the site is dark-only).
  A league without a file keeps its text label. The logos are trademarks of their owners, shown to
  identify the leagues.
- **Player photos** all come from `api/photo`: the NHL headshot when the NHL has really photographed the
  player (prospects it hasn't get a URL that redirects to a silhouette, treated as none), else the
  player's photo on the league sites puckscrape scrapes that carry photos: HockeyTech (OHL, WHL, QMJHL,
  with a birthdate check) and Liiga (an unambiguous name match only). Initials otherwise. Lookups are
  cached until the month changes, so every photo, NHL or league, is refreshed once a month.

## Data: Elite Prospects, with the NHL as fallback

The NHL's APIs only have players who were drafted or signed. Elite Prospects has everyone, including
prospects before their draft and undrafted players, but needs their paid API (their pages block
scraping). With a key stored as a Pages secret, search, careers and league browsing use it:

```
npx wrangler pages secret put EP_API_KEY --project-name nhl-microstats
```

The Elite Prospects field names in `epPlayer` were written from their documentation without a key to
test against: after adding one, open a player with `?src=ep&id=...` and adjust `epPlayer` if anything
comes back empty. League names map through `fit.py`'s aliases plus `EP_ALIASES` in `build.py`.

## Tests

```
python -m unittest discover -s tests   # tests/test_web.py: the spec matches every script
node --test web/tests/nhle.test.mjs    # the JS against the scripts' p_nhler / p_star, and the API proxy
```
