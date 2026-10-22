// NHLe prospect cards: the browser side of the models. The numbers all come from nhle-model.json,
// which web/build.py writes from the scripts, training/fit.py and training/coefficients.json.
//
// A season is pooled the way training/fit.py pools it for the fit: regular season and playoffs in
// every league with a factor, plus that model's World Juniors (regular season only), into
// pts/82 = sum(points x factor) / sum(games) x 82. It gets probabilities only if it's a season the
// models were fit on: 10+ regular-season games in rated leagues, and at least half of the season's
// league games in leagues with a factor (tournaments and short events aside).

export const SEASONS = [-1, 0, 1, 2, 3, 4, 5];
export const label = rel => rel === 0 ? "DY" : rel < 0 ? `D${rel}` : `D+${rel}`;
const sigmoid = x => 1 / (1 + Math.exp(-x));

/** The model's league key for an NHL API name or Elite Prospects slug, or null without a factor. */
export function leagueKey(spec, name) {
  const raw = String(name || "").trim().toLowerCase();
  const key = spec.aliases[raw] ?? spec.aliases[raw.replace(/\s+/g, "-")];
  return key && key in spec.factors ? key : null;
}
const isEvent = (spec, name) => {
  const raw = String(name || "").trim().toLowerCase();
  return spec.events.includes(raw) || spec.events.includes(raw.replace(/\s+/g, "-"));
};

export function probabilities(spec, group, rel, pts82) {
  const m = spec.models[group][String(rel)];
  const nhler = sigmoid(m.nhler[0] + m.nhler[1] * pts82);
  return { nhler, star: Math.min(sigmoid(m.star[0] + m.star[1] * pts82), nhler) };
}

/** "F" or "D" from any position code (C, LW, R, D, "C/LW", ...); null for goalies. */
export function positionGroup(pos) {
  const p = String(pos || "").toUpperCase();
  if (p.startsWith("G")) return null;
  return p.split("/")[0] === "D" ? "D" : "F";
}

/** First draft a player is eligible for: 18 by September 15 of the draft year. */
export function firstEligibleDraft(birthDate) {
  const [y, m, d] = String(birthDate).split("-").map(Number);
  return y + 18 + (m > 9 || (m === 9 && d > 15) ? 1 : 0);
}

/**
 * The card for one player. ``player`` is what /api/player returns:
 *   { name, position, birthDate, draft: {year, round, overall, team} | null,
 *     stats: [{ season: start year, type: "reg" | "po", league, team, gp, g, a, pts }] }
 * ``current`` is the start year of the season under way (later seasons are "not yet played").
 */
export function buildCard(spec, player, current) {
  const group = positionGroup(player.position);
  if (!group) return { group: null, seasons: [] };
  const draftYear = player.draft?.year || firstEligibleDraft(player.birthDate);
  const tournaments = new Set(spec.tournaments);
  const nhlGpThrough = y => player.stats.filter(s => s.season <= y && s.type === "reg" && leagueKey(spec, s.league) === "nhl")
    .reduce((t, s) => t + s.gp, 0);

  const seasons = SEASONS.map(rel => {
    const year = draftYear - 1 + rel, model = spec.models[group][String(rel)];   // DY starts the autumn before the draft
    let leagueGp = 0, ratedGp = 0;
    const rows = player.stats.filter(s => s.season === year).map(s => {
      const key = leagueKey(spec, s.league), tournament = tournaments.has(key);
      if (s.type === "reg" && !tournament && !isEvent(spec, s.league)) { leagueGp += s.gp; if (key) ratedGp += s.gp; }
      let why = null;
      if (!key) why = "no factor";
      else if (tournament && (s.type !== "reg" || !model.tournaments.includes(key))) why = "not in this model";
      return { ...s, key, factor: key ? spec.factors[key] : null, used: !why && s.gp > 0, why };
    });
    const used = rows.filter(r => r.used);
    const gp = used.reduce((t, r) => t + r.gp, 0);
    const regGp = used.filter(r => r.type === "reg" && !tournaments.has(r.key)).reduce((t, r) => t + r.gp, 0);
    const out = { rel, label: label(rel), year, rows, gp, played: gp > 0, future: year > current, inProgress: year === current };
    if (!out.played) return out;
    out.nhlePts = used.reduce((t, r) => t + r.pts * r.factor, 0);
    out.pts82 = out.nhlePts / gp * 82;
    if (regGp < spec.min_gp) out.unrated = `under ${spec.min_gp} games in rated leagues`;
    else if (ratedGp < spec.min_coverage * leagueGp) out.unrated = "most games in leagues without a factor";
    if (out.unrated) return out;
    Object.assign(out, probabilities(spec, group, rel, out.pts82));
    if (model.known && nhlGpThrough(year) >= 200) { out.nhler = 1; out.known = true; }
    return out;
  });
  const latest = [...seasons].reverse().find(s => s.nhler != null);
  return { group, draftYear, drafted: !!player.draft?.year, seasons, latest };
}
