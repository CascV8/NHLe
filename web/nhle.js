// NHLe prospect cards: the browser side of the models. The numbers all come from nhle-model.json,
// which web/build.py writes from the scripts, training/fit.py and training/coefficients.json.
//
// A season is pooled the way training/fit.py pools it for the fit: regular season and playoffs in
// every league with a factor, into pts/82 = sum(points x factor) / sum(games) x 82, a playoff game
// counting as spec.po_weight games. The World Juniors
// that model uses are not pooled: each is its own term, played + per_pts82 x (NHLe pts/82 there), and
// for the WJC-20 both move with the player's age at the tournament.
// A season gets probabilities only if it's one the models were fit on: 10+ regular-season games in
// rated leagues, and at least half of the season's league games in leagues with a factor
// (tournaments and short events aside).

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

/**
 * The World Juniors log-odds for one outcome. ``wjc`` maps a tournament to its NHLe pts/82; for the
 * WJC-20 both terms also move by ``ageTerms`` per year of ``age`` (at the tournament) from ``center``.
 */
export function wjcLogOdds(terms, ageTerms, wjc, age, center) {
  return Object.entries(wjc || {}).reduce((t, [k, x]) => {
    if (!terms[k]) return t;
    const [pa, wa] = ageTerms?.[k] ?? [0, 0], a = age - center;
    return t + terms[k][0] + pa * a + (terms[k][1] + wa * a) * x;
  }, 0);
}

/** P(NHLer) and P(star). ``age`` is at the WJC-20 (default: the model's typical age there). */
export function probabilities(spec, group, rel, pts82, wjc = {}, age = null) {
  const m = spec.models[group][String(rel)];
  const a = age ?? m.wjc_age_typical ?? spec.wjc_age_center;
  const lo = k => m[k][0] + m[k][1] * pts82 + wjcLogOdds(m.wjc[k], m.wjc_age[k], wjc, a, spec.wjc_age_center);
  const nhler = sigmoid(lo("nhler"));
  return { nhler, star: Math.min(sigmoid(lo("star")), nhler) };
}

/** Age in years on January 1 of ``year`` (the World Juniors), or null without a birth date. */
export function ageOnJan1(birthDate, year) {
  const born = Date.parse(birthDate);
  return Number.isNaN(born) ? null : (Date.UTC(year, 0, 1) - born) / (365.25 * 86400e3);
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
    const used = rows.filter(r => r.used && !tournaments.has(r.key));
    const gp = used.reduce((t, r) => t + r.gp, 0);
    const regGp = used.filter(r => r.type === "reg").reduce((t, r) => t + r.gp, 0);
    const out = { rel, label: label(rel), year, rows, gp, played: gp > 0, future: year > current, inProgress: year === current };
    if (!out.played) return out;
    const w = r => r.type === "po" ? spec.po_weight ?? 1 : 1;
    out.nhlePts = used.reduce((t, r) => t + r.pts * r.factor * w(r), 0);
    out.pts82 = out.nhlePts / used.reduce((t, r) => t + r.gp * w(r), 0) * 82;
    // Each World Juniors this model uses: NHLe pts/82 at the tournament, its own term in the model.
    out.wjc = {};
    for (const r of rows.filter(r => r.used && tournaments.has(r.key))) {
      const w = out.wjc[r.key] ??= { pts: 0, gp: 0 };
      w.pts += r.pts; w.gp += r.gp;
    }
    for (const [k, w] of Object.entries(out.wjc)) out.wjc[k] = w.pts * spec.factors[k] / w.gp * 82;
    if (regGp < spec.min_gp) out.unrated = `under ${spec.min_gp} games in rated leagues`;
    else if (ratedGp < spec.min_coverage * leagueGp) out.unrated = "most games in leagues without a factor";
    if (out.unrated) return out;
    out.wjcAge = ageOnJan1(player.birthDate, year + 1);
    Object.assign(out, probabilities(spec, group, rel, out.pts82, out.wjc, out.wjcAge));
    if (model.known && nhlGpThrough(year) >= 200) { out.nhler = 1; out.known = true; }
    return out;
  });
  const latest = [...seasons].reverse().find(s => s.nhler != null);
  return { group, draftYear, drafted: !!player.draft?.year, seasons, latest };
}
