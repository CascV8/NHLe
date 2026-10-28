// Cloudflare Pages Function behind nhle.html: player search, a player's career by league, draft
// classes and league leaderboards, from Elite Prospects (when the EP_API_KEY secret is set) or the
// NHL's own APIs (which have no CORS headers, hence this proxy). Every upstream URL is built here
// from validated parameters, so it can't be used as an open proxy. Responses are edge-cached.
//
//   GET /api/sources                        {"ep": true|false}
//   GET /api/search?q=bedard                [{src, id, name, pos, team, league, born, nation}]
//   GET /api/player?src=ep|nhl&id=123       {src, id, name, position, birthDate, nation, headshot, draft, stats}
//   GET /api/draft?year=2025                [{src, id, name, pos, round, overall, team, club, league, nation}]
//   GET /api/league?league=ohl&season=2025  [{src, id, name, pos, team, gp, g, a, pts}]    (Elite Prospects only)
//   GET /api/photo?nhl=&name=&born=&seasons=ohl:2025,liiga:2024   {url, league, season}: the NHL headshot, else a
//                                           league-site photo (OHL, WHL, QMJHL, Liiga); cached until the month changes
//   GET /api/img?u=<leaguestat photo>        the photo itself, so the page can cut out its white background
//
// Elite Prospects' API (https://api.eliteprospects.com, a paid key) covers prospects before they're
// drafted and players who never are; the NHL's covers drafted and signed players only.

const NHL = "https://api-web.nhle.com/v1";
const SEARCH = "https://search.d3.nhle.com/api/v1/search/player";
const RECORDS = "https://records.nhl.com/site/api";
const EP = "https://api.eliteprospects.com/v1";
const TTL = { search: 3600, player: 3600, draft: 86400, league: 21600, sources: 300 };   // seconds at the edge

/** Seconds until the next month starts (UTC): league photos are looked up again once a month. */
export function untilNextMonth(now = new Date()) {
  return Math.max(60, Math.ceil((Date.UTC(now.getUTCFullYear(), now.getUTCMonth() + 1, 1) - now) / 1000));
}

// Player photos from league sites, for prospects without an NHL headshot (sources as in puckscrape's
// leagues/hockeytech.py and leagues/liiga.py). HockeyTech's key is the public one chl.ca sends.
const HOCKEYTECH = "https://lscluster.hockeytech.com/feed/index.php";
const HOCKEYTECH_KEY = "f1aa699db3d81487";
const HOCKEYTECH_CLIENTS = { ohl: "ohl", whl: "whl", qmjhl: "lhjmq" };
const LIIGA = "https://liiga.fi/api/v2";

const json = (body, status = 200, ttl = 0) => new Response(JSON.stringify(body), { status, headers: {
  "content-type": "application/json; charset=utf-8", "cache-control": ttl ? `public, max-age=${ttl}` : "no-store" } });
class HttpError extends Error { constructor(status, msg) { super(msg); this.status = status; } }

async function get(url) {
  const r = await fetch(url, { headers: { "user-agent": "nhl-microstats NHLe cards", accept: "application/json" } });
  if (r.status === 404) throw new HttpError(404, "not found");
  if (!r.ok) throw new HttpError(502, `upstream ${r.status}`);
  return r.json();
}
const ep = (env, path, params = {}) =>
  get(`${EP}${path}?${new URLSearchParams({ ...params, apiKey: env.EP_API_KEY })}`).then(d => d.data ?? d);

// ── Normalizers (exported for the tests) ───────────────────────────────────────────────────────

/** A season as its start year: 20222023, "2022-2023", "2022-23" or {startYear} -> 2022. */
export function startYear(s) {
  if (s && typeof s === "object") return s.startYear ?? startYear(s.slug ?? s.name);
  const m = String(s).match(/^(\d{4})/);
  return m ? +m[1] : null;
}

export function nhlPlayer(d) {
  const seen = new Set(), stats = [];
  for (const s of d.seasonTotals || []) {
    const row = { season: startYear(s.season), type: s.gameTypeId === 3 ? "po" : "reg", league: s.leagueAbbrev,
      team: s.teamName?.default ?? "", gp: s.gamesPlayed || 0, g: s.goals || 0, a: s.assists || 0, pts: s.points || 0 };
    const key = JSON.stringify(row);   // the NHL lists some stints twice
    if (!seen.has(key)) { seen.add(key); stats.push(row); }
  }
  const dd = d.draftDetails;
  // Players the NHL hasn't photographed get a generic silhouette; treat that as no photo.
  const headshot = d.headshot && !/default|silhouette/i.test(d.headshot) ? d.headshot : null;
  return { src: "nhl", id: String(d.playerId), name: `${d.firstName?.default ?? ""} ${d.lastName?.default ?? ""}`.trim(),
    position: d.position, birthDate: d.birthDate, nation: d.birthCountry, headshot,
    draft: dd ? { year: dd.year, round: dd.round, overall: dd.overallPick, team: dd.teamAbbrev } : null, stats };
}

const epStat = (block, k) => +(block?.[k] ?? block?.[k.toLowerCase()] ?? 0) || 0;

/** Elite Prospects player + player-stats + draft-selections -> the same shape as nhlPlayer. */
export function epPlayer(p, lines, drafts) {
  const stats = [];
  for (const l of lines || []) {
    const base = { season: startYear(l.season), league: l.league?.slug ?? l.league?.name ?? l.leagueName ?? "",
      team: l.teamName ?? l.team?.name ?? "" };
    for (const [type, block] of [["reg", l.regularStats], ["po", l.postseasonStats]]) {
      const gp = epStat(block, "GP");
      if (gp) stats.push({ ...base, type, gp, g: epStat(block, "G"), a: epStat(block, "A"), pts: epStat(block, "PTS") });
    }
  }
  const nhlDraft = (drafts || []).find(x => /nhl/i.test(x.draftType?.slug ?? x.draftType?.name ?? "nhl"));
  return { src: "ep", id: String(p.id), name: p.name ?? `${p.firstName ?? ""} ${p.lastName ?? ""}`.trim(),
    position: p.position ?? p.detailedPosition?.[0], birthDate: p.dateOfBirth, nation: p.nationality?.name ?? null,
    headshot: p.imageUrl ?? null,
    draft: nhlDraft ? { year: nhlDraft.year, round: nhlDraft.round, overall: nhlDraft.overall,
      team: nhlDraft.team?.name ?? nhlDraft.team?.shortName ?? "" } : null, stats };
}

/** A name for matching across sites: no accents, case or punctuation; "Last, First" turned around. */
export function nameKey(name) {
  let s = String(name || "");
  if ((s.match(/, /g) || []).length === 1) { const [last, first] = s.split(", "); s = `${first} ${last}`; }
  return s.normalize("NFKD").replace(/[̀-ͯ]/g, "").toLowerCase().replace(/[^a-z ]/g, "").replace(/\s+/g, " ").trim();
}

/** HockeyTech's regular-season id for a start year; names vary ("2025-26 Regular Season", "2025 - 26 ..."). */
export function hockeytechSeason(seasons, year) {
  for (const s of seasons) {
    const name = s.season_name.toLowerCase(), years = (name.match(/\d{4}/g) || []).map(Number);
    if (years[0] === year && name.includes("regular") && !/pre|prospect|showcase/.test(name)) return s.season_id;
  }
  return null;
}

async function hockeytechPhoto(league, year, name, born) {
  const kit = params => get(`${HOCKEYTECH}?${new URLSearchParams({ feed: "modulekit", key: HOCKEYTECH_KEY,
    client_code: HOCKEYTECH_CLIENTS[league], fmt: "json", lang: "en", ...params })}`).then(d => d.SiteKit);
  const season = hockeytechSeason((await kit({ view: "seasons" })).Seasons || [], year);
  if (!season) return null;
  const rows = (await kit({ view: "statviewtype", type: "topscorers", season_id: season, first: 0, limit: 5000 })).Statviewtype || [];
  for (const r of rows.filter(r => nameKey(r.name) === nameKey(name))) {
    const p = (await kit({ view: "player", category: "profile", player_id: r.player_id })).Player || {};
    if (p.primary_image && (!born || !p.birthdate || p.birthdate === born)) return p.primary_image;   // same name, same birthday
  }
  return null;
}

async function liigaPhoto(year, name) {
  const rows = await get(`${LIIGA}/players/stats/summed/${year + 1}/${year + 1}/runkosarja/false?dataType=basicStats`);
  const hits = rows.filter(r => r.pictureUrl && nameKey(`${r.firstName} ${r.lastName}`) === nameKey(name));
  return hits.length === 1 ? hits[0].pictureUrl : null;   // no birthdates here: only take an unambiguous match
}

// ── Routes ─────────────────────────────────────────────────────────────────────────────────────

const ID = /^\d{1,10}$/, YEAR = /^(19|20)\d\d$/, SLUG = /^[a-z0-9-]{2,40}$/;
const NHL_HEADSHOT = /^https:\/\/assets\.nhle\.com\/mugs\/[\w/.-]{1,120}\.png$/;
const LEAGUESTAT_PHOTO = /^https:\/\/assets\.leaguestat\.com\/[a-z]{2,10}\/\d{2,4}x\d{2,4}\/\d{1,10}\.jpg$/;
const LIIGA_PHOTO = /^https:\/\/liiga-backend-prod\.s3\.eu-north-1\.amazonaws\.com\/\d{4}\/[\w%.:-]{1,120}\/\d{1,12}\.png$/;

const routes = {
  sources: async (_, env) => ({ ep: !!env.EP_API_KEY }),

  // ?nhl=<NHL headshot URL>&name=Nikita Klepov&born=2008-06-27&seasons=ohl:2025,ohl:2024 (newest first)
  //   -> {url, league, season}: the NHL headshot if it's a real photo, else a league-site photo. Cached for the month.
  async photo(q) {
    const name = (q.get("name") || "").slice(0, 80), born = q.get("born") || "", nhl = q.get("nhl") || "";
    if (nameKey(name).length < 3 || (born && !/^\d{4}-\d\d-\d\d$/.test(born))) throw new HttpError(400, "bad name or birthdate");
    if (nhl && !NHL_HEADSHOT.test(nhl)) throw new HttpError(400, "bad NHL headshot");
    if (nhl) {
      // A prospect the NHL hasn't photographed has a headshot URL that redirects to a generic silhouette.
      const r = await fetch(nhl, { method: "HEAD", redirect: "manual" }).catch(() => null);
      if (r?.status === 200) return { url: nhl, league: "nhl" };
    }
    const tries = (q.get("seasons") || "").split(",").slice(0, 4).map(s => s.split(":"))
      .filter(([lg, y]) => (lg in HOCKEYTECH_CLIENTS || lg === "liiga") && YEAR.test(y || ""));
    for (const [league, y] of tries) {
      const url = await (league === "liiga" ? liigaPhoto(+y, name) : hockeytechPhoto(league, +y, name, born)).catch(() => null);
      // HockeyTech's photos are JPEGs on a white studio background: the page removes it (via api/img, for CORS).
      if (url) return { url, league, season: +y, ...(league !== "liiga" && { cutout: `api/img?u=${encodeURIComponent(url)}` }) };
    }
    return { url: null };
  },

  // ?u=<photo URL>: the image itself, relayed so the page's canvas may read its pixels (to cut out a HockeyTech
  // photo's background, or to draw the card into a downloadable PNG). Only the card's photo sources.
  async img(q) {
    const u = q.get("u") || "";
    if (![LEAGUESTAT_PHOTO, NHL_HEADSHOT, LIIGA_PHOTO].some(re => re.test(u))) throw new HttpError(400, "not a card photo");
    const r = await fetch(u, { headers: { "user-agent": "nhl-microstats NHLe cards" } });
    if (!r.ok || !/^image\//.test(r.headers.get("content-type") || "")) throw new HttpError(502, `upstream ${r.status}`);
    return new Response(r.body, { headers: { "content-type": r.headers.get("content-type"),
      "cache-control": `public, max-age=${untilNextMonth()}` } });
  },

  async search(q, env) {
    const term = (q.get("q") || "").trim().slice(0, 60);
    if (term.length < 2) return [];
    if (env.EP_API_KEY) {
      const rows = await ep(env, "/players", { q: term, limit: 20 });
      return rows.map(p => ({ src: "ep", id: String(p.id), name: p.name, pos: p.position ?? "",
        team: p.latestStats?.team?.name ?? "", league: p.latestStats?.league?.name ?? "",
        born: startYear(p.dateOfBirth), nation: p.nationality?.name ?? "" }));
    }
    const rows = await get(`${SEARCH}?${new URLSearchParams({ culture: "en-us", limit: 20, q: term })}`);
    return rows.map(p => ({ src: "nhl", id: String(p.playerId), name: p.name, pos: p.positionCode ?? "",
      team: p.teamAbbrev || p.lastTeamAbbrev || "", league: "", born: null, nation: p.birthCountry ?? "", active: p.active }));
  },

  async player(q, env) {
    const src = q.get("src"), id = q.get("id") || "";
    if (!ID.test(id)) throw new HttpError(400, "bad id");
    if (src === "ep") {
      if (!env.EP_API_KEY) throw new HttpError(400, "Elite Prospects isn't configured");
      const [p, lines, drafts] = await Promise.all([ep(env, `/players/${id}`),
        ep(env, "/player-stats", { player: id, limit: 1000 }),
        ep(env, "/draft-selections", { player: id }).catch(() => [])]);
      return epPlayer(p, lines, drafts);
    }
    return nhlPlayer(await get(`${NHL}/player/${id}/landing`));   // headshot unchecked: api/photo checks it, monthly
  },

  async draft(q) {
    const year = q.get("year") || "";
    if (!YEAR.test(year)) throw new HttpError(400, "bad year");
    const d = await get(`${RECORDS}/draft?${new URLSearchParams({ cayenneExp: `draftYear=${year}`, sort: "overallPickNumber" })}`);
    return (d.data || []).filter(p => p.playerId).sort((a, b) => a.overallPickNumber - b.overallPickNumber).map(p => ({
      src: "nhl", id: String(p.playerId), name: p.playerName, pos: p.position, round: p.roundNumber,
      overall: p.overallPickNumber, team: p.triCode, club: p.amateurClubName, league: p.amateurLeague, nation: p.countryCode }));
  },

  async league(q, env) {
    const league = q.get("league") || "", year = q.get("season") || "";
    if (!SLUG.test(league) || !YEAR.test(year)) throw new HttpError(400, "bad league or season");
    if (!env.EP_API_KEY) throw new HttpError(400, "Browsing by league needs Elite Prospects");
    const rows = await ep(env, "/player-stats", { league, season: `${year}-${+year + 1}`,
      sort: "-regularStats.PTS", limit: 150 });
    return rows.filter(l => l.player).map(l => ({ src: "ep", id: String(l.player.id), name: l.player.name,
      pos: l.player.position ?? "", born: startYear(l.player.dateOfBirth), team: l.teamName ?? l.team?.name ?? "",
      gp: epStat(l.regularStats, "GP"), g: epStat(l.regularStats, "G"), a: epStat(l.regularStats, "A"),
      pts: epStat(l.regularStats, "PTS") }));
  },
};

export async function onRequestGet({ request, env, params }) {
  const name = (params.path || [])[0];
  const route = routes[name];
  if (!route) return json({ error: "not found" }, 404);
  const url = new URL(request.url);
  const cache = typeof caches !== "undefined" ? caches.default : null;
  // The source is part of the answer, so the cache key carries it: adding the key doesn't serve stale NHL rows.
  // Photos carry the month in their key, so every lookup starts fresh on the 1st.
  const month = name === "photo" || name === "img" ? `&_m=${new Date().toISOString().slice(0, 7)}` : "";
  const key = new Request(`${url.origin}${url.pathname}?${url.searchParams}&_ep=${env.EP_API_KEY ? 1 : 0}${month}`);
  const hit = cache && await cache.match(key);
  if (hit) return hit;
  try {
    const out = await route(url.searchParams, env);
    const res = out instanceof Response ? out : json(out, 200, name === "photo" ? untilNextMonth() : TTL[name]);
    if (cache) await cache.put(key, res.clone());
    return res;
  } catch (e) {
    return json({ error: e.message }, e.status || 502);
  }
}
