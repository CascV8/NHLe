// The website's JS against the Python scripts, and the API proxy's normalizers.
// Run from the repo root: node --test web/tests/   (needs Python on the PATH)
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { test } from "node:test";

import { buildCard, firstEligibleDraft, leagueKey, positionGroup, probabilities } from "../nhle.js";
import { epPlayer, hockeytechSeason, nameKey, nhlPlayer, onRequestGet, startYear } from "../functions/api/[[path]].js";

const ROOT = fileURLToPath(new URL("../../", import.meta.url));
function python(code) {
  for (const exe of [process.env.PYTHON, "python3", "python"].filter(Boolean)) {
    const r = spawnSync(exe, ["-c", code], { cwd: ROOT, encoding: "utf-8" });
    if (r.status === 0) return JSON.parse(r.stdout);
    if (r.error?.code !== "ENOENT" && r.stderr && !/not found|was not found/i.test(r.stderr)) throw new Error(r.stderr);
  }
  throw new Error("Python not found");
}
const SPEC = python("import json,sys; sys.path.insert(0,'web'); import build; print(json.dumps(build.model_spec()))");
const close = (a, b, tol = 1e-12) => assert.ok(Math.abs(a - b) < tol, `${a} != ${b}`);
const landing = JSON.parse(readFileSync(new URL("./fixtures/nhl_landing_bedard.json", import.meta.url), "utf-8"));

test("probabilities match every script's p_nhler and p_star", () => {
  const ref = python(`
import json, importlib.util, sys
sys.path.insert(0, 'web'); import build
out = {}
for g, models in build.model_spec()['models'].items():
    for rel, m in models.items():
        spec = importlib.util.spec_from_file_location('m', m['script']); mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out[g + rel] = [[mod.p_nhler(x), mod.p_star(x)] for x in (0, 5, 12.5, 30, 60, 90)]
print(json.dumps(out))`);
  for (const [k, rows] of Object.entries(ref)) {
    const group = k[0], rel = +k.slice(1);
    [0, 5, 12.5, 30, 60, 90].forEach((x, i) => {
      const p = probabilities(SPEC, group, rel, x);
      close(p.nhler, rows[i][0]); close(p.star, rows[i][1]);
    });
  }
});

test("Bedard's card pools leagues, playoffs and the right World Juniors", () => {
  const player = nhlPlayer(landing);
  assert.equal(player.name, "Connor Bedard");
  assert.deepEqual(player.draft, { year: 2023, round: 1, overall: 1, team: "CHI" });
  const card = buildCard(SPEC, player, 2026);
  assert.equal(card.group, "F");
  assert.deepEqual(card.seasons.map(s => s.year), [2021, 2022, 2023, 2024, 2025, 2026, 2027]);
  const [d1, dy, dp1, dp2, dp3, dp4, dp5] = card.seasons;
  const f = SPEC.factors;
  // D-1 (2021-22): WHL + WJC-18; his WJC-20 games aren't in the D-1 model.
  close(d1.pts82, (100 * f.whl + 7 * f["wjc-18"]) / 66 * 82);
  assert.equal(d1.rows.find(r => r.league === "WJC-20").why, "not in this model");
  // Draft year: WHL regular season + playoffs + WJC-20.
  close(dy.pts82, ((143 + 20) * f.whl + 23 * f["wjc-20"]) / 71 * 82);
  close(dy.nhler, probabilities(SPEC, "F", 0, dy.pts82).nhler);
  // D+1 is NHL only (the WC and "International" are events without a factor).
  close(dp1.pts82, 61 / 68 * 82);
  // 150 NHL games through D+2, 219 through D+3: NHLer becomes a known outcome from D+3.
  assert.ok(!dp2.known && dp2.nhler < 1);
  assert.ok(dp3.known); assert.equal(dp3.nhler, 1);
  // The season under way: 3 NHL games is too few to rate, so the card's headline is D+3.
  assert.ok(dp4.inProgress && dp4.played && dp4.unrated);
  assert.ok(dp5.future && !dp5.played);
  assert.equal(card.latest, dp3);
});

test("a season mostly in unrated leagues isn't rated", () => {
  const card = buildCard(SPEC, { name: "X", position: "C", birthDate: "2009-03-01", draft: null, stats: [
    { season: 2025, type: "reg", league: "OHL", team: "T", gp: 20, g: 5, a: 5, pts: 10 },
    { season: 2025, type: "reg", league: "U18 AAA", team: "T", gp: 30, g: 30, a: 30, pts: 60 },
    { season: 2025, type: "reg", league: "Hlinka Gretzky Cup", team: "C", gp: 5, g: 3, a: 3, pts: 6 }] }, 2026);
  assert.equal(card.draftYear, 2027);
  const d1 = card.seasons[0];
  assert.equal(d1.year, 2025);
  close(d1.pts82, 10 * SPEC.factors.ohl / 20 * 82);
  assert.equal(d1.unrated, "most games in leagues without a factor");   // 20 of 50 league games; the event doesn't count
});

test("league names from the NHL and Elite Prospects map to model keys", () => {
  for (const [name, key] of [["WHL", "whl"], ["HockeyAllsvenskan", "allsvenskan"], ["J20 Nationell", "u20-nationell"],
    ["j20-nationell", "u20-nationell"], ["U20 SM-liiga", "u20-sm-sarja"], ["Czechia U20", "czechia-u20"], ["Big Ten", "ncaa"],
    ["WJC-20", "wjc-20"], ["Liiga", "liiga"], ["NHL", "nhl"], ["shl", "shl"]])
    assert.equal(leagueKey(SPEC, name), key, name);
  for (const name of ["ECHL", "WJC-20 D1A", "U18 SM-sarja", "16U AAA", "", null]) assert.equal(leagueKey(SPEC, name), null, name);
});

test("positions and draft eligibility", () => {
  assert.equal(positionGroup("C"), "F"); assert.equal(positionGroup("LW/RW"), "F");
  assert.equal(positionGroup("D"), "D"); assert.equal(positionGroup("G"), null);
  assert.equal(firstEligibleDraft("2005-07-17"), 2023);   // Bedard
  assert.equal(firstEligibleDraft("2007-09-15"), 2025);   // 18 on the cut-off
  assert.equal(firstEligibleDraft("2007-09-16"), 2026);
});

test("Elite Prospects rows normalize to the same shape", () => {
  const p = epPlayer({ id: 1, name: "A B", dateOfBirth: "2008-01-02", position: "C", nationality: { name: "Sweden" } },
    [{ season: { slug: "2025-2026" }, league: { slug: "j20-nationell" }, teamName: "Frölunda HC J20",
       regularStats: { GP: 40, G: 20, A: 25, PTS: 45 }, postseasonStats: { GP: 4, G: 1, A: 2, PTS: 3 } },
     { season: { slug: "2025-2026" }, league: { slug: "shl" }, teamName: "Frölunda HC", regularStats: { GP: 0 } }],
    [{ year: 2026, round: 1, overall: 5, team: { name: "Team" }, draftType: { slug: "nhl-entry-draft" } }]);
  assert.equal(p.stats.length, 2);   // a season with no games is dropped
  assert.deepEqual(p.stats[1], { season: 2025, league: "j20-nationell", team: "Frölunda HC J20", type: "po", gp: 4, g: 1, a: 2, pts: 3 });
  assert.equal(p.draft.overall, 5);
  assert.equal(startYear(20222023), 2022); assert.equal(startYear("2022-23"), 2022); assert.equal(startYear({ startYear: 2019 }), 2019);
});

test("league photos: names match across sites, seasons by HockeyTech's naming", () => {
  assert.equal(nameKey("Massé, Maxim"), "maxim masse");        // the QMJHL writes Last, First
  assert.equal(nameKey("Juraj Slafkovský"), "juraj slafkovsky");
  assert.equal(nameKey("Ryan O'Reilly"), "ryan oreilly");
  const seasons = [{ season_id: "1", season_name: "2025-26 Pre-Season" }, { season_id: "2", season_name: "2025 - 26 Regular Season" },
    { season_id: "3", season_name: "2026 Playoffs" }, { season_id: "4", season_name: "2024-25 | Regular Season" }];
  assert.equal(hockeytechSeason(seasons, 2025), "2");
  assert.equal(hockeytechSeason(seasons, 2024), "4");
  assert.equal(hockeytechSeason(seasons, 2023), null);
  assert.equal(nhlPlayer({ ...landing, headshot: "https://assets.nhle.com/mugs/nhl/default-skater.png" }).headshot, null);
});

test("duplicate NHL stints are dropped", () => {
  const dup = { ...landing, seasonTotals: [...landing.seasonTotals, landing.seasonTotals.at(-1)] };
  assert.equal(nhlPlayer(dup).stats.length, nhlPlayer(landing).stats.length);
});

test("the API rejects bad parameters without calling upstream", async () => {
  const call = async (path, qs, env = {}) => {
    const res = await onRequestGet({ request: new Request(`https://x.dev/api/${path}?${qs}`), env, params: { path: [path] } });
    return [res.status, await res.json()];
  };
  assert.deepEqual(await call("sources", ""), [200, { ep: false }]);
  assert.equal((await call("player", "src=nhl&id=../../etc"))[0], 400);
  assert.equal((await call("draft", "year=20x5"))[0], 400);
  assert.equal((await call("league", "league=ohl&season=2025"))[0], 400);   // needs Elite Prospects
  assert.equal((await call("nope", ""))[0], 404);
  assert.deepEqual(await call("search", "q=a"), [200, []]);
  assert.equal((await call("photo", "name=x&seasons=ohl:2025"))[0], 400);
  assert.equal((await call("photo", "name=Some%20One&born=June&seasons=ohl:2025"))[0], 400);
  assert.deepEqual(await call("photo", "name=Some%20One&seasons=echl:2025,ohl:20x5"), [200, { url: null }]);   // nothing to look up
});
