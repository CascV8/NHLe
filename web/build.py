"""
Build the NHLe prospect cards into a website.

  python web/build.py SITE_DIR [FUNCTIONS_DIR]

Writes SITE_DIR/nhle.html, nhle.js, nhle-model.json and nhle-logos/, and copies the Pages
Function (web/functions/) into FUNCTIONS_DIR (default: SITE_DIR/../functions).
The microstats site runs this at deploy (see nhl-microstats' update-site.yml).

nhle-model.json is everything the browser needs, read from this repo so the
site can't drift from the scripts: the league factors (the scripts'
LEAGUE_FACTORS), the coefficients (training/coefficients.json, which fit.py
writes into the scripts), and how training/fit.py builds a season (the
World Juniors each model uses, NHL league names, events, minimum games and
coverage). Nothing is imported, so it runs on a bare Python 3.8+.
"""

import ast
import json
import shutil
import sys
from pathlib import Path

WEB = Path(__file__).resolve().parent
ROOT = WEB.parent

# Elite Prospects league slugs and names that differ from the NHL API names fit.py maps.
EP_ALIASES = {
    "j20-nationell": "u20-nationell", "u20-nationell": "u20-nationell", "j20-superelit": "u20-nationell",
    "u20-sm-sarja": "u20-sm-sarja", "u20-sm-liiga": "u20-sm-sarja", "czechia-u20": "czechia-u20",
    "hockeyallsvenskan": "allsvenskan", "allsvenskan": "allsvenskan", "national-league": "nl",
    "swiss-league": "sl", "wjc-20": "wjc-20", "wjc-18": "wjc-18", "dnl-u20": "dnl",
    "j18-nationell": "u18-nationell", "j18-allsvenskan": "u18-nationell", "u20-elit": "swiss-u20",
    "ligue-magnus": "france", "alpshl": "alpshl", "alps-hockey-league": "alpshl", "ice-hockey-league": "icehl",
    "icehl": "icehl", "ebel": "icehl", "metal-ligaen": "denmark", "eliteserien": "norway",
    "polska-hokej-liga": "poland", "slovakia-u20": "slovakia-u20",
    "czechia-u18": "czechia-u18", "germany3": "oberliga", "oberliga": "oberliga", "usports": "usports",
}
EP_EVENTS = {"wc", "og", "olympics", "champions-hl", "spengler-cup", "hlinka-gretzky-cup", "whc-17",
             "u17-wc", "wjac-19", "4-nations", "international", "international-jr", "exhibition"}


def assignments(path):
    """Literal top-level assignments in a Python file, without running it."""
    out = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def model_spec():
    fit = assignments(ROOT / "training" / "fit.py")
    factors = assignments(ROOT / "Forwards" / "fwd-d0.py")["LEAGUE_FACTORS"]
    coef = json.loads((ROOT / "training" / "coefficients.json").read_text(encoding="utf-8"))
    fitted = {(m["pos"], m["rel"]): m for m in coef["models"]}
    models = {"F": {}, "D": {}}
    for pos, rel, script, tournaments in fit["MODELS"]:
        m = fitted[(pos, rel)]
        models[pos][str(rel)] = {
            "script": script,
            "nhler": [m["nhler"]["intercept"], m["nhler"]["coef"]],
            "star": [m["star"]["intercept"], m["star"]["coef"]],
            # The curve each coefficient multiplies: "linear" pts/84, "log" log(1 + pts/84) or "sqrt"
            "shape": {"nhler": m["nhler"].get("shape", "linear"), "star": m["star"].get("shape", "linear")},
            # pts/84 above the top 0.5% of the model's sample is held there (no extrapolating)
            "x_cap": m.get("x_cap"),
            # World Juniors terms: {tournament: [played, per NHLe pts/84 there]}
            "wjc": {"nhler": m["nhler"].get("wjc", {}), "star": m["star"].get("wjc", {})},
            # WJC-20 slopes per year of age at the tournament, from fit["WJC_AGE_CENTER"]
            "wjc_age": {"nhler": m["nhler"].get("wjc_age", {}), "star": m["star"].get("wjc_age", {})},
            "wjc_age_typical": m.get("wjc_age_median"),
            "tournaments": sorted(tournaments),
            "known": rel >= 2,   # D+2 to D+5 report a reached outcome as 100%
            "n": m["n"],
        }
    aliases = {k.lower(): v for k, v in fit["LEAGUE_ALIASES"].items()}
    aliases.update(EP_ALIASES)
    aliases.update({k: k for k in factors})
    # Names for leagues the network solve left without a factor (too few players) are dropped
    aliases = {k: v for k, v in aliases.items() if v in factors}
    return {
        "classes": coef["classes"],
        "factors": factors,
        "aliases": aliases,
        "tournaments": sorted(fit["TOURNAMENTS"]),
        "events": sorted({e.lower() for e in fit["EVENTS"]} | EP_EVENTS),
        "min_gp": fit["MIN_GP"],
        "min_coverage": fit["MIN_COVERAGE"],
        "po_weight": coef.get("po_weight", fit["PO_WEIGHT"]),   # per position
        "wjc_age_center": fit["WJC_AGE_CENTER"],
        # The Pareto group's share of regulars per position, examples near the line, and every confirmed
        # star: NHL id -> the season (start year) of his first star season (training/stars.json).
        "star_share": coef["star_share"],
        "star_examples": coef["star_examples"],
        "stars": json.loads((ROOT / "training" / "stars.json").read_text(encoding="utf-8"))["stars"],
        "models": models,
        # League logos for the card, by league key (web/logos/fetch.py and logos.json); a league without one keeps
        # its text label. invert: a dark single-colour mark, inverted on the dark theme so it stays visible.
        "logos": {key: {"src": f"nhle-logos/{v['file']}", "invert": v["invert_dark"]}
                  for key, v in json.loads((WEB / "logos" / "logos.json").read_text(encoding="utf-8")).items()
                  if key in factors},
    }


def build(site, functions=None):
    site = Path(site)
    functions = Path(functions) if functions else site.parent / "functions"
    site.mkdir(parents=True, exist_ok=True)
    shutil.copy(WEB / "nhle.html", site / "nhle.html")
    shutil.copy(WEB / "nhle.js", site / "nhle.js")
    spec = model_spec()
    (site / "nhle-model.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    (site / "nhle-logos").mkdir(exist_ok=True)
    for logo in spec["logos"].values():
        shutil.copy(WEB / "logos" / Path(logo["src"]).name, site / logo["src"])
    shutil.copytree(WEB / "functions", functions, dirs_exist_ok=True)
    return site


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    print(f"NHLe cards written to {build(*sys.argv[1:])}")
