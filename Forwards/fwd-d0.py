"""
NHLe Forward Draft Model
=========================
Outputs two probabilities on a single bar (Iacobelli-style):
  - P(NHL player) : reaches 200 GP
  - P(Star)       : top 15% peak GAR among NHL forwards (subset of NHL player)

Trained on draft classes 2008-2018 (2019+ excluded — insufficient career time).
Star threshold calibrated from Hockey-Reference GAR data (through 2025):
  Peak GAR (avg of best 3 seasons) >= ~14.7 among 200GP+ forwards.
  Examples near threshold: RNH, Nazem Kadri, Mark Stone, Alex DeBrincat, Robert Thomas.

Playoff stats pooled directly with regular season (same NHLe factor, equal weight).

WJC-20:
  Points are pooled into pts/82 via the WJC-20 NHLe factor (0.3055).
  Additionally, a log-odds adjustment is applied based on the player's share
  of their team's total points — capturing dominance independent of nation strength.
  Neutral at ~12% team share. Max boost ~+14% at 50% base. Max drag ~-8%.

Usage:
    python nhle_model.py                  # interactive
    python nhle_model.py --pts82 <value>  # skip input, use pre-calculated pts/82

League names (no spaces, use dashes):
    whl, ohl, qmjhl, ushl, ncaa, ahl, shl, liiga, khl
    u20-sm-sarja, u20-nationell, czechia-u20, u21-elite, etc.
    Type 'leagues' at any prompt to list all.
"""

import math
import sys

# ── Model coefficients ────────────────────────────────────────────────────────
# log-odds = INTERCEPT + COEF * nhl_pts_per82
NHLER_INTERCEPT = -3.158896
NHLER_COEF      =  0.238853

STAR_INTERCEPT  = -4.432684
STAR_COEF       =  0.160648

# WJC team-share adjustment (log-odds, applied after pts/82 model)
# Neutral at 12% team share. Tanh curve, capped at +0.60 / -0.40.
WJC_ADJ_CENTER  = 0.077  # neutral = average fwd share (1/13 forwards per team)
WJC_ADJ_SCALE   = 0.08   # curve width
WJC_ADJ_MAX_POS =  0.60  # max positive log-odds boost  (~+14% at 50% base)
WJC_ADJ_MAX_NEG = -0.40  # max negative log-odds drag   (~-10% at 50% base)
# ─────────────────────────────────────────────────────────────────────────────

LEAGUE_FACTORS = {
    "nhl":          1.0,
    "shl":          0.6645193365,
    "liiga":        0.6285014507,
    "khl":          0.555534104,
    "czechia":      0.523293024,
    "nl":           0.4934002709,
    "del":          0.3957446585,
    "ahl":          0.3895545856,
    "vhl":          0.3667523758,
    "slovakia":     0.3567670587,
    "ncaa":         0.3179628572,
    "allsvenskan":  0.3070513875,
    "wjc-20":       0.305254264,
    "wjc-18":       0.2207053337,
    "sl":           0.2426592162,
    "mestis":       0.228940698,
    "del2":         0.2269787804,
    "latvia":       0.2112029502,
    "whl":          0.1943040421,
    "czechia2":     0.1824508245,
    "ohl":          0.1767277524,
    "u20-sm-sarja": 0.1695791393,
    "qmjhl":        0.1673133115,
    "mhl":          0.152009593,
    "u20-nationell":0.1519723109,
    "ushl":         0.1726835286,
    "czechia-u20":  0.1037596731,
    "bchl":         0.09306532439,
    "slovakia2":    0.0915762792,
    "u21-elit":    0.04983498016,
    "dnl":          0.02992688863,
}


def parse_league(raw):
    key = raw.strip().lower()
    if key in LEAGUE_FACTORS:
        return key, LEAGUE_FACTORS[key]
    try:
        return key, float(key)
    except ValueError:
        raise ValueError(f"Unknown league '{raw}'. Type 'leagues' to list.")


def wjc_log_odds_adj(player_pts, team_pts):
    """
    Log-odds adjustment based on player's share of team's WJC scoring.
    Neutral at 7.7% fwd share (average forward on any nation, 1/13 forwards).
    Captures dominance signal independent of nationality.
    """
    if team_pts <= 0:
        return 0.0
    share = player_pts / team_pts
    z = (share - WJC_ADJ_CENTER) / WJC_ADJ_SCALE
    raw = math.tanh(z)  # smooth -1 to +1
    return raw * WJC_ADJ_MAX_POS if raw >= 0 else raw * abs(WJC_ADJ_MAX_NEG)


def p_nhler(pts82, wjc_adj=0.0):
    lo = NHLER_INTERCEPT + NHLER_COEF * pts82 + wjc_adj
    return 1.0 / (1.0 + math.exp(-lo))


def p_star(pts82, wjc_adj=0.0):
    lo = STAR_INTERCEPT + STAR_COEF * pts82 + wjc_adj
    raw = 1.0 / (1.0 + math.exp(-lo))
    return min(raw, p_nhler(pts82, wjc_adj))


def compute_pts82(stints):
    total_nhle_pts = sum(s["pts"] * s["factor"] for s in stints)
    total_gp       = sum(s["gp"] for s in stints)
    return (total_nhle_pts / total_gp) * 82 if total_gp > 0 else 0.0


def render_bar(p_nhl, p_str, width=40):
    n_star  = round(p_str * width)
    n_nhler = max(round(p_nhl * width), n_star)
    n_empty = width - n_nhler
    return "[" + "★" * n_star + "█" * (n_nhler - n_star) + "░" * n_empty + "]"


def show_result(pts82, stints=None, wjc_meta=None):
    wjc_adj = wjc_meta["adj"] if wjc_meta else 0.0
    pn = p_nhler(pts82, wjc_adj)
    ps = p_star(pts82, wjc_adj)

    print()
    print("  " + "═" * 52)

    if stints:
        for s in stints:
            tag = {"reg": "Reg    ", "po": "Playoff", "wjc": "WJC-20 "}[s["kind"]]
            ppg = s["pts"] / s["gp"]
            print(f"  {tag}  {s['league_key'].upper():<14}"
                  f"{s['pts']:>5.1f}pts/{s['gp']:>4.1f}gp "
                  f"({ppg:.3f} PPG) × {s['factor']:.4f}")
        total_nhle = sum(s["pts"] * s["factor"] for s in stints)
        total_gp   = sum(s["gp"] for s in stints)
        print(f"  {'─' * 52}")
        print(f"  Total NHLe pts    : {total_nhle:.3f} over {total_gp:.1f} GP")

    if wjc_meta:

        direction = "boost" if wjc_adj >= 0 else "drag"
        print(f"  WJC adjustment    : {wjc_adj:+.3f} log-odds {direction}")  #


    print(f"  NHL pts / 82      : {pts82:.1f}")
    print(f"  {'─' * 52}")
    print()
    print(f"  {render_bar(pn, ps)}")
    print(f"  ★★ Star (top 15% GAR) : {ps:.1%}")
    print(f"  ██ NHL player (200 GP+): {pn:.1%}  (includes star %)")
    print(f"  ░░ Bust (<200 GP)      : {1 - pn:.1%}")
    print()
    print("  Star ≈ RNH, Kadri, Mark Stone, DeBrincat, Robert Thomas")
    print("  " + "═" * 52)
    print()


def print_leagues():
    print("\n  League              NHLe Factor")
    print("  " + "-" * 34)
    for k in sorted(LEAGUE_FACTORS):
        print(f"  {k:<20} {LEAGUE_FACTORS[k]:.4f}")
    print()


def ask(prompt, default=None):
    suffix = f" [Enter={default}]" if default is not None else ""
    val = input(f"    {prompt}{suffix}: ").strip()
    return str(default) if (val == "" and default is not None) else val


def collect_stints(label, kind, optional=False):
    stints = []
    print(f"\n  {label}")
    print(f"  {'─' * 40}")
    stint_num = 1
    while True:
        prefix = f"Stint {stint_num}" if stint_num > 1 else "League"
        raw = ask(f"{prefix} (or 'done'/'leagues')",
                  default="done" if optional else None)
        if raw.lower() in ("done", ""):
            break
        if raw.lower() == "leagues":
            print_leagues()
            continue
        try:
            league_key, factor = parse_league(raw)
        except ValueError as e:
            print(f"    ! {e}"); continue
        try:
            pts = float(ask("Points"))
            gp  = float(ask("Games played"))
        except ValueError:
            print("    ! Invalid number."); continue
        if gp <= 0:
            print("    ! GP must be > 0."); continue
        stints.append({"kind": kind, "pts": pts, "gp": gp,
                        "league_key": league_key, "factor": factor})
        print(f"    → {league_key.upper()}: {pts}pts/{gp}gp "
              f"({pts/gp:.3f} PPG) × {factor:.4f} NHLe")
        more = ask("Add another stint?", default="n").lower()
        if more not in ("y", "yes"):
            break
        stint_num += 1
    return stints


def collect_one_tournament(label, league_key):
    """Collect stats for one tournament (WJC-20 or WJC-18). Returns (stint_list, meta_or_None)."""
    print(f"\n  {label}  (optional)")
    print(f"  {'─' * 40}")
    raw = ask("Player points (or Enter to skip)", default="skip")
    if raw.lower() in ("skip", ""):
        return [], None
    try:
        player_pts = float(raw)
        gp         = float(ask("Games played"))
        team_pts   = float(ask("Team points by forwards only (all games)"))
    except ValueError:
        print(f"    ! Invalid number, skipping {label}.")
        return [], None
    if gp <= 0 or team_pts <= 0:
        return [], None

    factor = LEAGUE_FACTORS[league_key]
    share  = player_pts / team_pts
    adj    = wjc_log_odds_adj(player_pts, team_pts)
    direction = "boost" if adj >= 0 else "drag"

    print(f"    → {league_key.upper()}: {player_pts}pts/{gp}gp ({player_pts/gp:.3f} PPG) × {factor:.4f} NHLe")
    print(f"    → Fwd share: {share:.1%} of {team_pts:.0f} fwd pts  →  {adj:+.3f} log-odds {direction}")

    stint = {"kind": "wjc", "pts": player_pts, "gp": gp,
             "league_key": league_key, "factor": factor}
    meta  = {"adj": adj, "player_pts": player_pts, "team_pts": team_pts, "label": label}
    return [stint], meta


def collect_wjc():
    """Collect WJC-20 and WJC-18. Returns (combined_stints, combined_meta)."""
    stints20, meta20 = collect_one_tournament("WJC-20", "wjc-20")
    stints18, meta18 = collect_one_tournament("WJC-18", "wjc-18")

    all_stints = stints20 + stints18

    # Combine log-odds adjustments — both tournaments signal projectability
    combined_adj = (meta20["adj"] if meta20 else 0.0) + (meta18["adj"] if meta18 else 0.0)
    combined_meta = {"adj": combined_adj} if (meta20 or meta18) else None

    return all_stints, combined_meta


def main():
    args = sys.argv[1:]

    if len(args) == 2 and args[0] == "--pts82":
        show_result(float(args[1]))
        return

    print("\n  NHLe Forward Draft Model")
    print("  " + "=" * 52)
    print("  Type 'leagues' at any league prompt to list all options.")

    reg = collect_stints("REGULAR SEASON  (required — add all splits)", "reg")
    if not reg:
        print("\n  ! No regular season data. Exiting."); sys.exit(1)

    po          = collect_stints("PLAYOFFS  (optional — 'done' to skip)", "po", optional=True)
    wjc, wjc_meta = collect_wjc()

    all_stints = reg + po + wjc
    pts82 = compute_pts82(all_stints)
    show_result(pts82, all_stints, wjc_meta)


if __name__ == "__main__":
    main()