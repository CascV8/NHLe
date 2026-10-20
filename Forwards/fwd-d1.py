"""
NHLe Forward D+1 Model
=======================
Predicts NHL outcomes using the season AFTER the draft year (age ~19).

Outputs two probabilities on a single bar (Iacobelli-style):
  - P(NHL player) : reaches 200 GP
  - P(Star)       : peak GAR (avg best 3 seasons) >= 14.7, top 15% of forwards

D+1 model is more confident than draft year at equivalent pts/82:
  - An extra year of data reduces projection uncertainty
  - NHL/AHL D+1 stats are especially strong signals
  - Junior D+1 returnees are systematically undervalued by pts/82 alone
    (OHL/WHL/QMJHL factors compress stats heavily)

Trained on draft classes 2012-2016 (2017+ excluded — insufficient career time
for peak GAR to emerge by end of dataset).

Star threshold: peak GAR >= ~14.7 (HR data through 2025).
  Examples: RNH, Nazem Kadri, Mark Stone, Alex DeBrincat, Robert Thomas.

WJC-20 included with team-share log-odds adjustment (players are age 19,
fully eligible). Same adjustment logic as draft year model.

Usage:
    python fwd-d1.py                  # interactive
    python fwd-d1.py --pts82 <value>  # pre-calculated pts/82

League names (no spaces, use dashes):
    nhl, ahl, whl, ohl, qmjhl, ushl, ncaa, shl, liiga, khl, etc.
    Type 'leagues' at any prompt to list all.
"""

import math
import sys

# ── Model coefficients (D+1, age ~19) ────────────────────────────────────────
# Trained on 2012-2016 draft classes, anchored to published round success rates.
# log-odds = INTERCEPT + COEF * nhl_pts_per82
NHLER_INTERCEPT = -2.335101
NHLER_COEF      =  0.147737

STAR_INTERCEPT  = -2.900780
STAR_COEF       =  0.069297

# WJC-20 team-share log-odds adjustment (same as draft year model)
WJC_ADJ_CENTER  = 0.077
WJC_ADJ_SCALE   = 0.08
WJC_ADJ_MAX_POS =  0.15  # D+1 has large reg season sample, WJC is minor signal
WJC_ADJ_MAX_NEG = -0.10
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
    "u21-elite":    0.04983498016,
    "dnl":          0.02992688863,
}


def parse_league(raw):
    key = raw.strip().lower()
    if key in LEAGUE_FACTORS:
        return key, LEAGUE_FACTORS[key]
    try:
        factor = float(key)
    except ValueError:
        raise ValueError(f"Unknown league '{raw}'. Type 'leagues' to list.") from None
    if not (0 < factor <= 1):
        raise ValueError(f"Custom NHLe factor must be between 0 and 1, got '{raw}'.")
    return key, factor


def parse_num(raw):
    """float() that rejects nan, inf and negative values."""
    val = float(raw)
    if not math.isfinite(val) or val < 0:
        raise ValueError(f"expected a non-negative number, got '{raw}'")
    return val


def wjc_log_odds_adj(player_pts, team_pts):
    if team_pts <= 0:
        return 0.0
    share = player_pts / team_pts
    z = (share - WJC_ADJ_CENTER) / WJC_ADJ_SCALE
    raw = math.tanh(z)
    return raw * WJC_ADJ_MAX_POS if raw >= 0 else raw * abs(WJC_ADJ_MAX_NEG)


def p_nhler(pts82, wjc_adj=0.0):
    lo = NHLER_INTERCEPT + NHLER_COEF * pts82 + wjc_adj
    return 1.0 / (1.0 + math.exp(-lo))


def p_star(pts82, wjc_adj=0.0):
    lo = STAR_INTERCEPT + STAR_COEF * pts82 + wjc_adj
    raw = 1.0 / (1.0 + math.exp(-lo))
    return min(raw, p_nhler(pts82, wjc_adj))


def compute_pts82(stints):
    total_nhle = sum(s["pts"] * s["factor"] for s in stints)
    total_gp   = sum(s["gp"] for s in stints)
    return (total_nhle / total_gp) * 82 if total_gp > 0 else 0.0


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
        print(f"  WJC adjustment    : {wjc_adj:+.3f} log-odds {direction}")

    print(f"  NHL pts / 82      : {pts82:.1f}")
    print(f"  {'─' * 52}")
    print()
    print(f"  {render_bar(pn, ps)}")
    print(f"  ★★ Star (top 15% GAR) : {ps:.1%}")
    print(f"  ██ NHL player (200 GP+): {pn:.1%}  (includes star %)")
    print(f"  ░░ Bust (<200 GP)      : {1-pn:.1%}")
    print()
    print("  Star ≈ RNH, Kadri, Mark Stone, DeBrincat, Robert Thomas")
    print("  D+1 model — age ~19, higher confidence than draft year.")
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
            pts = parse_num(ask("Points"))
            gp  = parse_num(ask("Games played"))
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
    print(f"\n  {label}  (optional)")
    print(f"  {'─' * 40}")
    raw = ask("Player points (or Enter to skip)", default="skip")
    if raw.lower() in ("skip", ""):
        return [], None
    try:
        player_pts = parse_num(raw)
        gp         = parse_num(ask("Games played"))
        team_pts   = parse_num(ask("Team points by forwards only (all games)"))
    except ValueError:
        print(f"    ! Invalid number, skipping.")
        return [], None
    if gp <= 0 or team_pts <= 0 or player_pts > team_pts:
        print(f"    ! Need GP > 0 and team pts >= player pts, skipping {label}.")
        return [], None
    factor = LEAGUE_FACTORS[league_key]
    share  = player_pts / team_pts
    adj    = wjc_log_odds_adj(player_pts, team_pts)
    direction = "boost" if adj >= 0 else "drag"
    print(f"    → {league_key.upper()}: {player_pts}pts/{gp}gp "
          f"({player_pts/gp:.3f} PPG) × {factor:.4f} NHLe")
    print(f"    → Fwd share: {share:.1%} of {team_pts:.0f} fwd pts "
          f"→ {adj:+.3f} log-odds {direction}")
    stint = {"kind": "wjc", "pts": player_pts, "gp": gp,
             "league_key": league_key, "factor": factor}
    meta  = {"adj": adj, "player_pts": player_pts, "team_pts": team_pts}
    return [stint], meta


def collect_wjc():
    stints20, meta20 = collect_one_tournament("WJC-20", "wjc-20")
    combined_adj = (meta20["adj"] if meta20 else 0.0)
    combined_meta = {"adj": combined_adj} if meta20 else None
    return stints20, combined_meta


def main():
    # Bars and box-drawing characters crash cp1252 when output is redirected
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = sys.argv[1:]

    if len(args) == 2 and args[0] == "--pts82":
        try:
            pts82 = parse_num(args[1])
        except ValueError as e:
            sys.exit(f"  ! --pts82: {e}")
        show_result(pts82)
        return

    print("\n  NHLe Forward D+1 Model  (season after draft, age ~19)")
    print("  " + "=" * 52)
    print("  Type 'leagues' at any league prompt to list all options.")
    print("  Include NHL stats if player cracked the roster (factor = 1.0).")

    reg = collect_stints("REGULAR SEASON  (required — add all splits)", "reg")
    if not reg:
        print("\n  ! No regular season data. Exiting."); sys.exit(1)

    po  = collect_stints("PLAYOFFS  (optional — 'done' to skip)", "po", optional=True)
    wjc, wjc_meta = collect_wjc()

    all_stints = reg + po + wjc
    pts82 = compute_pts82(all_stints)
    show_result(pts82, all_stints, wjc_meta)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\n  Cancelled.")
        sys.exit(130)
