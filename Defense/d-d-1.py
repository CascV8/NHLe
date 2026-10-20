"""
NHLe Defenseman D-1 Model (pre-draft year, age ~17)
=====================================================
Calibrated for defensemen — lower pts/82 thresholds than forward models.
D-men develop slower; a 17yo scoring 0.4 PPG in OHL is elite for a d-man.

WJC-18 included — team pts = defenseman pts only.

Usage:
    python nhle_model_d1_d.py
    python nhle_model_d1_d.py --pts82 <value>
"""

import math, sys

NHLER_INTERCEPT = -2.576679
NHLER_COEF      =  0.148630
STAR_INTERCEPT  = -3.439546
STAR_COEF       =  0.117490

WJC_ADJ_CENTER  = 0.077
WJC_ADJ_SCALE   = 0.08
WJC_ADJ_MAX_POS =  0.50
WJC_ADJ_MAX_NEG = -0.35

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
        return key, float(key)
    except ValueError:
        raise ValueError(f"Unknown league '{raw}'. Type 'leagues' to list.")


def wjc_log_odds_adj(player_pts, team_pts):
    if team_pts <= 0:
        return 0.0
    share = player_pts / team_pts
    z = (share - WJC_ADJ_CENTER) / WJC_ADJ_SCALE
    raw = math.tanh(z)
    return raw * WJC_ADJ_MAX_POS if raw >= 0 else raw * abs(WJC_ADJ_MAX_NEG)


def p_nhler(pts82, adj=0.0):
    return 1.0 / (1.0 + math.exp(-(NHLER_INTERCEPT + NHLER_COEF * pts82 + adj)))


def p_star(pts82, adj=0.0):
    raw = 1.0 / (1.0 + math.exp(-(STAR_INTERCEPT + STAR_COEF * pts82 + adj)))
    return min(raw, p_nhler(pts82, adj))


def compute_pts82(stints):
    total_nhle = sum(s["pts"] * s["factor"] for s in stints)
    total_gp   = sum(s["gp"] for s in stints)
    return (total_nhle / total_gp) * 82 if total_gp > 0 else 0.0


def render_bar(p_nhl, p_str, width=40):
    n_star  = round(p_str * width)
    n_nhler = max(round(p_nhl * width), n_star)
    n_empty = width - n_nhler
    return "[" + "★" * n_star + "█" * (n_nhler - n_star) + "░" * n_empty + "]"


def show_result(pts82, stints=None, wjc_adj=0.0):
    pn = p_nhler(pts82, wjc_adj)
    ps = p_star(pts82, wjc_adj)

    print()
    print("  " + "═" * 52)
    if stints:
        for s in stints:
            tag = {"reg": "Reg    ", "po": "Playoff", "wjc": "WJC-18 "}[s["kind"]]
            ppg = s["pts"] / s["gp"]
            print(f"  {tag}  {s['league_key'].upper():<14}"
                  f"{s['pts']:>5.1f}pts/{s['gp']:>4.1f}gp "
                  f"({ppg:.3f} PPG) × {s['factor']:.4f}")
        print(f"  {'─' * 52}")
        print(f"  Total NHLe pts    : {sum(s['pts']*s['factor'] for s in stints):.3f} "
              f"over {sum(s['gp'] for s in stints):.1f} GP")

    if wjc_adj != 0.0:
        direction = "boost" if wjc_adj >= 0 else "drag"
        print(f"  WJC-18 adj        : {wjc_adj:+.3f} log-odds {direction}")

    print(f"  NHL pts / 82      : {pts82:.1f}")
    print(f"  {'─' * 52}")
    print()
    print(f"  {render_bar(pn, ps)}")
    print(f"  ★★ Star (top 15% GAR) : {ps:.1%}")
    print(f"  ██ NHL player (200 GP+): {pn:.1%}  (includes star %)")
    print(f"  ░░ Bust (<200 GP)      : {1-pn:.1%}")
    print()
    print("  Star ≈ Hedman, Makar, Q.Hughes, Heiskanen, Dahlin")
    print("  Defenseman D-1 model — age ~17.")
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
    i = 1
    while True:
        raw = ask(f"{'Stint '+str(i) if i>1 else 'League'} (or done/leagues)",
                  default="done" if optional else None)
        if raw.lower() in ("done", ""):
            break
        if raw.lower() == "leagues":
            print_leagues(); continue
        try:
            lk, fac = parse_league(raw)
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
                       "league_key": lk, "factor": fac})
        print(f"    → {lk.upper()}: {pts}pts/{gp}gp ({pts/gp:.3f} PPG) × {fac:.4f}")
        if ask("Add another?", default="n").lower() not in ("y", "yes"):
            break
        i += 1
    return stints


def collect_wjc18():
    print(f"\n  WJC-18  (optional — team pts = d-man pts only)")
    print(f"  {'─' * 40}")
    raw = ask("Player points (or Enter to skip)", default="skip")
    if raw.lower() in ("skip", ""):
        return [], 0.0
    try:
        pp = float(raw)
        gp = float(ask("Games played"))
        tp = float(ask("Team d-man points (all games)"))
    except ValueError:
        return [], 0.0
    if gp <= 0 or tp <= 0:
        return [], 0.0
    fac = LEAGUE_FACTORS["wjc-18"]
    adj = wjc_log_odds_adj(pp, tp)
    direction = "boost" if adj >= 0 else "drag"
    print(f"    → WJC-18: {pp}pts/{gp}gp × {fac:.4f}, "
          f"d-man share {pp/tp:.1%} → {adj:+.3f} {direction}")
    return [{"kind": "wjc", "pts": pp, "gp": gp,
             "league_key": "wjc-18", "factor": fac}], adj


def main():
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == "--pts82":
        show_result(float(args[1]))
        return

    print("\n  NHLe Defenseman D-1 Model  (pre-draft year, age ~17)")
    print("  " + "=" * 52)
    print("  D-men pts/82 are lower than forwards — calibrated accordingly.")
    print("  Type 'leagues' at any prompt to list all options.")

    reg = collect_stints("REGULAR SEASON  (required — add all splits)", "reg")
    if not reg:
        print("\n  ! No data. Exiting."); sys.exit(1)

    po = collect_stints("PLAYOFFS  (optional)", "po", optional=True)
    wjc_stints, total_wjc_adj = collect_wjc18()

    all_stints = reg + po + wjc_stints
    pts82 = compute_pts82(all_stints)
    show_result(pts82, all_stints, total_wjc_adj)


if __name__ == "__main__":
    main()