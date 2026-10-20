"""
NHLe Forward D+5 Model
=======================
Predicts NHL outcomes using 5 seasons after the draft year (age ~23).

Outputs two probabilities on a single bar (Iacobelli-style):
  - P(NHL player) : reaches 200 GP
  - P(Star)       : peak GAR (avg best 3 seasons) >= 14.7, top 15% of forwards

If the player has ALREADY reached 200 GP or the GAR threshold, outputs 100%.

No WJC — D+5 players are 23, not eligible.

By D+5 the signal is extremely clear:
  - NHL production → confirmed NHLer, star depends entirely on rate
  - AHL with modest production → almost certain bust
  - Strong European pro → meaningful but uncertain NHLer probability
  - Still in junior/ECHL → certain bust

Trained directly on real D+5 data from 2008-2012 draft classes.
Star threshold: peak GAR >= ~14.7 (HR data through 2025).

Usage:
    python fwd-d5.py                  # interactive
    python fwd-d5.py --pts82 <value>  # pre-calculated pts/82
"""

import math
import sys

# ── Model coefficients (D+5, age ~23) ────────────────────────────────────────
# Trained directly on real D+5 data from 2008-2012 draft classes.
NHLER_INTERCEPT = -3.472162
NHLER_COEF      =  0.226039

STAR_INTERCEPT  = -7.221153
STAR_COEF       =  0.205414
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


def p_nhler(pts82):
    return 1.0 / (1.0 + math.exp(-(NHLER_INTERCEPT + NHLER_COEF * pts82)))


def p_star(pts82):
    raw = 1.0 / (1.0 + math.exp(-(STAR_INTERCEPT + STAR_COEF * pts82)))
    return min(raw, p_nhler(pts82))


def compute_pts82(stints):
    total_nhle = sum(s["pts"] * s["factor"] for s in stints)
    total_gp   = sum(s["gp"] for s in stints)
    return (total_nhle / total_gp) * 82 if total_gp > 0 else 0.0


def render_bar(p_nhl, p_str, width=40):
    n_star  = round(p_str * width)
    n_nhler = max(round(p_nhl * width), n_star)
    n_empty = width - n_nhler
    return "[" + "★" * n_star + "█" * (n_nhler - n_star) + "░" * n_empty + "]"


def show_result(pts82, stints=None, known_nhler=False, known_star=False):
    pn = 1.0 if (known_nhler or known_star) else p_nhler(pts82)
    ps = 1.0 if known_star else p_star(pts82)

    print()
    print("  " + "═" * 52)

    if known_star:
        print("  ⚡ Known outcome: player has already reached star threshold.")
    elif known_nhler:
        print("  ⚡ Known outcome: player has already reached 200 GP.")

    if stints:
        for s in stints:
            tag = {"reg": "Reg    ", "po": "Playoff"}[s["kind"]]
            ppg = s["pts"] / s["gp"]
            print(f"  {tag}  {s['league_key'].upper():<14}"
                  f"{s['pts']:>5.1f}pts/{s['gp']:>4.1f}gp "
                  f"({ppg:.3f} PPG) × {s['factor']:.4f}")
        total_nhle = sum(s["pts"] * s["factor"] for s in stints)
        total_gp   = sum(s["gp"] for s in stints)
        print(f"  {'─' * 52}")
        print(f"  Total NHLe pts    : {total_nhle:.3f} over {total_gp:.1f} GP")

    print(f"  NHL pts / 82      : {pts82:.1f}")
    print(f"  {'─' * 52}")
    print()
    print(f"  {render_bar(pn, ps)}")
    print(f"  ★★ Star (top 15% GAR) : {ps:.1%}")
    print(f"  ██ NHL player (200 GP+): {pn:.1%}  (includes star %)")
    print(f"  ░░ Bust (<200 GP)      : {1-pn:.1%}")
    print()
    print("  Star ≈ RNH, Kadri, Mark Stone, DeBrincat, Robert Thomas")
    print("  D+5 model — age ~23. No WJC (not eligible).")
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


def ask_known_outcomes():
    print(f"\n  KNOWN OUTCOMES  (if already achieved, outputs 100%)")
    print(f"  {'─' * 40}")
    nhler = ask("Has player already reached 200 NHL GP?", default="n").lower()
    known_nhler = nhler in ("y", "yes")
    star  = ask("Has player already hit star-level GAR (top 15%)?",
                default="n").lower()
    known_star  = star in ("y", "yes")
    return known_nhler, known_star


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

    print("\n  NHLe Forward D+5 Model  (5 seasons after draft, age ~23)")
    print("  " + "=" * 52)
    print("  No WJC — players are 23 and not eligible.")
    print("  Type 'leagues' at any prompt to list all options.")

    known_nhler, known_star = ask_known_outcomes()

    reg = collect_stints("REGULAR SEASON  (required — add all splits)", "reg")
    if not reg:
        print("\n  ! No regular season data. Exiting."); sys.exit(1)

    po = collect_stints("PLAYOFFS  (optional)", "po", optional=True)

    all_stints = reg + po
    pts82 = compute_pts82(all_stints)
    show_result(pts82, all_stints, known_nhler, known_star)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\n  Cancelled.")
        sys.exit(130)
