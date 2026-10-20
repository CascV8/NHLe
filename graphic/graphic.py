"""
NHLE Card Generator
═══════════════════
Run:   python graphic.py
Output: output/<OUTPUT_FILE> next to this script

Drop league logo PNGs into logos/leagues/ (e.g. WHL.png, NHL.png).
For undrafted players set draft_year, draft_pick, draft_team all to None.
For seasons not yet played, set nhle to None — they'll show as greyed out.
"""

# ══════════════════════════════════════════════════════════════════════════════
#  ✏️  EDIT HERE — change these values to generate a card for any player
# ══════════════════════════════════════════════════════════════════════════════

OUTPUT_FILE     = "Connor_Bedard_2024.png"   # ← filename of the saved card

PLAYER_NAME     = "Connor Bedard"
COUNTRY         = "Canada"
POSITION        = "F"               # F, D, or G

DRAFT_YEAR      = 2023
DRAFT_PICK      = 1
DRAFT_TEAM      = "Chicago Blackhawks"
GENERATION_YEAR = 2024

# Each season label maps to three values that directly drive the graphs:
#
#   nhle  → the player's actual NHLE score that season
#              - this is the dot/line position in the bottom line graph
#              - set to None if the season hasn't happened yet (shows greyed bar)
#
#   nhler → P(NHL player) in %, star % included (the model's "NHL player" line)
#              - top of the green (nhler) segment in the top bar chart
#
#   star  → P(star) in % (the model's "Star" line)
#              - top of the gold (star) segment in the top bar chart
#              - grey (bust) fills the rest up to 100%
#
# Example: nhler=65, star=40 means:
#   40% chance of being a star, 25% of being a non-star NHLer, 35% of a bust

NHLE_VALUES = {
    "D-1":  {"nhle": 40,   "nhler": 55,  "star": 30},
    "DY":   {"nhle": 72,   "nhler": 75,  "star": 45},
    "D+1":  {"nhle": 55,   "nhler": 65,  "star": 38},
    "D+2":  {"nhle": 60,   "nhler": 68,  "star": 42},
    "D+3":  {"nhle": 58,   "nhler": 66,  "star": 40},
    "D+4":  {"nhle": None, "nhler": 35,  "star": 20},  # ← None = not yet played
    "D+5":  {"nhle": None, "nhler": 72,  "star": 48},  # ← None = not yet played
}

# Which league the player was in (or is projected to be in) each season
# Leave as "" for future seasons
SEASONS = {
    "D-1": "WHL",
    "DY":  "WHL",
    "D+1": "NHL",
    "D+2": "NHL",
    "D+3": "NHL",
    "D+4": "",
    "D+5": "",
}

# ══════════════════════════════════════════════════════════════════════════════
#  Everything below is the engine 
# ══════════════════════════════════════════════════════════════════════════════

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from PIL import Image

_HERE      = os.path.dirname(os.path.abspath(__file__))
LEAGUE_DIR = os.path.join(_HERE, "logos", "leagues")
OUTPUT_DIR = os.path.join(_HERE, "output")

BG          = "#FFFFFF"
NHLER_GREEN = "#00D020"
STAR_GOLD   = "#FFC400"
BAR_BLUE    = "#5BB8F5"
BUST_GREY   = "#D9D9D9"
TEXT_DARK   = "#111111"
PLACEHOLDER = "#DDDDDD"

LEAGUE_DISPLAY = {
    "OHL":"OHL","WHL":"WHL","QMJHL":"QMJHL","USHL":"USHL","NCAA":"NCAA",
    "SHL":"SHL","Liiga":"Liiga","KHL":"KHL","AHL":"AHL","NHL":"NHL",
    "NL":"NL","DEL":"DEL","ECHL":"ECHL","Allsvenskan":"Allsv.","VHL":"VHL",
}


def _find_logo(directory, name):
    for ext in (".png", ".PNG", ".jpg", ".JPG", ".jpeg"):
        path = os.path.join(directory, name + ext)
        if os.path.exists(path):
            try:
                return np.array(Image.open(path).convert("RGBA"))
            except Exception:
                return None
    return None


def _place_img_data(ax, img, cx, cy, target_w_data, xlim, ax_bbox_w_px, zorder=6):
    data_range  = xlim[1] - xlim[0]
    px_per_data = ax_bbox_w_px / data_range
    target_px   = target_w_data * px_per_data
    zoom        = target_px / img.shape[1]
    im = OffsetImage(img, zoom=zoom, interpolation="lanczos")
    ab = AnnotationBbox(im, (cx, cy), frameon=False, pad=0,
                        box_alignment=(0.5, 0.5), zorder=zorder)
    ax.add_artist(ab)


def _grey_box(ax, x, y, w, h, label="", data=False):
    kw = {} if data else {"transform": ax.transAxes}
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004",
        facecolor=PLACEHOLDER, edgecolor="#BBBBBB", linewidth=0.6,
        clip_on=False, zorder=5, **kw,
    ))
    if label:
        tkw = {} if data else {"transform": ax.transAxes}
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=6.5, color="#777777", fontfamily="monospace",
                zorder=6, **tkw)


def _ordinal(n):
    s = "th" if 11 <= (n % 100) <= 13 else {1:"st",2:"nd",3:"rd"}.get(n%10,"th")
    return f"{n}{s}"


def generate_nhle_graphic(
    player_name, country, position,
    draft_year=None, draft_pick=None, draft_team=None,
    nhle_values=None, seasons=None,
    output_path=None, generation_year=None,
):
    if output_path is None:
        output_path = f"{player_name.replace(' ', '_')}_nhle.png"
    if generation_year is None:
        if draft_year is not None:
            generation_year = draft_year + max(1, len(nhle_values) - 1)
        else:
            generation_year = max(1, len(nhle_values))

    labels      = list(nhle_values.keys())
    n           = len(labels)
    xs          = np.arange(n, dtype=float)

    # nhle can be None for seasons not yet played — use NaN so they're skipped
    raw_nhle    = [nhle_values[k]["nhle"] for k in labels]
    has_nhle    = [v is not None for v in raw_nhle]   # True = season has data
    player_vals = np.array([v if v is not None else np.nan for v in raw_nhle], dtype=float)
    nhler_vals  = np.array([nhle_values[k]["nhler"] for k in labels], dtype=float)
    star_vals   = np.array([nhle_values[k]["star"]  for k in labels], dtype=float)

    CHART_XLIM = (0.0, float(n - 1))
    MARGIN     = 0.35
    PAD_XLIM   = (CHART_XLIM[0] - MARGIN, CHART_XLIM[1] + MARGIN)

    fig = plt.figure(figsize=(12, 8.6), dpi=150, facecolor=BG)
    gs  = fig.add_gridspec(
        4, 1,
        height_ratios=[1.10, 3.0, 0.28, 2.6],
        hspace=0.0,
        left=0.012, right=0.988, top=0.97, bottom=0.015,
    )
    ax_hdr   = fig.add_subplot(gs[0])
    ax_chart = fig.add_subplot(gs[1])
    ax_lbls  = fig.add_subplot(gs[2])
    ax_bars  = fig.add_subplot(gs[3])

    for ax in (ax_hdr, ax_chart, ax_lbls, ax_bars):
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    # ── HEADER ────────────────────────────────────────────────────────────
    ax_hdr.set_xlim(0, 1)
    ax_hdr.set_ylim(0, 1)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    hdr_bb   = ax_hdr.get_window_extent(renderer=renderer)
    ax_bb    = hdr_bb

    M_px = hdr_bb.width * 0.018
    M    = M_px / hdr_bb.width
    M_y  = M_px / hdr_bb.height

    SQ_H = 1.0 - 2 * M_y
    SQ_W = SQ_H * (hdr_bb.height / hdr_bb.width)
    SQ_Y = M_y

    # Year
    hdr_in        = hdr_bb.height / fig.dpi
    year_fontsize = hdr_in * (1.0 - 2 * M_y) * 72 * 0.88
    year_fontsize = max(20, min(120, year_fontsize))

    year_obj = ax_hdr.text(1.0 - M, 0.5, str(generation_year),
                transform=ax_hdr.transAxes,
                fontsize=year_fontsize, fontweight="bold", color=TEXT_DARK,
                va="center", ha="right", fontfamily="monospace")

    fig.canvas.draw()
    year_bb   = year_obj.get_window_extent(renderer=renderer)
    year_w    = year_bb.width / ax_bb.width
    YEAR_LEFT = 1.0 - M - year_w

    # Legend
    legend_items = [("star", STAR_GOLD), ("nhler", NHLER_GREEN), ("bust", BUST_GREY)]
    sw      = SQ_W * 0.40
    sh      = SQ_H / 5
    lbl_w   = 0.055
    lg_right = YEAR_LEFT - M
    lg_top   = 1.0 - M_y - sh / 2
    lg_bot   = M_y + sh / 2
    item_dy  = (lg_top - lg_bot) / (len(legend_items) - 1)

    for j, (lbl, color) in enumerate(legend_items):
        cy = lg_top - j * item_dy
        ax_hdr.add_patch(mpatches.Rectangle(
            (lg_right - sw, cy - sh / 2), sw, sh,
            transform=ax_hdr.transAxes, facecolor=color,
            edgecolor="none", clip_on=False, zorder=5,
        ))
        ax_hdr.text(lg_right - sw - M * 0.5, cy, lbl,
                    transform=ax_hdr.transAxes,
                    fontsize=8.5, color=TEXT_DARK,
                    va="center", ha="right", fontfamily="monospace")

    LEGEND_LEFT = lg_right - sw - lbl_w

    # Team logo
    TEAM_X = M
    NAME_X = TEAM_X + SQ_W + M
    NAME_Y = 1.0 - M_y
    DRAFT_Y = M_y

    # Player name (auto-scaled to ~50% width)
    fontsize = 34
    name_obj = ax_hdr.text(NAME_X, NAME_Y, player_name,
                transform=ax_hdr.transAxes,
                fontsize=fontsize, fontweight="bold", color=TEXT_DARK,
                va="top", ha="left", fontfamily="monospace")

    fig.canvas.draw()
    bb          = name_obj.get_window_extent(renderer=renderer)
    name_w_frac = bb.width / ax_bb.width
    target_w = 0.50
    if name_w_frac > 0:
        scale    = (target_w * 0.96) / name_w_frac
        fontsize = max(14, min(48, fontsize * scale))
        name_obj.set_fontsize(fontsize)

    fig.canvas.draw()
    bb          = name_obj.get_window_extent(renderer=renderer)
    name_w_frac = bb.width / ax_bb.width

    # Flag (right after name)
    FLAG_X      = NAME_X + name_w_frac + M
    NAME_MAX_X  = FLAG_X - M
    available_w = NAME_MAX_X - NAME_X

    if draft_year is None or draft_pick is None:
        _grey_box(ax_hdr, TEAM_X, SQ_Y, SQ_W, SQ_H, "no\nteam")
    else:
        _grey_box(ax_hdr, TEAM_X, SQ_Y, SQ_W, SQ_H, "team\nlogo")
    _grey_box(ax_hdr, FLAG_X, SQ_Y, SQ_W, SQ_H, "flag")

    # Draft line
    if draft_year is None or draft_pick is None:
        draft_line = f"Undrafted — {position.upper()}"
    else:
        team_part  = f" — {draft_team}" if draft_team else ""
        draft_line = f"#{draft_pick} — {draft_year}{team_part} — {position.upper()}"

    draft_fontsize = 13
    draft_obj = ax_hdr.text(NAME_X, DRAFT_Y, draft_line,
                transform=ax_hdr.transAxes,
                fontsize=draft_fontsize, fontweight="bold", color="#444444",
                va="bottom", ha="left", fontfamily="monospace")

    fig.canvas.draw()
    draft_bb     = draft_obj.get_window_extent(renderer=renderer)
    draft_w_frac = draft_bb.width / ax_bb.width
    if draft_w_frac > 0:
        scale          = (available_w * 0.96) / draft_w_frac
        draft_fontsize = max(10, min(17, draft_fontsize * scale))
        draft_obj.set_fontsize(draft_fontsize)

    ax_hdr.axhline(0.0, color="#CCCCCC", linewidth=1.0)

    # ── STACKED BAR CHART ────────────────────────────────────────────────
    ax = ax_chart
    ax.set_xlim(*PAD_XLIM)
    ax.set_ylim(0, 100)

    bar_w = 0.55
    lo = np.minimum(nhler_vals, star_vals)
    hi = np.maximum(nhler_vals, star_vals)

    star_h  = lo
    nhler_h = hi - lo
    bust_h  = 100.0 - hi

    # Draw each season's bar — faded/desaturated for unplayed seasons
    for i in range(n):
        alpha = 1.0 if has_nhle[i] else 0.30   # dim future seasons
        ax.bar(xs[i], star_h[i],  width=bar_w, bottom=0,     color=STAR_GOLD,   edgecolor="none", alpha=alpha, zorder=3)
        ax.bar(xs[i], nhler_h[i], width=bar_w, bottom=lo[i], color=NHLER_GREEN, edgecolor="none", alpha=alpha, zorder=3)
        ax.bar(xs[i], bust_h[i],  width=bar_w, bottom=hi[i], color=BUST_GREY,   edgecolor="none", alpha=alpha, zorder=3)

    MIN_H = 5
    for i in range(n):
        if not has_nhle[i]:
            continue   # skip % labels for future seasons
        if star_h[i]  >= MIN_H: ax.text(xs[i], star_h[i]/2,          f"{star_h[i]:.0f}%",  ha="center", va="center", fontsize=7.5, fontweight="bold", color="white",   fontfamily="monospace", zorder=5)
        if nhler_h[i] >= MIN_H: ax.text(xs[i], lo[i]+nhler_h[i]/2,   f"{nhler_h[i]:.0f}%", ha="center", va="center", fontsize=7.5, fontweight="bold", color="white",   fontfamily="monospace", zorder=5)
        if bust_h[i]  >= MIN_H: ax.text(xs[i], hi[i]+bust_h[i]/2,    f"{bust_h[i]:.0f}%",  ha="center", va="center", fontsize=7.5, fontweight="bold", color="#555555", fontfamily="monospace", zorder=5)

    for yv in np.arange(20, 100, 20):
        ax.axhline(yv, color="#DDDDDD", linewidth=0.7, zorder=1)
    ax.set_xticks([])

    # ── LABEL STRIP ───────────────────────────────────────────────────────
    ax_lbls.set_xlim(*PAD_XLIM)
    ax_lbls.set_ylim(0, 1)
    ax_lbls.axhline(0.98, color="#BBBBBB", linewidth=0.8)
    ax_lbls.axhline(0.02, color="#BBBBBB", linewidth=0.8)
    for i, lbl in enumerate(labels):
        ax_lbls.text(float(i), 0.50, lbl, ha="center", va="center",
                     fontsize=10, color="#333333",
                     fontfamily="monospace", fontweight="bold")

    # ── NHLE LINE GRAPH ───────────────────────────────────────────────────
    ab = ax_bars
    ab.set_xlim(*PAD_XLIM)
    played_vals = player_vals[~np.isnan(player_vals)]
    y_top = played_vals.max() * 1.85 if len(played_vals) > 0 and played_vals.max() > 0 else 100
    ab.set_ylim(0, y_top)

    ab.axhline(0, color="#CCCCCC", linewidth=1.0, zorder=1)

    # Only draw the line/dots for seasons that have data
    played_xs   = xs[~np.isnan(player_vals)]
    played_ys   = player_vals[~np.isnan(player_vals)]
    if len(played_xs) > 0:
        ab.plot(played_xs, played_ys, color=BAR_BLUE, linewidth=2.5,
                marker="o", markersize=7, markerfacecolor=BAR_BLUE,
                markeredgecolor="white", markeredgewidth=1.4, zorder=3)

    label_y     = played_ys.max() * 1.30 if len(played_ys) > 0 else y_top * 0.70
    logo_cy     = played_ys.max() * 1.62 if len(played_ys) > 0 else y_top * 0.88
    logo_w_data = 0.55
    logo_h_data = (played_ys.max() * 0.34) if len(played_ys) > 0 else y_top * 0.18
    ax_w_px     = ab.get_window_extent().width

    for i, lbl in enumerate(labels):
        league     = seasons.get(lbl, "")
        val        = player_vals[i]
        league_img = _find_logo(LEAGUE_DIR, league)

        if has_nhle[i]:
            # Season has data — show value label and logo normally
            ab.text(float(i), label_y, f"{val:.1f}",
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color=TEXT_DARK, fontfamily="monospace", zorder=4)
            if league_img is not None:
                _place_img_data(ab, league_img, cx=float(i), cy=logo_cy,
                                target_w_data=logo_w_data,
                                xlim=PAD_XLIM, ax_bbox_w_px=ax_w_px)
            elif league:
                disp = LEAGUE_DISPLAY.get(league, league[:6])
                _grey_box(ab, float(i)-logo_w_data/2, logo_cy-logo_h_data/2,
                          logo_w_data, logo_h_data, disp, data=True)
        else:
            # Season not yet played — show a dashed future indicator
            ab.text(float(i), y_top * 0.50, "—",
                    ha="center", va="center", fontsize=14,
                    color="#CCCCCC", fontfamily="monospace", zorder=4)

    ab.set_xticks([])
    ab.set_yticks([])

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"✅  Saved → {output_path}")
    return output_path


if __name__ == "__main__":
    # The ✅ in the save message crashes cp1252 when output is redirected
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    generate_nhle_graphic(
        player_name     = PLAYER_NAME,
        country         = COUNTRY,
        position        = POSITION,
        draft_year      = DRAFT_YEAR,
        draft_pick      = DRAFT_PICK,
        draft_team      = DRAFT_TEAM,
        generation_year = GENERATION_YEAR,
        nhle_values     = NHLE_VALUES,
        seasons         = SEASONS,
        output_path     = os.path.join(OUTPUT_DIR, OUTPUT_FILE),
    )