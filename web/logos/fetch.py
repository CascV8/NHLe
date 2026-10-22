"""
Download the league logos shown under the NHLe line on the card.

  python web/logos/fetch.py        # needs Pillow (pip install pillow)

Writes web/logos/<league key>.png (or .svg), named by the model's league keys,
and web/build.py copies whatever is here into the site. A league without a file
keeps its text label on the card, so adding one is just dropping a file in.

Sources: the NHL's own asset server for the NHL shield, Wikimedia Commons for
the NCAA logo, and TheSportsDB's league badges (free test API key "123") for the
rest. The logos are trademarks of their leagues, shown to identify them.
"""

import io
import json
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
UA = {"User-Agent": "Mozilla/5.0 (NHLe league logos)"}
HEIGHT = 96   # px; drawn at about 28px, so this covers 3x screens

# Logos are drawn transparent on the card; dark single-colour ones are inverted on the dark theme (logos.json).
SVG = {
    "nhl": "https://assets.nhle.com/logos/nhl/svg/NHL_light.svg",
    "ncaa": "https://upload.wikimedia.org/wikipedia/commons/d/dd/NCAA_logo.svg",
}
# A white wordmark would vanish on the light theme: recolour it to the site's ink (and invert it on the dark one).
RECOLOR = {"shl": (15, 26, 36)}
TSDB = "https://r2.thesportsdb.com/images/media/league/badge/"
PNG = {   # TheSportsDB league ids in comments
    "ahl": "bm25ej1582197507.png",          # 4738 American AHL
    "ohl": "y4z5ks1644535179.png",          # 5159 Canadian OHL
    "whl": "94w3kx1644535361.png",          # 5160 Canadian WHL
    "qmjhl": "8jqk3a1786013896.png",        # 5161 Canadian QMJHL
    "shl": "3q8qpc1787927189.png",          # 4419 Swedish Hockey League
    "allsvenskan": "xcb4bz1644535507.png",  # 5162 Swedish Hockey Allsvenskan
    "liiga": "thj2p31780638296.png",        # 4931 Finnish Liiga
    "mestis": "kpsk5m1615592034.png",       # 4935 Finnish Mestis
    "khl": "tfmf8l1615492008.png",          # 4920 Russian KHL
    "vhl": "tyad1s1630125596.png",          # 4919 Russian VHL
    "nl": "bibbao1626129533.png",           # 4934 Swiss National League
    "sl": "s4dm1f1644535592.png",           # 5163 Swiss League
    "del": "ttrhds1615493883.png",          # 4925 German DEL
    "del2": "ph2dmw1615592487.png",         # 4936 German DEL 2
    "slovakia": "xv0o1g1770377468.png",     # 4932 Slovak Extraliga
    "slovakia2": "19y0ho1615496565.png",    # 4929 Slovak Hockey League
}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return r.read()


def is_dark_mono(path):
    """A dark single-colour mark (black or grey, little colour) that would vanish on the dark theme."""
    img = Image.open(path).convert("RGBA")
    px = [p for p in img.getdata() if p[3] > 128]
    if not px:
        return False
    sat = sum(max(p[:3]) - min(p[:3]) for p in px) / len(px) / 255
    lum = sum(0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2] for p in px) / len(px) / 255
    return sat < 0.12 and lum < 0.45


def write_manifest():
    """logos.json: each logo's file, and whether the card should invert it on the dark theme."""
    out = {}
    for p in sorted(HERE.glob("*.*")):
        if p.suffix == ".png":
            out[p.stem] = {"file": p.name, "invert_dark": is_dark_mono(p)}
        elif p.suffix == ".svg":
            out[p.stem] = {"file": p.name, "invert_dark": False}   # the NHL shield and NCAA disc read on both themes
    (HERE / "logos.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print("dark mono:", [k for k, v in out.items() if v["invert_dark"]])


def main():
    for key, url in SVG.items():
        (HERE / f"{key}.svg").write_bytes(get(url))
        print(f"{key}.svg")
    for key, name in PNG.items():
        img = Image.open(io.BytesIO(get(TSDB + name))).convert("RGBA")
        img = img.crop(img.getbbox() or (0, 0, *img.size))   # trim transparent margins
        img.thumbnail((HEIGHT * 3, HEIGHT), Image.LANCZOS)
        if key in RECOLOR:
            ink = Image.new("RGBA", img.size, (*RECOLOR[key], 255))
            ink.putalpha(img.getchannel("A"))
            img = ink
        img.save(HERE / f"{key}.png", optimize=True)
        print(f"{key}.png {img.size}")
        time.sleep(1)   # be polite to TheSportsDB
    write_manifest()


if __name__ == "__main__":
    write_manifest() if "--manifest" in sys.argv else main()
