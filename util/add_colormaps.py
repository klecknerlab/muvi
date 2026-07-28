#!/usr/bin/env python3
"""
add_colormaps.py — bake matplotlib colormaps into muvi's ``colormaps.xml``.

The muvi player's geometry / tube / volume colours come from a FIXED set of colormaps baked into
``muvi/view/shaders/colormaps.xml`` as 256-entry RGB tables (base64). They are NOT looked up from
matplotlib at run time, so a new colormap must be baked into that file. This script does exactly
that: it samples one or more matplotlib colormaps to 256 RGB ``uint8`` values and appends them as
``<ColorMap>`` entries, byte-for-byte compatible with muvi's own format (floor rounding, RGB
row-major, length 256). muvi loads every entry dynamically into one packed texture at run time, so
adding entries here is all that is needed — no shader regeneration.

Two conveniences:
  * The file is always rewritten with its entries sorted **alphabetically by name** (case
    insensitive), so the player's colormap dropdown stays ordered as colormaps are added.
  * A name that is not a known matplotlib colormap is reported and skipped — every valid name is
    still added.

The FIRST time it changes ``colormaps.xml`` it saves a pristine snapshot, ``colormaps.original.xml``
(never overwritten afterwards); ``reset_colormaps.py`` restores from it to undo every change.

Usage
-----
Run with the Python that has ``muvi`` + ``matplotlib`` installed (e.g. the muvi conda env)::

    # Add the default set — matplotlib's Perceptually Uniform Sequential + Sequential + Diverging:
    python add_colormaps.py

    # Add specific colormaps by matplotlib name:
    python add_colormaps.py cool hot turbo gist_earth

    # Overwrite entries that already exist (default: skip ones already present):
    python add_colormaps.py --replace magma inferno

    # Point at a specific colormaps.xml (default: the installed muvi's, found via importlib):
    python add_colormaps.py --xml /path/to/colormaps.xml

    # Skip saving the pristine snapshot (default: saves colormaps.original.xml on first change):
    python add_colormaps.py --no-snapshot

After running, restart the muvi player: the new names appear in the colormap dropdown and are valid
values for a geometry ``.vtp``'s MuviDisplay ``colormap`` field. To revert, run ``reset_colormaps.py``.

The named colormap groups are matplotlib's, per
https://matplotlib.org/stable/users/explain/colors/colormaps.html
"""
import argparse
import base64
import importlib.util
import os
import re
import shutil
import sys

import numpy as np

# matplotlib colormap category groups (the default set added by this script).
PERCEPTUALLY_UNIFORM_SEQUENTIAL = ["viridis", "plasma", "inferno", "magma", "cividis"]
SEQUENTIAL = [
    "Greys", "Purples", "Blues", "Greens", "Oranges", "Reds", "YlOrBr", "YlOrRd", "OrRd",
    "PuRd", "RdPu", "BuPu", "GnBu", "PuBu", "YlGnBu", "PuBuGn", "BuGn", "YlGn",
]
DIVERGING = [
    "PiYG", "PRGn", "BrBG", "PuOr", "RdGy", "RdBu", "RdYlBu", "RdYlGn", "Spectral",
    "coolwarm", "bwr", "seismic",
    "berlin", "managua", "vanimo",   # Crameri scientific diverging maps (matplotlib >= 3.10)
]
DEFAULT_COLORMAPS = PERCEPTUALLY_UNIFORM_SEQUENTIAL + SEQUENTIAL + DIVERGING


def find_colormaps_xml():
    """Locate the installed muvi's ``colormaps.xml`` without importing muvi (no GL needed)."""
    spec = importlib.util.find_spec("muvi")
    if spec is None or spec.origin is None:
        raise FileNotFoundError("could not locate the 'muvi' package; use --xml to point at colormaps.xml")
    path = os.path.join(os.path.dirname(spec.origin), "view", "shaders", "colormaps.xml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"colormaps.xml not found at {path}")
    return path


def snapshot_path(xml_path):
    """The pristine-original snapshot path for a given colormaps.xml (`…/colormaps.original.xml`)."""
    return os.path.splitext(xml_path)[0] + ".original.xml"


def get_cmap(name):
    """A matplotlib colormap by name; raises (KeyError/ValueError) if ``name`` is not known."""
    import matplotlib

    if hasattr(matplotlib, "colormaps"):
        return matplotlib.colormaps[name]            # matplotlib >= 3.5; KeyError if unknown
    import matplotlib.cm as cm

    return cm.get_cmap(name)                          # older matplotlib


def bake(name):
    """256-entry RGB ``uint8`` bytes for a matplotlib colormap.

    Uses floor rounding (``* 255`` then ``astype(uint8)``) to match muvi's own baked entries exactly
    (verified against the shipped ``viridis``). Returns 768 raw bytes, RGB row-major.
    """
    lut = get_cmap(name)(np.linspace(0.0, 1.0, 256))[:, :3]   # (256, 3) floats in [0, 1]
    return (lut * 255).astype(np.uint8).tobytes()


def entry_body(short_name, display_name, data):
    """One ``<ColorMap>…</ColorMap>`` block (no leading indent / trailing newline)."""
    b64 = base64.b64encode(data).decode("ascii")
    return (
        f'<ColorMap ShortName="{short_name}" Name="{display_name}" '
        f'Length="256" Format="RGB" Encoding="Base64">\n'
        f"    {b64}\n"
        f"  </ColorMap>"
    )


def add_colormaps(names, xml_path, replace=False, snapshot=True):
    """Bake ``names`` into ``xml_path``, keeping all entries sorted by name.

    Existing entries are preserved verbatim (their display Names and data), so only new/replaced
    colormaps are (re)baked. The file is rewritten sorted alphabetically by ShortName (case
    insensitive) whenever anything changes (an addition, a replacement, or just re-ordering). A name
    matplotlib does not know is reported and skipped. Returns (added, replaced, skipped, missing).
    """
    text = open(xml_path, encoding="utf-8").read()

    # ShortName -> the full <ColorMap>…</ColorMap> block, original text preserved.
    bodies = {short: block for block, short in
              re.findall(r'(<ColorMap ShortName="([^"]+)".*?</ColorMap>)', text, flags=re.DOTALL)}
    existing = set(bodies)

    added, replaced, skipped, missing = [], [], [], []
    for name in names:
        try:
            data = bake(name)
        except Exception:
            missing.append(name)
            continue
        if name in existing and not replace:
            skipped.append(name)
            continue
        (replaced if name in existing else added).append(name)
        bodies[name] = entry_body(name, name, data)   # new/replaced: display Name = ShortName

    ordered = sorted(bodies.items(), key=lambda kv: kv[0].lower())
    out = '<?xml version="1.0"?>\n<ColorMaps>\n' + "".join(f"  {b}\n" for _, b in ordered) + "</ColorMaps>\n"

    if out != text:                                   # changed: an add/replace, or a re-sort
        snap = snapshot_path(xml_path)
        if snapshot and not os.path.exists(snap):     # capture the original ONCE, for reset_colormaps.py
            shutil.copy2(xml_path, snap)
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(out)

    return added, replaced, skipped, missing


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Bake matplotlib colormaps into muvi's colormaps.xml (kept sorted by name).",
        epilog="With no NAMES, adds matplotlib's Perceptually-Uniform-Sequential + Sequential + Diverging groups.",
    )
    ap.add_argument("names", nargs="*", help="matplotlib colormap names to add (default: the three groups above)")
    ap.add_argument("--xml", default=None, help="path to colormaps.xml (default: the installed muvi's)")
    ap.add_argument("--replace", action="store_true", help="overwrite entries already present (default: skip them)")
    ap.add_argument("--no-snapshot", action="store_true", help="do not save the pristine colormaps.original.xml")
    args = ap.parse_args(argv)

    names = args.names or DEFAULT_COLORMAPS
    xml_path = args.xml or find_colormaps_xml()

    added, replaced, skipped, missing = add_colormaps(
        names, xml_path, replace=args.replace, snapshot=not args.no_snapshot
    )

    print(f"colormaps.xml: {xml_path}")
    print(f"  added:    {', '.join(added) or '(none)'}")
    if replaced:
        print(f"  replaced: {', '.join(replaced)}")
    if skipped:
        print(f"  skipped (already present): {', '.join(skipped)}")
    if missing:
        print(f"  MISSING from matplotlib (not added): {', '.join(missing)}", file=sys.stderr)
    if added or replaced:
        print("Restart the muvi player to pick up the new colormaps (reset_colormaps.py reverts).")


if __name__ == "__main__":
    main()
