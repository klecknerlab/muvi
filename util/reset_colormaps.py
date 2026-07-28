#!/usr/bin/env python3
"""
reset_colormaps.py — restore muvi's ``colormaps.xml`` to the original (pre-change) set.

``add_colormaps.py`` saves a pristine snapshot, ``colormaps.original.xml``, the first time it changes
``colormaps.xml`` (and never overwrites it afterwards). This restores ``colormaps.xml`` from that
snapshot, undoing every colormap addition/replacement in one step.

Usage
-----
Run with the Python that has ``muvi`` installed (e.g. the muvi conda env)::

    python reset_colormaps.py                 # restore the installed muvi's colormaps.xml
    python reset_colormaps.py --xml PATH       # restore a specific colormaps.xml
    python reset_colormaps.py --keep-snapshot  # do not delete colormaps.original.xml after restoring

By default the snapshot is removed after a successful restore, so the NEXT ``add_colormaps.py`` run
captures a fresh pristine copy. Restart the muvi player afterwards to pick up the reverted set.

(If the snapshot is missing but ``colormaps.xml`` is a git checkout, ``git checkout -- colormaps.xml``
also restores the committed version.)
"""
import argparse
import importlib.util
import os
import shutil
import sys


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


def main(argv=None):
    ap = argparse.ArgumentParser(description="Restore muvi's colormaps.xml to the original set.")
    ap.add_argument("--xml", default=None, help="path to colormaps.xml (default: the installed muvi's)")
    ap.add_argument("--keep-snapshot", action="store_true", help="keep colormaps.original.xml after restoring")
    args = ap.parse_args(argv)

    xml_path = args.xml or find_colormaps_xml()
    snap = snapshot_path(xml_path)

    if not os.path.isfile(snap):
        print(
            f"no original snapshot at {snap} — nothing to reset.\n"
            "(add_colormaps.py saves it on its first change. In a git checkout you can instead run\n"
            f" `git checkout -- {os.path.basename(xml_path)}`.)",
            file=sys.stderr,
        )
        return 1

    shutil.copy2(snap, xml_path)
    if not args.keep_snapshot:
        os.remove(snap)
    print(f"restored {xml_path} from the original snapshot" + ("" if args.keep_snapshot else " (snapshot removed)"))
    print("Restart the muvi player to pick up the reverted colormaps.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
