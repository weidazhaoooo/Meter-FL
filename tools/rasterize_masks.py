#!/usr/bin/env python3
"""Rasterize MeterFL_v2 labelme annotations into simscheme masks.

Replaces the stale MeterFL_v1 mask copies (which were rasterized by
MeterFL_v1/make_masks.py with two silent-drop bugs: singular "tick" label
not in its class map, and non-"polygon" shape_types skipped — MeasureBench
ticks are mostly oriented_rectangle).

Output: masks/<image-stem>.png at native image size, values
0=background, 1=ticks, 2=pointer (pointer drawn last, wins overlaps).

Labels tick/ticks -> 1, pointer -> 2. Shape types polygon, quadrilateral
and oriented_rectangle are drawn from their vertices; rectangle (2 corner
points) is expanded to the axis-aligned 4-corner box.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CLS = {"tick": 1, "ticks": 1, "pointer": 2}
POLY_TYPES = {"polygon", "quadrilateral", "oriented_rectangle"}


def shape_points(s):
    st = s.get("shape_type", "polygon")
    pts = [tuple(p) for p in s["points"]]
    if st in POLY_TYPES and len(pts) >= 3:
        return pts
    if st == "rectangle" and len(pts) == 2:
        (x0, y0), (x1, y1) = pts
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return None


def rasterize_tree(root: Path, strict=True):
    """(Re)build root/masks from all <client>/<side>/*.json. Returns stats.

    strict (default): any unknown label or unsupported shape_type/point-count
    aborts with a ValueError naming the offending file, instead of being
    silently skipped (the failure mode that produced the stale v1 masks).
    """
    out = root / "masks"
    out.mkdir(exist_ok=True)
    stats = Counter()
    skipped_shapes = Counter()
    for ann in sorted(root.glob("*/*/*.json")):
        d = json.load(open(ann))
        w, h = d.get("imageWidth"), d.get("imageHeight")
        if not w or not h:
            img = next(p for p in (ann.with_suffix(".png"), ann.with_suffix(".jpg")) if p.exists())
            w, h = Image.open(img).size
        m = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(m)
        for s in sorted(d["shapes"], key=lambda s: CLS.get(s["label"], 0)):  # ticks first
            c = CLS.get(s["label"])
            if c is None:
                if strict:
                    raise ValueError(f"{ann}: unknown label {s['label']!r}")
                skipped_shapes[f"label:{s['label']}"] += 1
                continue
            pts = shape_points(s)
            if pts is None:
                if strict:
                    raise ValueError(f"{ann}: unsupported shape {s.get('shape_type')!r} "
                                     f"with {len(s['points'])} points (label {s['label']!r})")
                skipped_shapes[f"type:{s.get('shape_type')}x{len(s['points'])}pts"] += 1
                continue
            draw.polygon(pts, fill=c)
        arr = np.asarray(m)
        old = out / (ann.stem + ".png")
        if old.exists():
            o = np.asarray(Image.open(old))
            o = o[..., 0] if o.ndim == 3 else o
            if o.shape == arr.shape and np.array_equal(o, arr):
                stats["unchanged"] += 1
            else:
                stats["changed"] += 1
                oc, nc = set(np.unique(o)), set(np.unique(arr))
                if nc - oc:
                    stats[f"gained:{sorted(nc - oc)}"] += 1
        else:
            stats["new"] += 1
        m.save(old)
        stats["written"] += 1
    return stats, skipped_shapes


def main():
    root = Path(__file__).resolve().parent.parent / "data"
    stats, skipped = rasterize_tree(root)
    print("masks:", dict(stats))
    print("skipped shapes:", dict(skipped) or "none")


if __name__ == "__main__":
    main()
