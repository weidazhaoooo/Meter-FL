#!/usr/bin/env python3
"""MeterFL_v2 full-dataset integrity validator (advisor P0.3) + correction
manifest (P0.2).

Checks, per image: image/JSON/mask identity, mask dimensions vs image,
allowed pixel values {0,1,2}, tick (class 1) and pointer (class 2) presence
cross-checked against the JSON annotation, foreground-area sanity,
sha256 provenance hashes, split/client assignment vs metadata, and
near-duplicate group train/test isolation.

Every failure gets an explicit status — no silent exclusion. Outputs:
  metadata/integrity_report.json     per-image records + summary
  metadata/correction_manifest.json  flagged samples with root-cause class
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent / "data"
MASKS = ROOT / "masks"
META = ROOT / "metadata"
TICK_LABELS = {"tick", "ticks"}
V1_POLY_ONLY = "polygon"  # what the buggy v1 rasterizer accepted


def sha256(p, buf=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def main():
    split = json.load(open(META / "train_test_split.json"))["split"]
    assign = {r["file"]: r for r in
              (json.loads(l) for l in open(META / "client_assignment.jsonl"))}
    nd_groups = json.load(open(META / "near_duplicate_groups.json"))

    records, flags = [], []
    imgs = sorted(p for p in ROOT.glob("*/*/*")
                  if p.suffix in (".jpg", ".png") and p.parent.parent.name not in ("masks", "metadata"))

    for img in imgs:
        client, side, stem = img.parent.parent.name, img.parent.name, img.stem
        rec = {"file": img.name, "client": client, "side": side, "status": "OK", "issues": []}

        jp = img.with_suffix(img.suffix + ".json")
        if not jp.exists():
            jp = img.with_suffix(".json")
        mp = MASKS / (stem + ".png")
        if not jp.exists():
            rec["issues"].append("MISSING_JSON")
        if not mp.exists():
            rec["issues"].append("MISSING_MASK")
        if rec["issues"]:
            rec["status"] = "FAIL"
            records.append(rec)
            continue

        with Image.open(img) as im:
            iw, ih = im.size
        m = np.asarray(Image.open(mp))
        if m.ndim == 3:
            m = m[..., 0]
        if m.shape != (ih, iw):
            rec["issues"].append(f"DIM_MISMATCH mask{m.shape} vs image{(ih, iw)}")
        bad_vals = set(np.unique(m).tolist()) - {0, 1, 2}
        if bad_vals:
            rec["issues"].append(f"BAD_PIXEL_VALUES {sorted(bad_vals)}")

        d = json.load(open(jp))
        shapes = d.get("shapes", [])
        json_tick = any(s["label"] in TICK_LABELS for s in shapes)
        json_ptr = any(s["label"] == "pointer" for s in shapes)
        mask_tick, mask_ptr = 1 in m, 2 in m

        # annotated-but-absent = pipeline error; unannotated = known gap
        if json_tick and not mask_tick:
            rec["issues"].append("TICK_ANNOTATED_BUT_ABSENT_IN_MASK")
        if json_ptr and not mask_ptr:
            rec["issues"].append("POINTER_ANNOTATED_BUT_ABSENT_IN_MASK")
        if not json_tick:
            rec["issues"].append("TICK_NOT_ANNOTATED")
        if not json_ptr:
            rec["issues"].append("POINTER_NOT_ANNOTATED")

        fg = float((m > 0).mean())
        rec["foreground_frac"] = round(fg, 5)
        if fg < 1e-4 or fg > 0.6:
            rec["issues"].append(f"FOREGROUND_AREA_SUSPECT {fg:.4f}")

        # v1-rasterizer damage classification (for the correction manifest)
        v1_flags = set()
        for s in shapes:
            if s["label"] == "tick":
                v1_flags.add("label_alias_tick")
            if s["label"] in TICK_LABELS | {"pointer"} and \
               (s.get("shape_type", "polygon") != V1_POLY_ONLY or len(s["points"]) < 3):
                v1_flags.add(f"v1_unsupported_shape:{s.get('shape_type')}")
        if v1_flags:
            rec["v1_rasterizer_flags"] = sorted(v1_flags)

        # split/client assignment consistency
        a = assign.get(img.name)
        if a is None:
            rec["issues"].append("NOT_IN_CLIENT_ASSIGNMENT")
        elif a["client"] != client or a["side"] != side:
            rec["issues"].append(f"ASSIGNMENT_MISMATCH meta={a['client']}/{a['side']}")
        if img.name not in split.get(client, {}).get(side, []):
            rec["issues"].append("NOT_IN_SPLIT_JSON")

        rec["hashes"] = {"image": sha256(img)[:16], "json": sha256(jp)[:16], "mask": sha256(mp)[:16]}
        hard = [i for i in rec["issues"]
                if not i.startswith(("TICK_NOT_ANNOTATED", "POINTER_NOT_ANNOTATED", "FOREGROUND_AREA_SUSPECT"))]
        rec["status"] = "FAIL" if hard else ("REVIEW" if rec["issues"] else "OK")
        records.append(rec)
        if rec["issues"] or "v1_rasterizer_flags" in rec:
            flags.append(rec)

    # near-duplicate isolation: same client must not have group members on both sides
    dup_violations = []
    side_of = {r["file"]: (r["client"], r["side"]) for r in records}
    for g in nd_groups:
        per_client = {}
        for f in g:
            if f in side_of:
                per_client.setdefault(side_of[f][0], set()).add(side_of[f][1])
        for c, sides in per_client.items():
            if len(sides) > 1:
                dup_violations.append({"client": c, "group": g})

    n = Counter(r["status"] for r in records)
    issue_counts = Counter(i.split()[0] for r in records for i in r["issues"])
    summary = {
        "total_images": len(records),
        "status_counts": dict(n),
        "issue_counts": dict(issue_counts),
        "duplicate_isolation_violations": len(dup_violations),
        "identity": {"images": len(imgs),
                     "jsons": sum(1 for r in records if "MISSING_JSON" not in r["issues"]),
                     "masks": sum(1 for r in records if "MISSING_MASK" not in r["issues"])},
    }
    json.dump({"summary": summary, "records": records, "dup_violations": dup_violations},
              open(META / "integrity_report.json", "w"), indent=1)

    manifest = [{
        "file": r["file"], "client": r["client"], "side": r["side"],
        "root_causes": sorted(set(
            (["missing_tick_annotation"] if "TICK_NOT_ANNOTATED" in r["issues"] else []) +
            (["missing_pointer_annotation"] if "POINTER_NOT_ANNOTATED" in r["issues"] else []) +
            (["area_outlier_review"] if any(i.startswith("FOREGROUND") for i in r["issues"]) else []) +
            [f for f in r.get("v1_rasterizer_flags", [])])),
        "action": ("needs_human_annotation"
                   if any(i.endswith("NOT_ANNOTATED") for i in r["issues"])
                   else "fixed_by_rasterizer_v2" if r.get("v1_rasterizer_flags")
                   else "needs_visual_QA"),
        "issues": r["issues"],
    } for r in flags]
    json.dump(manifest, open(META / "correction_manifest.json", "w"), indent=1)

    print(json.dumps(summary, indent=1))
    print(f"flagged samples in correction manifest: {len(manifest)}")
    for a, c in Counter(m["action"] for m in manifest).items():
        print(f"  {a}: {c}")
    if dup_violations:
        print("DUPLICATE ISOLATION VIOLATIONS:", dup_violations[:3])


if __name__ == "__main__":
    main()
