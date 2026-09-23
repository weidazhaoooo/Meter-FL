"""Build MeterFL_v2: attribute-grounded federated split of ALL 1,382 labeled
images, from the fresh full-pool VLM annotation (domain_labels_v2.jsonl).

Pipeline (paper: attributes -> ordered first-match rules -> client):
1) acquisition_domain per image: catalog | industrial | lab | handheld
   (first-match over acquisition_style + background_environment, same rules
   as the original Split-B).
2) client = instrument_type x domain; cells with < MERGE_MIN images merge
   into that instrument's handheld cell; single-cell instruments -> _all.
3) dHash near-duplicate groups (hamming <= HAM) computed over the full pool;
   members of a group that land in the same client are forced to train.
4) per-client train/test: shuffle(seed=0), n_test = max(1, int(0.2n)),
   bumped to 2 when n >= 7, drawn from non-forced images.
5) images + labelme JSONs copied (real copies) from MeterFL_v1 into
   MeterFL_v2/<client>/{train,test}/; masks/ rasterized from the copied
   JSONs (rasterize_masks.py — the MeterFL_v1 masks were stale: singular
   "tick" labels and non-polygon shape_types had been silently dropped).
"""
import json, random, shutil
from collections import Counter
from pathlib import Path

from PIL import Image

from rasterize_masks import rasterize_tree

SEED = 0
MERGE_MIN = 8
HAM = 5
HERE = Path(__file__).parent
SRC = HERE.parent / "MeterFL_v1"
DST = HERE.parent / "MeterFL_v2"


def domain_of(r):
    if r["acquisition_style"] in ("product_photo", "web_image") or \
       r["background_environment"] == "clean_white":
        return "catalog"
    if r["acquisition_style"] == "industrial_photo" or \
       r["background_environment"] == "industrial_scene":
        return "industrial"
    if r["acquisition_style"] == "lab_photo":
        return "lab"
    return "handheld"


def dhash(path, size=8):
    im = Image.open(path).convert("L").resize((size + 1, size), Image.BILINEAR)
    px = list(im.getdata())
    bits = 0
    for y in range(size):
        for x in range(size):
            bits = (bits << 1) | (px[y * (size + 1) + x] > px[y * (size + 1) + x + 1])
    return bits


def main():
    recs = [json.loads(l) for l in open(HERE / "domain_labels_v2.jsonl")]
    srcpath = {}
    for p in SRC.glob("*/*/*"):
        if p.suffix in (".jpg", ".png") and p.parent.parent.name not in ("masks", "metadata"):
            srcpath[p.name] = p
    assert len(srcpath) == len(recs) == 1382

    # --- clients ---
    cell = {r["file"]: (r["instrument_type"], domain_of(r)) for r in recs}
    sizes = Counter(cell.values())
    for f, (t, d) in cell.items():
        if sizes[(t, d)] < MERGE_MIN and d != "handheld":
            cell[f] = (t, "handheld")
    dom_count = {t: len({d for tt, d in set(cell.values()) if tt == t})
                 for t in {t for t, _ in cell.values()}}
    client = {f: (f"{t}_all" if dom_count[t] == 1 else f"{t}_{d}") for f, (t, d) in cell.items()}

    # --- near-duplicate groups (union-find over dHash hamming <= HAM) ---
    hashes = {f: dhash(srcpath[f]) for f in srcpath}
    files = sorted(hashes)
    parent = {f: f for f in files}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, a in enumerate(files):
        ha = hashes[a]
        for b in files[i + 1:]:
            if bin(ha ^ hashes[b]).count("1") <= HAM:
                parent[find(a)] = find(b)
    groups = {}
    for f in files:
        groups.setdefault(find(f), []).append(f)
    nd_groups = [g for g in groups.values() if len(g) > 1]

    forced = set()
    for g in nd_groups:
        by_client = Counter(client[f] for f in g)
        for f in g:
            if by_client[client[f]] >= 2:
                forced.add(f)

    # --- split & copy ---
    if DST.exists():
        shutil.rmtree(DST)
    (DST / "masks").mkdir(parents=True)
    membership = {}
    for f, c in client.items():
        membership.setdefault(c, []).append(f)
    split = {}
    for c in sorted(membership):
        names = sorted(membership[c])
        n = len(names)
        n_test = max(1, int(0.2 * n))
        if n >= 7:
            n_test = max(2, n_test)
        order = names[:]
        random.Random(SEED).shuffle(order)
        test = []
        for f in order:
            if len(test) >= n_test:
                break
            if f not in forced:
                test.append(f)
        test = set(test)
        split[c] = {"train": sorted(x for x in names if x not in test), "test": sorted(test)}
        for f in names:
            side = "test" if f in test else "train"
            d = DST / c / side
            d.mkdir(parents=True, exist_ok=True)
            sp = srcpath[f]
            shutil.copy2(sp, d / f)
            jp = sp.with_suffix(sp.suffix + ".json")
            if not jp.exists():
                jp = sp.with_suffix(".json")
            shutil.copy2(jp, d / jp.name)

    stats, skipped = rasterize_tree(DST)
    print(f"masks rasterized: {dict(stats)}  skipped shapes: {dict(skipped) or 'none'}")

    meta = DST / "metadata"
    meta.mkdir()
    with open(meta / "client_assignment.jsonl", "w") as fo:
        for r in sorted(recs, key=lambda r: r["file"]):
            r = dict(r)
            r["client"] = client[r["file"]]
            r["side"] = "test" if r["file"] in split[client[r["file"]]]["test"] else "train"
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump({"seed": SEED, "merge_min": MERGE_MIN, "dhash_hamming": HAM,
               "split": split}, open(meta / "train_test_split.json", "w"), indent=1)
    json.dump(sorted(nd_groups, key=len, reverse=True),
              open(meta / "near_duplicate_groups.json", "w"), indent=1)
    with open(meta / "client_summary.csv", "w") as fo:
        fo.write("client,n_train,n_test\n")
        for c in sorted(split):
            fo.write(f"{c},{len(split[c]['train'])},{len(split[c]['test'])}\n")

    print(f"near-dup groups: {len(nd_groups)} (largest {max(len(g) for g in nd_groups)}), forced-to-train imgs: {len(forced)}")
    for c in sorted(split):
        print(f"  {c:28s} train {len(split[c]['train']):4d}  test {len(split[c]['test']):3d}")


if __name__ == "__main__":
    main()
