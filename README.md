# MeterFL: A Federated Benchmark for Analog Meter Reading

MeterFL is a mask-annotated benchmark for studying **federated learning (FL)
for visual analog meter reading**. It contains **1,382 dial-meter images**
with pixel-level annotations of scale **ticks** and **pointers**, organized into
**10 pseudo-clients** derived from visible image attributes, plus human
reading ground truth for the test split.

Because public meter images carry no natural site identities, clients are
constructed from visual evidence: a vision-language model (VLM) describes each
image with a closed attribute vocabulary, and deterministic rules map
instrument type and acquisition domain to a client. The VLM never assigns
client IDs directly.

## At a glance

| | |
|---|---|
| Images | 1,382 (76 from MeasureBench, 1,306 of unknown original source) |
| Annotation | LabelMe polygons: `ticks`, `pointer`; rasterized 3-class masks |
| Clients | 10 pseudo-clients (instrument type × acquisition domain) |
| Split | per-client train/test, seed 0, ~20% test; near-duplicate controlled |
| Reading GT | 274 test images (unit, scale range, tick values, reading) |

| Client | Train | Test | | Client | Train | Test |
|---|---:|---:|---|---|---:|---:|
| pressure_gauge_handheld | 544 | 135 | | ammeter_lab | 16 | 3 |
| pressure_gauge_industrial | 462 | 115 | | ammeter_catalog | 13 | 3 |
| pressure_gauge_lab | 28 | 6 | | sphygmomanometer_all | 7 | 2 |
| pressure_gauge_catalog | 26 | 6 | | voltmeter_all | 6 | 2 |
| tachometer_all † | 4 | 1 | | ammeter_handheld † | 2 | 1 |

† Below the minimum-size gate (train ≥ 5 and test ≥ 2). The standard protocol
uses the remaining **8 clients**: 1,102 training images, 272 segmentation test
images, and 269 test images whose ground-truth masks decode to a reading.

## Layout

```
data/
  <client>/{train,test}/<image>.{jpg,png}   # image
  <client>/{train,test}/<image>.json        # LabelMe annotation
  masks/<image-stem>.png                    # 0 background, 1 tick, 2 pointer
  readings/<image-stem>.json                # reading GT (test images)
  metadata/
    client_assignment.jsonl                 # VLM attributes + client + side
    client_summary.csv                      # per-client train/test counts
    train_test_split.json                   # split, seed and gate parameters
    near_duplicate_groups.json              # dHash groups (Hamming <= 5)
    sources.csv                             # per-image source
    integrity_report.json                   # per-image checks + sha256
    correction_manifest.json                # flagged samples and status
annotation/
  ANNOTATION_GUIDE.md                       # closed attribute vocabulary
  domain_labels_v2.jsonl                    # raw VLM attribute labels
tools/
  build_meterfl_v2.py                       # client/split construction (reference)
  rasterize_masks.py                        # LabelMe -> masks
  validate_integrity.py                     # integrity checks
  tick_reading.py                           # tick-count reading decoder
```

## Client construction

1. **Attributes.** A VLM labels seven attributes per image (instrument type,
   viewpoint, crop level, background, image quality, artifacts, acquisition
   style) from a closed vocabulary; see `annotation/ANNOTATION_GUIDE.md`.
2. **Acquisition domain** by ordered first-match rules: product/web photo or
   clean white background → `catalog`; industrial photo or industrial scene →
   `industrial`; lab photo → `lab`; otherwise `handheld`.
3. **Client** = instrument type × domain. Cells with fewer than 8 images merge
   into that instrument's handheld cell; instruments with a single cell form
   one `_all` client.
4. **Near-duplicates.** dHash groups (Hamming ≤ 5) are computed over the full
   pool; same-client group members are forced into train, so no test image has
   a near-duplicate twin in its client's training set.
5. **Split.** Per client, shuffle with seed 0 and hold out
   `max(1, int(0.2 n))` test images (at least 2 when n ≥ 7).

`tools/build_meterfl_v2.py` documents the exact procedure; it was run on the
pre-split image pool, which is not part of this release. The released split
files are the reference.

## Annotation formats

- **LabelMe JSON**: shapes labeled `ticks` (a few legacy shapes use `tick`) and
  `pointer`, mostly polygons with some oriented rectangles.
  `tools/rasterize_masks.py` handles all shape types and both tick labels.
- **Masks**: single-channel PNG, same size as the image, values
  {0 background, 1 tick, 2 pointer}.
- **Readings**: `unit`, `scale_min`, `scale_max`, `tick_values`, `reading_gt`,
  flags `seg_anomaly` / `annot_issue`, and free-text `notes`.

## Evaluation

- **Segmentation**: average pointer IoU over the 8 clients (mean of
  per-client IoUs) and client IoU STD (population standard deviation).
- **Reading**: decode predicted masks with `tools/tick_reading.py` into a
  normalized reading f ∈ [0, 1]; the full-scale error is
  %FS = |f_pred − f_gt| × 100, where f_gt decodes the ground-truth mask. Report
  median %FS and the share of images within 5 %FS.

## Known issues

`metadata/correction_manifest.json` lists 69 flagged samples: 64 were fixed by
the current rasterizer and 5 still lack a tick or pointer annotation
(`needs_human_annotation`). They are kept and flagged rather than dropped.

## Data sources

`data/metadata/sources.csv` lists the source of every image.

| Source | Images | Files |
|---|---:|---|
| [MeasureBench](https://huggingface.co/datasets/FlagEval/MeasureBench) (real split) | 76 | `ammeter_*`, `pressure_gauge_*`, `sphygmomanometer_*`, `tachometer_*`, `voltmeter_*` |
| Unknown original source | 1,306 | `voc_*` |

- **MeasureBench**: Lin et al., *Do Vision-Language Models Measure Up?
  Benchmarking Visual Measurement Reading with MeasureBench*, CVPR 2026
  ([arXiv:2510.26865](https://arxiv.org/abs/2510.26865)), released under
  CC BY-SA 4.0. Please also credit MeasureBench when you use these images.
- **Unknown source**: these images were gathered from publicly available
  meter-image collections, and their original source was not recorded. We do
  not claim ownership of them. If you hold rights to any image and want it
  credited or removed, please open an issue.

All tick/pointer annotations, masks, reading ground truth, attribute labels,
client assignments and splits were created for MeterFL.

## License

- **Data** (annotations, masks, readings, metadata, and images to the extent
  we can license them): [CC BY-SA 4.0](LICENSE), matching MeasureBench.
- **Code** in `tools/`: [MIT](LICENSE-CODE).

## Citation

A paper describing MeterFL is under review; citation details will be added
here.
