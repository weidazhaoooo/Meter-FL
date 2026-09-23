# Visual-domain attribute annotation guide (v2, full 1382-image pool)

Annotate ONLY what is visible in the image. NEVER estimate or mention the
meter reading/value. One JSON object per image, one per line (JSONL).

Output fields and closed vocabulary (exactly these values):

- file: image filename (basename with extension)
- instrument_type: ammeter | pressure_gauge | sphygmomanometer | tachometer | voltmeter | other
- viewpoint: front | mild_tilt | strong_tilt
- crop_level: closeup | full_dial | partial_dial | cropped_edge
- background_environment: clean_white | plain_indoor | real_scene | industrial_scene | cluttered | unknown
- image_quality: high_quality | low_resolution | blurry | dark
- visual_artifacts: array, subset of [glare, reflection, shadow, occlusion, motion_blur, none]
  (use ["none"] when nothing applies; never mix "none" with others)
- acquisition_style: handheld_photo | lab_photo | industrial_photo | product_photo | web_image | unknown
- short_reason: one factual sentence about visible evidence (scene, mounting, lighting) — no reading values
- confidence: 0.0-1.0 overall confidence in the attribute assignment

Field notes:
- instrument_type is judged from the dial face (units A/V/bar/MPa/psi/rpm,
  cuff/bulb for sphygmomanometer). Pressure gauges dominate this pool.
- acquisition_style: industrial_photo = fixed/inspection capture at an
  industrial site; lab_photo = experiment bench with instruments/wiring;
  product_photo = staged catalog shot; handheld_photo = casual phone-style shot.
- image_quality picks the dominant impairment; high_quality only if none.
