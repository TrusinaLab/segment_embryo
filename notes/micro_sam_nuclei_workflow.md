# micro-SAM 3D nuclei — Napari workflow (Step 2)

**Processing step 2** — instance segmentation of nuclei on DAPI after plane split. See [`processing_steps.md`](processing_steps.md) for the full pipeline.

| | |
|---|---|
| **Script** | `run_micro_sam_nuclei.py` |
| **Launcher** | `scripts/run_micro_sam_nuclei.bat` |
| **Input** | `data/test segment embryo/` — **channel 1 (Wnt), channel 2 (DAPI)** from step 1 |
| **Output** | `data/test_cell_labels/` — save **`committed_objects`** only |
| **More detail** | [`SAM_napari_notes.md`](SAM_napari_notes.md), [`progress.md`](progress.md) |

---

## Launch

From Python (recommended if you use notebooks or IDE):

```python
from run_micro_sam_nuclei import run_nuclei_segmentation

run_nuclei_segmentation()
run_nuclei_segmentation(z_range="50-80")
run_nuclei_segmentation(bottom_z_third=True)
```

Or terminal / bat:

```bat
python run_micro_sam_nuclei.py
scripts\run_micro_sam_nuclei.bat
```

Requires step 1 masked TIFFs on disk. The viewer opens with **Annotator 3d** already attached (not a blank napari).

**Default behaviour:** opens Napari immediately (no embedding progress bar on launch). Set z range in the dock if needed, then click **Compute Embeddings** in Annotator 3d (Napari progress bar). If a matching zarr cache already exists, that step loads quickly.

Embeddings cache: `data/embeddings/segment_dapi_vit_b_lm.zarr` (or `..._z50_80.zarr` etc. per z range)

Optional flags:

```bat
conda run -n micro-sam-napari python run_micro_sam_nuclei.py --resume
conda run -n micro-sam-napari python run_micro_sam_nuclei.py --model vit_l_lm
conda run -n micro-sam-napari python run_micro_sam_nuclei.py --list-z
```

**Z subset:** limit which planes are loaded, embedded, and segmented. Indices are **absolute z numbers from TIFF names** (e.g. `..._z50c1...`), inclusive.

**In Napari (recommended):** dock **Z range (segmentation)** on the right — set **Z min** / **Z max**, click **Apply Z range**, then **Compute Embeddings** in Annotator 3d if that z range has no cache yet.

```bat
scripts\run_micro_sam_nuclei.bat
conda run -n micro-sam-napari python run_micro_sam_nuclei.py --list-z
```

Optional CLI for the **initial** load only (still adjustable in the dock):

| Flag | Initial load | Embedding cache example |
|------|----------------|-------------------------|
| *(none)* | Full segment stack | `segment_dapi_vit_b_lm.zarr` |
| `--z-range 50-80` | Absolute z 50–80 | `..._z50_80.zarr` |
| `--bottom-z-third` | Lowest-Z ⅓ | `..._bottom_z3.zarr` |

With `--resume`, existing labels are cropped to the active z planes when you apply a range. Save new labels only for the z range you used.

---

## Embeddings progress (CPU)

| Where | What you see |
|-------|----------------|
| **Napari** | Bottom status progress bar when you click **Compute Embeddings** — should advance slice-by-slice after our UI refresh patch; window may still feel frozen on CPU |

micro-SAM disables background threads in napari ≥ 0.5, so long runs block the UI. Confirm the embedding path in the dock matches `data/embeddings/...` for your z range.

---

## Steps in Napari

1. **Compute Embeddings** — loads a finished zarr cache if present, otherwise runs in Napari (progress bar). Model **`vit_b_lm`** recommended.  
   Path is auto-set to a **`.zarr` file** (not the parent `data/embeddings` folder). If an earlier run for that z range failed, a new name is used (`_v2`, `_v3`, …); delete old incomplete folders in Explorer when you like.

2. Select layer **`point_prompts`**. Place **green** points inside nuclei, **red** on background.

3. **Segment** the current z-slice; refine with more points if needed.

4. **Segment All Slices** (`Shift+S`) to propagate through Z.

5. **Commit** into **`committed_objects`** (see below — layer choice matters).

6. **File → Save Selected Layer(s)...** — select **`committed_objects`** only → `data/test_cell_labels/`.

### Commit after automatic segmentation (many nuclei)

If you ran **automatic segmentation** (labels on **`auto_segmentation`**):

1. In the **Commit** widget, set **layer** → **`auto_segmentation`** (not `current_object`).
2. Leave **commit_path** empty unless you want a Zarr archive.
3. Click **Commit [C]** → check **`committed_objects`**, then save that layer (step 6).

### Commit after interactive segmentation (one nucleus at a time)

1. **layer** → **`current_object`** (must be non-empty after Segment / Segment All Slices).
2. **Commit** → repeat with new point prompts for each nucleus.

Use a descriptive filename, e.g. `22A_E1_Wnt3_segment_dapi_labels.tif`, so runs do not overwrite each other.

---

## Layers that matter

| Layer | Role |
|-------|------|
| `image` | DAPI stack passed to SAM |
| `point_prompts` | Your clicks |
| `current_object` | Work in progress |
| **`committed_objects`** | **Export this** |
| `auto_segmentation` | Automatic mask generation — **Commit** from this layer when using AMG |

Do **not** export `current_object` or `point_prompts` for downstream scripts.

---

## Resume / QC

- Step 2 starts **fresh** (empty `committed_objects`). Use `--resume` only to continue editing saved labels.
- The viewer shows **channel 1 (Wnt) / channel 2 (DAPI)** from step 1 only (no extra **`image`** layer). In the Annotator dock, **Image Layer** defaults to **channel 2** (DAPI).
- Label **shape must match** the DAPI stack (same z-range as step 1 save).
- To inspect masked channels without SAM: `scripts/run_view_segment.bat` (view only).

### Re-segment fused nuclei (reference overlay)

```bat
scripts\run_micro_sam_resegment.bat
```

| Layer | Role |
|-------|------|
| `reference_labels_qc` | Frozen contours from your saved labels — use to spot fused IDs |
| `committed_objects` | Editable copy — SAM commit and **Save** here |

Same embedding cache as step 2. After fixes, save **`committed_objects`** only to `data/test_cell_labels/`.

**Solo-cell inspection** (separate tool, no SAM): `scripts/run_inspect_cell_labels.bat` — see `run_inspect_cell_labels.py`.

**Split fused cell** (separate tool): `scripts/run_split_fused_cell.bat` — default mode **z cut** for nuclei stacked along z (one line through z in xz/yz view); **xy lines per z** when the neck is visible in each slice. See `run_split_fused_cell.py`.

---

## Downstream

- **Step 4 — VE / EPI:** `scripts/run_epi_ve_classifier.bat` (uses `data/test_cell_labels/`).
- **Embryo cup prototype:** `run_embryo_cup_mask.py` (nuclei dilation — not a substitute for step 3 DAPI+Wnt cup).

---

## Related code

| File | Role |
|------|------|
| `run_micro_sam_nuclei.py` | Step 2 entry point |
| `run_micro_sam_resegment.py` | Re-segment with `reference_labels_qc` overlay |
| `run_inspect_cell_labels.py` | Inspect one label ID at a time (no micro-SAM) |
| `segmentation/micro_sam_nuclei.py` | Load DAPI, launch Annotator 3d |
| `scripts/run_micro_sam_3d.bat` | Blank Annotator 3d (no preloaded stack) |
| `run_view_segment.py` | Optional QC of step 1 TIFFs |
