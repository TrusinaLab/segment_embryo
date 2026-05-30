# micro-SAM 3D nuclei — Napari workflow (Step 2)

**Processing step 2** — instance segmentation of nuclei on DAPI after plane split. See [`processing_steps.md`](processing_steps.md) for the full pipeline.

| | |
|---|---|
| **Script** | `run_micro_sam_nuclei.py` |
| **Launcher** | `scripts/run_micro_sam_nuclei.bat` |
| **Input** | `data/test segment embryo/` — **channel 1 (DAPI)** from step 1 |
| **Output** | `data/test_cell_labels/` — save **`committed_objects`** only |
| **More detail** | [`SAM_napari_notes.md`](SAM_napari_notes.md), [`progress.md`](progress.md) |

---

## Launch

```bat
scripts\run_micro_sam_nuclei.bat
```

Requires step 1 masked TIFFs on disk. The viewer opens with **Annotator 3d** already attached (not a blank napari).

**Default behaviour:** if embeddings are not cached yet, the script **precomputes them in the terminal first** (tqdm progress bar, one step per z-slice), then opens Napari.

Embeddings cache: `data/embeddings/segment_dapi_vit_b_lm.zarr`

Optional flags:

```bat
conda run -n micro-sam-napari python run_micro_sam_nuclei.py --no-resume
conda run -n micro-sam-napari python run_micro_sam_nuclei.py --model vit_l_lm
conda run -n micro-sam-napari python run_micro_sam_nuclei.py --skip-precompute
conda run -n micro-sam-napari python run_micro_sam_nuclei.py --precompute-only
scripts\precompute_embeddings.bat
```

---

## Embeddings progress (CPU)

| Where | What you see |
|-------|----------------|
| **Terminal (recommended)** | `Compute Image Embeddings 3D` tqdm bar when you run `run_micro_sam_nuclei.bat` (default) or `precompute_embeddings.bat` |
| **Napari** | Bottom status progress bar when you click **Compute Embeddings** — should advance slice-by-slice after our UI refresh patch; window may still feel frozen on CPU |

micro-SAM disables background threads in napari ≥ 0.5, so long runs block the UI. Prefer terminal precompute, then click **Compute Embeddings** in Napari only to **load** the zarr cache (confirm path matches `data/embeddings/...`).

---

## Steps in Napari

1. If embeddings were precomputed: **Compute Embeddings** loads the cache (fast). Otherwise click it once (watch terminal or Napari progress bar). Model **`vit_b_lm`** recommended.

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

- If `data/test_cell_labels/` already has a label file, it is loaded into **`committed_objects`** on startup (use `--no-resume` for a fresh run).
- Label **shape must match** the DAPI stack (same z-range as step 1 save).
- To inspect masked channels without SAM: `scripts/run_view_segment.bat` (view only).

---

## Downstream

- **Step 4 — VE / EPI:** `scripts/run_epi_ve_classifier.bat` (uses `data/test_cell_labels/`).
- **Embryo cup prototype:** `run_embryo_cup_mask.py` (nuclei dilation — not a substitute for step 3 DAPI+Wnt cup).

---

## Related code

| File | Role |
|------|------|
| `run_micro_sam_nuclei.py` | Step 2 entry point |
| `segmentation/micro_sam_nuclei.py` | Load DAPI, launch Annotator 3d |
| `scripts/run_micro_sam_3d.bat` | Blank Annotator 3d (no preloaded stack) |
| `run_view_segment.py` | Optional QC of step 1 TIFFs |
