# Progress notes — restart here

Last updated: 2026-05-30

Checkpoint for the mouse gastrula Napari / micro-SAM workflow. Use this file to resume after a break.

---

## Done so far

| Step | Status | Notes |
|------|--------|--------|
| **Plane split (step 1)** | Documented / may be done | Masked intensity TIFFs → `data/test segment embryo/`. See [`plane_split_napari_workflow.md`](plane_split_napari_workflow.md), [`processing_steps.md`](processing_steps.md). |
| **micro-SAM cell segmentation** | **Saved** | Nuclei/cells segmented in **Annotator 3d**; **`committed_objects`** exported to **`data/test_cell_labels/`**. |
| **Embeddings cache** | Optional | Set `embeddings_save_path` in Embedding Settings to avoid recomputing (Zarr). See [`SAM_napari_notes.md`](SAM_napari_notes.md). |

---

## Data on disk (two different outputs)

Do not confuse these folders — they hold different things:

| Folder | What it is | How it was made |
|--------|------------|-----------------|
| `data/test segment embryo/` | Masked **fluorescence** TIFFs (one file per z per channel) | Plane-split widget → **Save masked TIFFs** |
| `data/test_cell_labels/` | **Instance label** volume(s) — integer ID per cell, 0 = background | micro-SAM → commit cells → save **`committed_objects`** layer |

Raw inputs stay in `22A_E1_Wnt3/` (not in git). Local outputs, TIFFs, `.npy`, and Zarr (`.zattrs`, `data/`, etc.) are listed in `.gitignore`.

**Filename tip:** For new runs, include dataset + content + z-range, e.g. `22A_E1_Wnt3_nuclei_middle10z_labels.tif`, so saves do not overwrite each other.

---

## micro-SAM — save & reload checklist

### Layers that matter

- **`committed_objects`** — final segmentation (this is what was saved).
- **`current_object`** — work in progress; not the export.
- **`point_prompts`** — prompts only; not the export.

### Commit workflow (before save)

1. Segment with green/red points → segment slice.
2. Optional: **Segment All Slices** (`Shift+S`) through Z.
3. **Commit** each finished object into `committed_objects`.
4. Repeat until all cells of interest are committed.

### How it was saved

- Napari: **File → Save Selected Layer(s)...** on **`committed_objects`** only.
- Location: **`data/test_cell_labels/`**.

### Optional: full annotation archive

- **commit** widget → `commit_path` → Zarr (prompts + results for reproducibility).
- For analysis / VE–EPI features, the flat **label TIFF or `.npy`** in `data/test_cell_labels/` is what scripts need.

### Reload sanity check

1. Label shape matches the image stack (**ZYX**).
2. Background = 0; each cell = unique positive integer.
3. Same z-slices / subset as when segmenting (note global z indices if only middle 10 slices were loaded).

---

## Environment & performance (reference)

| Topic | Current setup |
|-------|----------------|
| Conda env | `micro-sam-napari` (CPU PyTorch on laptop) |
| Launch SAM | `scripts/run_micro_sam_3d.bat` or Plugins → micro-sam → Annotator 3d |
| Model | **`vit_b_lm`** recommended |
| Faster embeddings | Nvidia mini PC + GPU conda env; or precompute + copy Zarr cache |
| More RAM | Helps only if memory full / swapping; otherwise GPU + cache + smaller z-subset |

Details: [`SAM_napari_notes.md`](SAM_napari_notes.md).

---

## Pipeline map

```text
22A_E1_Wnt3/  (raw TIFFs)
    │
    ├─ Step 1: plane split  →  data/test segment embryo/     (masked images)
    │
    └─ micro-SAM (parallel)  →  data/test_cell_labels/       (cell label IDs)
            │
            └─ Next: EPI / VE classifier  →  see EPI_VE_classifier.md
```

---

## Next steps (pick up here)

1. **Verify labels** — Open `data/test_cell_labels/` in Napari with the same channel stack; confirm alignment.
2. **EPI / VE classification** — [`EPI_VE_classifier.md`](EPI_VE_classifier.md):
   - Script: per-label features (`distance_from_surface`, `radial_alignment`, shape metrics).
   - Napari: `napari-feature-classifier` on label layer + feature table.
   - Export VE / EPI label volumes.
3. **Step 3 (project script)** — Still TBD in [`processing_steps.md`](processing_steps.md); may combine embryo mask + cell labels.
4. **Optional repo wiring** — Constants / viewer script for `data/test_cell_labels/` if we want `run_view_cell_labels.py` (not implemented yet).

---

## Related docs

| File | Topic |
|------|--------|
| [`processing_steps.md`](processing_steps.md) | Numbered scripts (plane split, view segment) |
| [`plane_split_napari_workflow.md`](plane_split_napari_workflow.md) | Step 1 Napari UI |
| [`SAM_napari_notes.md`](SAM_napari_notes.md) | micro-SAM install, embeddings cache, CPU tips |
| [`EPI_VE_classifier.md`](EPI_VE_classifier.md) | VE vs EPI after cell labels |
| [`segmentation_plan.md`](segmentation_plan.md) | Longer-term shapes / Ex–Em–VE–EPI plan |
