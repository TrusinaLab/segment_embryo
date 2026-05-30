# Processing steps — mouse embryo segmentation

Numbered pipeline for this project. Run scripts in order. All use the **micro-sam-napari** conda environment.

**Restart checkpoint (what’s done, data folders, next steps):** [`progress.md`](progress.md)

Full Napari instructions for step 1: [`plane_split_napari_workflow.md`](plane_split_napari_workflow.md)

---

## Step 1 — Slice embryo from trophectoderm

**Goal:** Separate the embryonic region from trophectoderm (and extraembryonic tissue) by drawing divider lines, fitting a plane or curved surface, and saving the kept side.

| | |
|---|---|
| **Script** | `run_plane_split.py` |
| **Launcher** | `scripts/run_plane_split.bat` |
| **Input** | Raw TIFFs in `22A_E1_Wnt3/` (middle 10 z-slices loaded) |
| **Output** | Masked TIFFs in `data/test segment embryo/` |
| **Napari workflow** | [`plane_split_napari_workflow.md`](plane_split_napari_workflow.md) |

```bat
scripts\run_plane_split.bat
```

1. Draw divider lines on 2–3 z-slices  
2. **Build split** (plane or interpolated surface)  
3. Choose **Label to keep** → **Save masked TIFFs**

---

## Step 2 — View saved segment

**Goal:** Open the masked TIFFs from step 1 in Napari for inspection or as the starting point for the next step.

| | |
|---|---|
| **Script** | `run_view_segment.py` |
| **Launcher** | `scripts/run_view_segment.bat` |
| **Input** | `data/test segment embryo/` |
| **Output** | (view only) |

```bat
scripts\run_view_segment.bat
```

Requires step 1 output on disk. Use **Reload from disk** if files change.

---

## Step 3 — *(not yet implemented)*

**Goal:** Next segmentation stage on the step-1 embryo segment (e.g. further regional splits or ROI refinement).

| | |
|---|---|
| **Script** | — |
| **Input** | `data/test segment embryo/` |
| **Output** | TBD |

---

## Cell labels (micro-SAM, parallel track)

**Goal:** Instance segmentation of nuclei/cells; saved outside the numbered scripts above.

| | |
|---|---|
| **Tool** | Plugins → micro-sam → Annotator 3d (or `scripts/run_micro_sam_3d.bat`) |
| **Input** | Raw or masked stack in Napari |
| **Output** | Label volume(s) in **`data/test_cell_labels/`** (`committed_objects` layer) |
| **Details** | [`progress.md`](progress.md), [`SAM_napari_notes.md`](SAM_napari_notes.md) |
| **Downstream** | [`EPI_VE_classifier.md`](EPI_VE_classifier.md) |

---

## Other scripts (outside this pipeline)

These are **not** numbered steps. They load raw data or explore alternate workflows from earlier development.

| Script | Purpose |
|--------|---------|
| `main.py` / `scripts/run_napari.bat` | Raw channel viewer (middle z-subset), no segmentation |
| `run_embryo_roi.py` | Alternate step 1: manual polygon ROI (older workflow) |
| `run_embryo_roi_auto.py` | Alternate step 1: automatic embryo ROI |
| `scripts/run_micro_sam_3d.bat` | micro-sam click-to-segment (separate tool) |

When in doubt, use the numbered steps above.

---

## Quick reference

```
Step 1   run_plane_split.py     22A_E1_Wnt3  →  data/test segment embryo/
Step 2   run_view_segment.py    data/test segment embryo/  →  Napari view
Step 3   (future)
```
