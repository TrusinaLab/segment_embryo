# Processing steps — mouse embryo segmentation

Numbered pipeline for this project. Run scripts in order. All use the **micro-sam-napari** conda environment.

**Restart checkpoint (what’s done, data folders, next steps):** `[progress.md](progress.md)`

Full Napari instructions for step 1: `[plane_split_napari_workflow.md](plane_split_napari_workflow.md)`

---

## Step 1 — Slice embryo from trophectoderm

**Goal:** Separate the embryonic region from trophectoderm (and extraembryonic tissue) by drawing divider lines, fitting a plane or curved surface, and saving the kept side.


|                     |                                                                    |
| ------------------- | ------------------------------------------------------------------ |
| **Script**          | `run_plane_split.py`                                               |
| **Launcher**        | `scripts/run_plane_split.bat`                                      |
| **Input**           | Raw TIFFs in `22A_E1_Wnt3/` (full z-stack loaded)                  |
| **Output**          | Masked TIFFs in `data/test segment embryo/`                        |
| **Napari workflow** | `[plane_split_napari_workflow.md](plane_split_napari_workflow.md)` |


```bat
scripts\run_plane_split.bat
```

1. Draw divider lines on 2–3 z-slices
2. **Build split** (plane or interpolated surface)
3. Choose **Label to keep** → **Save masked TIFFs**

---

## Step 2 — Segment 3D nuclei (micro-SAM)

**Goal:** Instance-segment cell nuclei on **DAPI** from the plane-split stack using micro-SAM **Annotator 3d**; save label IDs per nucleus.


|                     |                                                                  |
| ------------------- | ---------------------------------------------------------------- |
| **Script**          | `run_micro_sam_nuclei.py`                                        |
| **Launcher**        | `scripts/run_micro_sam_nuclei.bat`                               |
| **Input**           | `data/test segment embryo/` (channel 1 = Wnt, channel 2 = DAPI)  |
| **Output**          | `data/test_cell_labels/` (`committed_objects` layer)             |
| **Napari workflow** | `[micro_sam_nuclei_workflow.md](micro_sam_nuclei_workflow.md)`   |


```bat
scripts\run_micro_sam_nuclei.bat
```

1. **Compute Embeddings** in Napari (Annotator 3d dock)
2. Point prompts → segment → **Segment All Slices** (`Shift+S`)
3. **Commit** objects → save **`committed_objects`** to `data/test_cell_labels/`

Requires step 1. Cache: `data/embeddings/`. See [`micro_sam_nuclei_workflow.md`](micro_sam_nuclei_workflow.md) for progress bars. Optional QC: `run_view_segment.py`.

---

## Step 3 — Embryo cup: segment embryo from background (DAPI + Wnt)

**Goal:** Outline the **3D embryo cup** and separate **embryo from background** using **3D dilation**, hole fill, and closing on segmented tissue — not on nuclei alone.

### Why DAPI + Wnt (not nuclei only)


| Input                | Role                                                   |
| -------------------- | ------------------------------------------------------ |
| **Wnt** (channel 1)  | Cytoplasmic / membrane signal — extends beyond nucleus |
| **DAPI** (channel 2) | Nuclear / cell positions                               |


**micro-SAM nuclei labels** (`data/test_cell_labels/`) are still used for **per-cell** work (VE/EPI), but **dilating nuclei only does not cover the Wnt signal**. Embryo–background masking should start from objects segmented on **both fluorescence channels**.

### Method (validated idea; labels TBD)

3D **per-object dilation + hole fill + closing** works well for a continuous embryo outline (see `data/epi_ve/cell_union_mask_padded_r6.png`). Apply the same recipe after segmenting **DAPI+Wnt objects**:

1. Load **DAPI** and **Wnt** from `22A_E1_Wnt3/` (same z-subset as other steps, e.g. middle 10 slices).
2. **Segment** and **identify objects** on those channels (micro-SAM or other — tissue/cell bodies, not nuclei-only).
3. Build a **combined label volume** (union of objects across channels, or one segmentation driven by both).
4. **Dilate each object in 3D** (default ball **r = 6** per object, then merge).
5. **Fill holes** (per z-slice + 3D `binary_fill_holes`) and **close** (ball r = 2) to bridge gaps.
6. Output: binary **embryo cup** mask → mask background, save TIFFs, feed VE/EPI surface geometry if desired.


|            |                                                                                               |
| ---------- | --------------------------------------------------------------------------------------------- |
| **Script** | *TBD* (reuse `embryo_cup_mask_from_cells()` in `segmentation/cell_features.py` on new labels) |
| **Input**  | DAPI + Wnt channels; segmented object labels (not `committed_objects` nuclei)                 |
| **Output** | `data/embryo_cup_mask/` (planned); masked channels → `data/embryo_cup_segment/`               |


### Prototype (interim — nuclei labels only)

Until DAPI+Wnt segmentations exist, a **nucleus-based** cup mask is available for testing:


|              |                                            |
| ------------ | ------------------------------------------ |
| **Script**   | `run_embryo_cup_mask.py`                   |
| **Launcher** | `scripts/run_embryo_cup_mask.bat`          |
| **Input**    | `data/test_cell_labels/` + channels        |
| **Output**   | `data/embryo_cup_mask/embryo_cup_mask.tif` |


```bat
scripts\run_embryo_cup_mask.bat
```

Default **pad radius = 6**. Useful for proof-of-concept; **replace with Step 3 pipeline above** for production embryo–background masks.

---

## Step 4 — VE / EPI cell classification

**Goal:** Assign each segmented cell as **VE** (visceral endoderm) or **EPI** (epiblast).

### Manual labeling (recommended after full segmentation)

|                     |                                                            |
| ------------------- | ---------------------------------------------------------- |
| **Script**          | `run_ve_epi_manual.py`                                     |
| **Launcher**        | `scripts/run_ve_epi_manual.bat`                            |
| **Input**           | `data/test segment embryo/` + `data/test_cell_labels/`     |
| **Output**          | `data/epi_ve/ve_epi_manual.csv`, `ve_epi_manual_labels.tif` |

```bat
scripts\run_ve_epi_manual.bat
```

1. Crosshair on a nucleus → **Mark cell at crosshair as VE**
2. When done → **Assign EPI to all remaining cells**
3. **Save manual labels (CSV + TIFF)**

### Feature-based classifier (optional)

|                     |                                                            |
| ------------------- | ---------------------------------------------------------- |
| **Script**          | `run_epi_ve_classifier.py`                                 |
| **Launcher**        | `scripts/run_epi_ve_classifier.bat`                        |
| **Output**          | `data/epi_ve/cell_features.csv` (and optional predictions) |
| **Napari workflow** | `[epi_ve_workflow.md](epi_ve_workflow.md)`                 |

```bat
scripts\run_epi_ve_classifier.bat
```

Train with **napari-feature-classifier** or CSV + **Train RF & predict**

Uses **nucleus labels** from step 2. Surface features can later use the **Step 3** embryo cup mask when available.

---

## Optional — View step 1 segment (no SAM)


|              |                                |
| ------------ | ------------------------------ |
| **Script**   | `run_view_segment.py`          |
| **Launcher** | `scripts/run_view_segment.bat` |
| **Input**    | `data/test segment embryo/`    |
| **Output**   | (view only)                    |


```bat
scripts\run_view_segment.bat
```

---

## Blank micro-SAM (no preloaded stack)


|       |                                                              |
| ----- | ------------------------------------------------------------ |
| **Tool** | `scripts/run_micro_sam_3d.bat` → empty Annotator 3d       |
| **Use**  | Raw `22A_E1_Wnt3/` or custom data loaded manually in Napari |


---

## Other scripts (outside this pipeline)

These are **not** numbered steps. They load raw data or explore alternate workflows from earlier development.


| Script                               | Purpose                                               |
| ------------------------------------ | ----------------------------------------------------- |
| `main.py` / `scripts/run_napari.bat` | Raw channel viewer (middle z-subset), no segmentation |
| `run_embryo_roi.py`                  | Alternate step 1: manual polygon ROI (older workflow) |
| `run_embryo_roi_auto.py`             | Alternate step 1: automatic embryo ROI                |
| `scripts/run_micro_sam_3d.bat`       | micro-sam click-to-segment (separate tool)            |


When in doubt, use the numbered steps above.

---

## Quick reference

```
Step 1   run_plane_split.py           22A_E1_Wnt3  →  data/test segment embryo/
Step 2   run_micro_sam_nuclei.py      segment DAPI  →  data/test_cell_labels/
         run_view_segment.py          (optional QC of step 1 TIFFs)
Step 3   [planned] DAPI+Wnt segment → 3D dilate  →  data/embryo_cup_mask/
         [prototype] run_embryo_cup_mask.py  (nuclei labels only)
Step 4   run_ve_epi_manual.py         manual VE clicks  →  data/epi_ve/ve_epi_manual.*
         run_epi_ve_classifier.py     (optional RF)     →  data/epi_ve/
```

