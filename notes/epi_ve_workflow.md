# VE / EPI classification workflow

Classify segmented cells into **VE** (visceral endoderm, outer) and **EPI** (epiblast, inner) using **morphological features** and **distance from the embryo center of mass**.

---

## Prerequisites

| Item | Location |
|------|----------|
| Cell label volume | `data/test_cell_labels/` (micro-SAM `committed_objects` export) |
| Images (optional) | `data/test segment embryo/` (step 1 masked) or raw `22A_E1_Wnt3/` middle z |

Label stack **shape must match** the image stack (same z-subset as segmentation).

---

## Launch

### Manual VE labeling (click cells)

After step 2 cell labels are saved:

```bat
scripts\run_ve_epi_manual.bat
```

1. Open `run_ve_epi_manual.py` — **`cell_labels`** starts in **pick mode** (same as tool **5**).
2. Dock dropdown **On each pick** → **Mark picked cell as VE (red)** (default).
3. **Click** each VE nucleus on `cell_labels` — cell turns **red** on `ve_epi_manual`.
4. When done → **Assign EPI to all remaining cells** (blue), then **Save**.

| Layer | Role |
|-------|------|
| `cell_labels` | Pick here (tool 5) — click = pick that cell id |
| `ve_epi_manual` | **Red** = VE, **blue** = EPI |

If pick mode is off, press **5** with `cell_labels` selected. Dropdown **Off** = inspect picks only, no coloring.

### Feature-based classifier (RF or napari-feature-classifier)

```bat
scripts\run_epi_ve_classifier.bat
```

Or:

```bat
conda activate micro-sam-napari
python run_epi_ve_classifier.py
```

Plugin for interactive clicking (included in `environment.yml`):

```bat
conda activate micro-sam-napari
conda install -c conda-forge napari-feature-classifier
```

---

## Computed features (per cell)

| Feature | Meaning |
|---------|---------|
| `distance_from_embryo_com` | Euclidean distance from cell centroid to **embryo center of mass** (union of all cells, scaled coords) |
| `distance_from_surface` | EDT depth inside the cell union (larger = more interior) |
| `radial_alignment` | \|dot(major PCA axis, outward normal)\| — ~0 tangential (VE-like), ~1 radial (EPI-like) |
| `elongation`, `flatness`, `sphericity` | 3D PCA shape metrics |
| `volume`, `n_voxels` | Size |

Embryo COM is printed in the terminal when the viewer opens.

**Embryo surface mask (default):** per-cell dilation **r=6**, then hole fill (per-z + 3D `binary_fill_holes`) and **closing r=2**. Same mask segments the embryo cup from background — see `run_embryo_cup_mask.py`. Preview: `python scripts/plot_cell_union_mask.py` → `data/epi_ve/cell_union_mask_padded_r6.png`.

**Z anisotropy:** use dock widget **Recompute features** with `z_spacing` ≠ `xy_spacing` if voxels are not cubic.

---

## Train and classify

### Option A — napari-feature-classifier (recommended)

1. **Plugins → napari-feature-classifier → Initialize a Classifier**
2. Class names: `VE`, `EPI`
3. Click **≥10 example cells per class** on varied z-slices (outer flat cells vs inner radial cells)
4. **Run Classifier**
5. Export predictions from the plugin when satisfied

### Option B — Random Forest in this project

1. **Save features CSV** (dock widget) → `data/epi_ve/cell_features.csv`
2. Add column `manual_class`: `1` = VE, `2` = EPI for training rows only
3. **Train RF & predict** — select that CSV
4. Output: `data/epi_ve/cell_features_predicted.csv` + `ve_epi_predictions` labels layer

---

## Napari layers

| Layer | Role |
|-------|------|
| `channel 1` (Wnt), `channel 2` (DAPI), … | Fluorescence (segment or raw) |
| `cell_labels` | Instance IDs from micro-SAM |
| `ve_epi_predictions` | After RF: 1=VE, 2=EPI (optional) |

---

## Biological expectation

| Class | Position | Shape / orientation |
|-------|----------|---------------------|
| **VE** | Farther from embryo COM, near surface | Lower `radial_alignment`, flatter |
| **EPI** | Closer to embryo COM, interior | Higher `radial_alignment`, more elongated |

Ambiguous cells at the interface — add more training examples rather than forcing a single threshold.

---

## Related

- [`EPI_VE_classifier.md`](EPI_VE_classifier.md) — background and plugin notes
- [`progress.md`](progress.md) — pipeline checkpoint
