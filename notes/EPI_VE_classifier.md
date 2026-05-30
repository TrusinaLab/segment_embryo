# EPI / VE cell classifier

Classify segmented cells into **VE** (visceral endoderm, outer layer) and **EPI** (epiblast, inner layer) using cell geometry and a random forest trained from manual examples in Napari.

---

## Biological signal

| Feature | VE (outer) | EPI (inner) |
|--------|------------|-------------|
| Position | Periphery of embryo cup | Closer to center |
| Major axis vs surface | ~90° — **tangential** to surface | ~0° — **radial** toward interior |
| Shape | Flatter | More columnar / elongated |

This pattern is visible in cross-section: a thin tangential shell outside, radial “spokes” inside. Geometry-based classification should work well for most cells; expect ambiguity at the VE/EPI interface.

---

## Key features to compute (per cell)

For each label ID in the 3D segmentation:

1. **Centroid** — center of mass of voxels.
2. **Principal axes** — 3D PCA on voxel coordinates; **major axis** = eigenvector with largest eigenvalue.
3. **Shape metrics** — elongation, flatness, sphericity from eigenvalues λ₁ ≥ λ₂ ≥ λ₃.
4. **Distance from surface** — e.g. `scipy.ndimage.distance_transform_edt` on the union mask of all cells (or Em mask).
5. **Outward normal** — gradient of distance map at centroid (flip to point outward), or local plane from outer centroids on curved cups.
6. **Radial alignment** — main classifier input:

   ```python
   cos_angle = abs(np.dot(major_axis, outward_normal))  # 0 = tangential, 1 = radial
   ```

   Use `abs()` because PCA axis direction is arbitrary.

**Note:** scikit-image `orientation` from regionprops is relative to the **image x-axis**, not tangential vs radial on the embryo — custom features above are needed for VE/EPI.

**Z anisotropy:** if voxels are non-cubic, scale coordinates before PCA (e.g. multiply Z by `dz/dxy`).

---

## Recommended path

```text
micro-sam 3D labels
    → script: compute distance + radial alignment + shape features
    → (optional) napari-skimage-regionprops: intensity features from c1/c2
    → napari-feature-classifier: click VE/EPI examples → Run Classifier
    → export predictions → VE / EPI label volumes
```

### Step-by-step

1. **Segment nuclei/cells** with micro-sam (Annotator 3D); export **3D label volume** (`.tif` or `.npy`).
2. **Compute features** with a script: table with columns `label`, `roi_id`, plus `distance_from_surface`, `radial_alignment`, `elongation`, `flatness`, etc.
3. **Open in Napari:** label layer + attach `layer.features = features_df` (or load CSV via the feature-classifier plugin).
4. **Train interactively:** `Plugins → napari-feature-classifier → Initialize a Classifier` → name classes `VE`, `EPI` → click examples → **Run Classifier**.
5. **Iterate:** fix misclassified cells, add borderline examples, run again.
6. **Export:** predictions CSV or `.clf` model; build separate **VE** and **EPI** label volumes for analysis.

---

## Napari plugins

### Primary: `napari-feature-classifier`

Random forest on **per-object feature tables**. Best fit for VE vs EPI after custom geometry features are computed.

- Hub: https://napari-hub.org/plugins/napari-feature-classifier.html
- Needs: label image + feature table in `layer.features` (`label`, `roi_id`, feature columns).
- Workflow: Initialize → annotate with keys 1/2 (classes) → Run Classifier → Predictions layer.
- Aim for **10+ examples per class** before first run; annotate across **multiple z-slices** and curved vs flat regions.

### Feature extraction: `napari-skimage-regionprops`

`Tools → Measurement tables → Regionprops (nsr)` — area, eccentricity, major/minor axis, intensity stats. Complements but does not replace custom radial/tangential features.

### Alternative: APOC (`napari-accelerated-pixel-and-object-classification`)

`Tools → Segmentation post-processing → Object classification (APOC)` — RF from painted labels + standard/intensity features. Less flexible for custom 3D orientation features.

---

## Installation

In the existing conda env:

```bash
conda activate micro-sam-napari
pip install napari-feature-classifier napari-skimage-regionprops
```

(`environment.yml` does not include these yet — install with pip as above.)

---

## Feature table format

Required columns for `napari-feature-classifier`:

| Column | Meaning |
|--------|---------|
| `label` | Integer cell ID matching the label image |
| `roi_id` | Image identifier (e.g. `"embryo1"`) when using one stack |
| … | Numeric feature columns for training |

Load options:

- Programmatically: `label_layer.features = features_df`
- CSV: `Plugins → napari-feature-classifier → CSV Feature Loader`

---

## Script-only option (no plugin)

```python
from sklearn.ensemble import RandomForestClassifier

X = features_df[["distance_from_surface", "radial_alignment", "elongation"]]
y = features_df["manual_class"]  # e.g. 0=EPI, 1=VE

clf = RandomForestClassifier(n_estimators=200, class_weight="balanced")
clf.fit(X, y)
features_df["prediction"] = clf.predict(X)
```

Map `prediction` back to label IDs to write VE/EPI volumes.

---

## Post-processing and caveats

- **Spatial smoothing:** majority vote among neighboring cells if an isolated VE pixel sits inside EPI (and vice versa).
- **Layer constraint:** VE should form a contiguous outer shell — optional morphological cleanup.
- **Interface zone:** cells within ~1–2 cell diameters of the boundary may stay ambiguous; flag for manual review.
- **Segmentation QC:** merged/split nuclei distort PCA; review outliers first.
- **Simple thresholds:** rule `score = w1*distance + w2*cos_angle + w3*flatness` can work, but RF + a small manual training set usually generalizes better.

---

## Relation to manual segmentation plan

See `segmentation_plan.md`: Ex/Em can still use shape dividers; **step 3 (VE vs EPI)** can use this classifier instead of (or alongside) a single dividing line/spline. Manual paint corrections remain useful at the interface and for obvious segmentation errors.

---

## Next code step (optional)

Add `compute_cell_features.py` that:

- reads the label `.tif`,
- computes distance, radial alignment, and shape features,
- opens Napari with the table ready for `napari-feature-classifier`.
