# Manual segmentation plan — mouse embryo (Ex / Em / VE / EPI)

## Biological goal

Progressive manual annotation on 3D stacks, with easy correction at each step, ending in separate **3D label volumes** for **VE** and **EPI** (and optionally **Ex** and **Em**).

### Nested steps

1. **Embryo ROI** — isolate the ovoid-shaped mouse embryo.
2. **Ex vs Em** — split into extraembryonic (**Ex**) and embryonic (**Em**) “cups” (roughly half each).
3. **VE vs EPI** — within **Em**, separate visceral endoderm (**VE**) from embryonic ectoderm (**EPI**); EPI is closer to the center, VE overlays EPI peripherally.

---

### Steps 1–2 (embryo ROI + Ex/Em split): **Yes**

- **Step 1:** closed polygon around the embryo → raster mask `M_embryo`.
- **Step 2:** a dividing **line** (or open path) across the embryo → splits the plane into two half-planes.
- After **rotation** so **Em always points the same way** (e.g. “up” in image coordinates):
  - **upper** ∩ `M_embryo` → **Em**
  - **lower** ∩ `M_embryo` → **Ex**

This is **polygon ∩ half-plane** — straightforward to implement with NumPy / scikit-image masks.

### Step 3 (VE vs EPI inside Em): 


May need hand-fixes when VE/EPI stack heavily in projection; then use a **painted region or spline** inside Em, not only a single line.

### 3D

Logic extends to 3D, but requires a **z strategy** (annotate key slices, interpolate, or sparse 3D painting). SVG per slice alone does not produce a volume without an extra rasterization / interpolation step.

---

## What to save (SVG vs automation)

| Save | Purpose |
|------|---------|
| **Shapes layer data** (polygons, paths, lines) | Native Napari format; easy to edit and reload |
| **Raster labels** per step (`to_labels` or script) | Best for intersections (`&`, `\|`, `~`) |
| **SVG** | Optional for figures; parsing for pipelines is extra work |

**Workflow:** draw in **Shapes** → on “Build masks” / reload, **rasterize to masks** → store `embryo`, `Ex`, `Em`, `VE`, `EPI` (or one multi-label image with IDs).

---

## Napari layer design (easy to adjust)

Use **separate Shapes layers per step**, not one crowded layer:

| Layer name | Content |
|------------|---------|
| `embryo_outline` | Closed polygon around embryo |
| `ex_em_divider` | Line/path splitting Ex from Em (inside embryo only) |
| `ve_epi_divider` | Line/path splitting VE from EPI (inside Em only) |

- Fix step 2 → recompute only Ex/Em from layers 1 + 2.
- Fix step 3 → recompute VE/EPI from Em + layer 3.

**Optional:** a read-only **preview labels layer** showing Ex / Em / VE / EPI after each “Build masks” run.

---

## Intersection logic (programmatic)

```text
M_embryo  = rasterize(embryo polygon)

Em, Ex    = split_by_line(M_embryo, line_ex_em, upper=Em)   # after fixed rotation
M_em      = Em

VE, EPI   = split_by_line(M_em, line_ve_epi, ...)
centroid  = center_of_mass(M_em)   # or embryo interior
EPI       = component closer to centroid
VE        = other component inside M_em
```

**Rules:**

- Extend each dividing line across the relevant mask (full width of `M_embryo` or `M_em`).
- Lines must cross the mask so half-planes are unambiguous.

**Rotation:** define once per embryo or dataset (e.g. long axis vertical, **Em on +y**). Rotate masks (or images) before applying “upper = Em.”

---

## 3D VE / EPI layers

1. **2D per z:** run the same intersection on each slice where dividers exist.
2. **Sparse + interpolate:** draw steps 1–2 on key z; step 3 every N slices; interpolate between (e.g. `scipy.ndimage` or Napari plugins).
3. **Output:** 3D label volumes `VE(z,y,x)`, `EPI(z,y,x)` (optionally `Ex`, `Em`).

Use both image channels (`c1`, `c2`) to **validate** splits; geometry can still be driven by shapes.

---


## Recommended workflow (implementation order)

1. In Napari: three **Shapes** layers (`embryo_outline`, `ex_em_divider`, `ve_epi_divider`).
2. Add a **“Build masks”** control (e.g. dock widget next to Reload) that:
   - reads shape geometries,
   - computes `Ex`, `Em`, `VE`, `EPI`,
   - updates **Labels** preview layers.
3. Tweak shapes → **Build masks** again (no full app restart).
4. When satisfied, save label volumes (`.tif` or `.npy`) for analysis.

---

## Next code step (optional)

Extend `main.py` with a **“Build masks from shapes”** button that:

- reads the three shape layers from the active viewer,
- applies rotation + half-plane rules,
- writes preview label layers for Ex, Em, VE, and EPI.
