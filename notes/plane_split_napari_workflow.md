# Plane split — Napari workflow (Step 1)

**Processing step 1** — slice embryo from trophectoderm. See [`processing_steps.md`](processing_steps.md) for the full numbered pipeline. Step 2 (view saved output): `run_view_segment.py`.

Split a 10-slice middle z-subset using hand-drawn divider lines. Two modes are available:

- **plane** — one flat dividing surface (best when lines are nearly coplanar)
- **interpolated surface** — a curved dividing surface when lines on different z-slices do not lie on one plane

Entry point: `run_plane_split.py` (uses the **micro-sam-napari** conda environment).

---

## Launch (outside Napari)

```bat
scripts\run_plane_split.bat
```

Or:

```bat
conda activate micro-sam-napari
python run_plane_split.py
```

On startup the viewer loads **10 contiguous z-slices from the center** of each channel in `22A_E1_Wnt3`.

---

## Dock widgets

| Widget | Action |
|--------|--------|
| **Reload images from disk** | Re-read TIFFs and refresh image layers |
| **Split mode** | Choose `plane` or `interpolated surface` |
| **Build split** | Fit the divider and update the `plane_split` labels layer |
| **Save masked TIFFs** | Color legend + dropdown to pick label 1 or 2 (matches napari colors), then save TIFFs |

---

## Layers

| Layer | Type | Role |
|-------|------|------|
| `channel 1`, `channel 2`, … | Image | Middle 10 z-slices |
| `divider_lines` | Shapes | Divider lines you draw (red) |
| `plane_split` | Labels | Split result after **Build split** |

**Label values**

| Value | Meaning |
|-------|---------|
| 1 | Positive side of the divider |
| 2 | Negative side of the divider |

After **Build split**, the save widget shows a **Label colors in napari** legend with colored swatches matching the `plane_split` layer. The **Label to keep** dropdown uses the same colors and names (e.g. `Label 1 — red-brown (#782506), positive side`).

Side assignment is geometric (not biological). Pick the entry that matches the colored region you want to keep.

---

## Split modes

Choose **Split mode** before clicking **Build split**.

### `plane`

Fits a **single 3D plane** through all line endpoints (least squares). Every voxel is classified by which side of that flat surface it lies on.

**Use when:** the boundary is roughly flat — e.g. an Ex/Em interface that looks like a straight cut through the stack.

**Requires:** at least **3 points** total (each line adds 2 endpoints → draw at least **2 lines**).

### `interpolated surface`

Treats each annotated z-slice as having its own **2D dividing line**. Line endpoints are **linearly interpolated in z** between key slices, so the divider can bend through the volume.

**Use when:** you draw three (or more) lines on different z-slices and they **cannot** lie on one plane — the boundary curves through z.

**Requires:** lines on at least **2 different z-slices** (three lines on three slices is ideal).

**Example:** lines on z = 2, 5, and 8 at different positions → plane mode would compromise; interpolated surface mode follows each line on its slice and connects them smoothly in between.

---

## Steps in Napari

1. Select the **`divider_lines`** shapes layer.

2. Choose the **line** tool in the toolbar (not polygon or rectangle).

3. Move through **z** (mouse wheel or slider) to a slice where the boundary is visible.

4. Click two points to draw a line along the boundary on that slice.

5. Repeat on **2–3 other z-slices**, tracking the same biological boundary. Spread slices apart (top, middle, bottom of the 10-slice window).

6. In the dock widget, choose **Split mode**:
   - **plane** — boundary is approximately flat
   - **interpolated surface** — lines bend and do not share one plane

7. Click **Build split**.

8. Review the **`plane_split`** labels layer (toggle opacity in the layer list).

9. To refine: edit lines, change split mode if needed, click **Build split** again. No restart required.

10. Optional: **Reload images from disk** if TIFFs on disk have changed.

11. Check **Label colors in napari** (colored swatches above the dropdown), choose **Label to keep**, then click **Save masked TIFFs**:
    - Voxels with the chosen label are kept
    - Everything else is set to **0**
    - One TIFF per z-slice per channel is written to `data/test segment embryo/`
    - Preview layers `channel N (masked)` are added in Napari

---

## Drawing tips

- Draw **one line per z-slice** when possible. Multiple lines on the same slice are averaged.
- Lines should stay **in-plane** on each slice (endpoints at the same z). If a line spans z, a warning is printed and the mean z is used.
- Lines do not need the same length or angle; they should follow the **same boundary** as it moves through z.
- For curved boundaries, use **interpolated surface**. Plane mode averages away curvature.
- The split covers the **entire loaded volume**, including background. Saved TIFFs only retain label-1 voxels; background outside that region is zero.

---

## Saved output

| Location | Contents |
|----------|----------|
| `data/test segment embryo/` | Masked TIFFs, one file per z-slice per channel |

Filenames match the source pattern, e.g. `22A_E1_Wnt3_z50c1x0-512y0-512.tif`, using the **original absolute z indices** from the full stack (not 0–9 local indices).

---

## Console output

After **Build split**, the terminal prints a summary, for example:

```text
plane_split: fitted plane n=[...], d=...
  label 1 (positive side): ... voxels
  label 2 (negative side): ... voxels
```

or, for surface mode:

```text
plane_split: interpolated surface through z-slices [2, 5, 8]
  label 1 (positive side): ... voxels
  label 2 (negative side): ... voxels
```

---

## Common errors

| Message | Fix |
|---------|-----|
| Layer is empty | Draw at least one line on `divider_lines` |
| Need at least 3 points (plane mode) | Add another line |
| Need lines on at least 2 different z-slices (surface mode) | Draw lines on two or more distinct z-slices |
| Degenerate zero-length line | Redraw the line with two distinct endpoints |

---

## Related code

| File | Role |
|------|------|
| `run_plane_split.py` | Step 1 Napari entry point |
| `run_view_segment.py` | Step 2 — load saved segment TIFFs |
| `notes/processing_steps.md` | Numbered pipeline overview |
| `segmentation/plane_split.py` | Plane fit, surface interpolation, dock widgets |
| `image_io.py` | Loads middle 10 z-slices from `22A_E1_Wnt3` |
