import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import tifffile

from image_io import save_channel_stack_as_tiffs, segment_tiff_filename
from segmentation.plane_split import (
    fit_plane_from_points,
    keep_label_choice_text,
    keep_mask_from_split_labels,
    label_colors_legend_html,
    mask_volume,
    parse_keep_label_choice,
    rgba_to_hex,
    split_volume_by_interpolated_surface,
    split_volume_by_plane,
)

pts = np.array([[0, 0, 0], [0, 10, 0], [0, 10, 10], [5, 0, 0]])
n, d = fit_plane_from_points(pts)
labels = split_volume_by_plane((10, 20, 20), n, d)
assert labels.shape == (10, 20, 20)

lines = {
    0: np.array([[5.0, 5.0], [5.0, 15.0]]),
    5: np.array([[8.0, 5.0], [8.0, 15.0]]),
    9: np.array([[11.0, 5.0], [11.0, 15.0]]),
}
surf = split_volume_by_interpolated_surface((10, 20, 20), lines)
assert set(np.unique(surf)) <= {0, 1, 2}

volume = np.ones((3, 4, 5), dtype=np.uint16) * 100
keep = keep_mask_from_split_labels(labels[:3, :4, :5])
masked = mask_volume(volume, keep)
assert masked.dtype == np.uint16
assert (masked[~keep] == 0).all()

name = segment_tiff_filename("22A_E1_Wnt3", 50, 1, (512, 512))
assert name == "22A_E1_Wnt3_z50c1x0-512y0-512.tif"

with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp)
    paths = save_channel_stack_as_tiffs(
        masked, channel_idx=1, z_indices=[48, 49, 50], output_dir=out
    )
    assert len(paths) == 3
    arr = tifffile.imread(paths[1])
    assert arr.shape == (4, 5)

assert parse_keep_label_choice("Label 1 — red-brown (#782506), positive side") == 1
assert parse_keep_label_choice("Label 2 — cyan (#5bd5f8), negative side") == 2
assert rgba_to_hex(np.array([0.47, 0.15, 0.02, 1.0])) == "#782605"

print("ok")
