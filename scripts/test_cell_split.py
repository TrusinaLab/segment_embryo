import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation.cell_split import (
    CellSplitMode,
    apply_split_to_labels,
    build_split_field,
    detect_target_cell_from_divider,
    split_preview_volume,
    split_volume_by_z_cut,
)
from segmentation.plane_split import split_volume_by_interpolated_surface

labels = np.zeros((5, 20, 20), dtype=np.uint32)
labels[2, 4:16, 4:16] = 7

lines_by_z = {
    1: np.array([[6.0, 10.0], [14.0, 10.0]]),
    3: np.array([[6.0, 10.0], [14.0, 10.0]]),
}
split_field = split_volume_by_interpolated_surface(labels.shape, lines_by_z)
preview = split_preview_volume(labels, 7, split_field)
assert preview[2, 10, 8] == 1
assert preview[2, 10, 12] == 2

out, kept, new_id = apply_split_to_labels(labels, 7, split_field, keep_sub_label=1)
assert kept == 7
assert new_id != 7
assert new_id in np.unique(out)
assert int((out == 7).sum()) > 0
assert int((out == new_id).sum()) > 0

# z-cut: nuclei stacked along z at same y,x
labels_z = np.zeros((10, 20, 20), dtype=np.uint32)
labels_z[1:9, 8:12, 8:12] = 9
markers = np.array([[4.0, 10.0, 10.0], [6.0, 10.0, 10.0]])
z_field = split_volume_by_z_cut(labels_z.shape, markers)
assert z_field[3, 10, 10] == 1
assert z_field[7, 10, 10] == 2
z_preview = split_preview_volume(labels_z, 9, z_field)
assert z_preview[3, 10, 10] == 1
built = build_split_field(
    labels_z.shape, {}, CellSplitMode.Z_CUT, marker_points_zyx=markers
)
assert np.array_equal(built, z_field)

_, kept_z, new_z = apply_split_to_labels(labels_z, 9, z_field, keep_sub_label=1)
assert kept_z == 9 and new_z > 9


class _FakeShapes:
    def __init__(self, data: list, name: str = "cell_split_divider_lines"):
        self.data = data
        self.name = name


# plane / z-cut: line through cell 7 in xy at z=2
plane_shapes = _FakeShapes([np.array([[2.0, 10.0, 5.0], [2.0, 10.0, 15.0]])])
detected, counts = detect_target_cell_from_divider(
    labels, plane_shapes, CellSplitMode.PLANE
)
assert detected == 7 and counts[7] > 0

# z-cut: line along z through stacked cell 9
z_shapes = _FakeShapes([np.array([[3.0, 10.0, 10.0], [7.0, 10.0, 10.0]])])
detected_z, _ = detect_target_cell_from_divider(
    labels_z, z_shapes, CellSplitMode.Z_CUT
)
assert detected_z == 9

# surface: per-z lines through cell 7 (same z as the labeled slice)
surface_shapes = _FakeShapes(
    [
        np.array([[2.0, 10.0, 5.0], [2.0, 10.0, 15.0]]),
        np.array([[2.0, 10.0, 6.0], [2.0, 10.0, 14.0]]),
    ]
)
detected_s, _ = detect_target_cell_from_divider(
    labels, surface_shapes, CellSplitMode.SURFACE
)
assert detected_s == 7

print("ok")
