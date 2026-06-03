import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation.cell_features import minimal_label_features_table
from segmentation.label_focus import LabelFocusState, mask_labels_to_single_id

labels = np.zeros((3, 8, 8), dtype=np.uint32)
labels[1, 2:5, 2:5] = 7
labels[1, 5:8, 5:8] = 12

solo = mask_labels_to_single_id(labels, 7)
assert solo.sum() == 7 * 9
assert solo[1, 3, 3] == 7
assert solo[1, 6, 6] == 0

state = LabelFocusState(full_by_layer={"ref": labels.copy()})
assert state.set_solo(12)
assert state.volume_for_display("ref")[1, 6, 6] == 12
assert state.volume_for_display("ref")[1, 3, 3] == 0
state.clear_solo()
assert state.volume_for_display("ref")[1, 3, 3] == 7
assert not state.set_solo(99)

feat = minimal_label_features_table(np.array([0, 5, 0, 42, 101], dtype=np.uint32))
assert list(feat["index"]) == [5, 42, 101]
assert feat.loc[feat["index"] == 101, "name"].iloc[0] == "cell 101"

print("ok")
