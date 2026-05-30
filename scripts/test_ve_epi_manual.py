import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation.ve_epi_manual import (
    CLASS_EPI,
    CLASS_VE,
    ManualClassState,
    load_assignments_from_csv,
)

labels = np.zeros((4, 10, 10), dtype=np.uint32)
labels[1, 2:5, 2:5] = 1
labels[2, 6:9, 6:9] = 2

state = ManualClassState(instance_labels=labels)
state.assignments[1] = CLASS_VE
state.assignments[2] = CLASS_EPI
state.rebuild_class_volume()

vol = state.class_volume()
assert vol[1, 3, 3] == CLASS_VE
assert vol[2, 7, 7] == CLASS_EPI

state.assignments.clear()
state.assignments[1] = CLASS_VE
remaining = [lid for lid in state.all_cell_ids() if lid not in state.assignments]
for lid in remaining:
    state.assignments[lid] = CLASS_EPI
state.rebuild_class_volume()
assert state.counts()["ve"] == 1
assert state.counts()["epi"] == 1

with tempfile.TemporaryDirectory() as tmp:
    csv_path = Path(tmp) / "manual.csv"
    state.to_dataframe().to_csv(csv_path, index=False)
    loaded = load_assignments_from_csv(csv_path)
    assert loaded == {1: CLASS_VE, 2: CLASS_EPI}

print("ok")
