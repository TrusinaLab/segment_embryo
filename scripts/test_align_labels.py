import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from image_io import STACK_Z_MAX, STACK_Z_MIN, align_label_volume_to_reference

labels = np.zeros((133, 512, 512), dtype=np.uint32)
labels[STACK_Z_MIN + 1, 10, 10] = 42

ref_shape = (STACK_Z_MAX - STACK_Z_MIN + 1, 512, 512)
aligned = align_label_volume_to_reference(labels, ref_shape)
assert aligned.shape == ref_shape
assert aligned[1, 10, 10] == 42

same = align_label_volume_to_reference(aligned, ref_shape)
assert same is aligned or np.shares_memory(same, aligned)

print("ok")
