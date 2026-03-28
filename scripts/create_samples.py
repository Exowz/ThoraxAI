"""Copy 5 random images from each class in the test set to samples/."""

import random
import shutil
from pathlib import Path

SRC = Path("data/chest_xray/test")
DST = Path("samples")
N = 5

for cls in ("NORMAL", "PNEUMONIA"):
    src_dir = SRC / cls
    dst_dir = DST / cls
    dst_dir.mkdir(parents=True, exist_ok=True)

    images = list(src_dir.glob("*.jpeg")) + list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png"))
    if not images:
        print(f"No images found in {src_dir}")
        continue

    chosen = random.sample(images, min(N, len(images)))
    for img in chosen:
        shutil.copy2(img, dst_dir / img.name)
        print(f"  {cls}/{img.name}")

print(f"\nDone — {DST}/ ready.")
