"""
Simplified DensePose — stubs out detectron2 dependency.
Returns blank masks when real DensePose is not available.
"""

import numpy as np
from PIL import Image


class DensePose:
    def __init__(self, model_path="./checkpoints/densepose_", device="cuda"):
        self.device = device
        print("[DensePose] Using simplified mode (blank masks). Install detectron2 for full body parsing.")

    def __call__(self, image_or_path, resize=512):
        if isinstance(image_or_path, str):
            w, h = Image.open(image_or_path).size
        elif isinstance(image_or_path, Image.Image):
            w, h = image_or_path.size
        else:
            w, h = (768, 1024)
        return Image.fromarray(np.zeros((h, w), dtype=np.uint8))
