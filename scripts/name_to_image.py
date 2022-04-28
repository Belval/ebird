import cv2
import numpy as np
from PIL import Image

def main():
    codes = [
        "L1206350.npy",
        "L587433.npy",
        "L4319522.npy",
        "L736932.npy",
        "L127252.npy",
        "L253368.npy",
        "L1414144.npy",
        "L127260.npy",
        "L797480.npy",
        "L3942674.npy",
        "L127257.npy",
        "L577212.npy",
        "L816901.npy",
        "L2015010.npy",
        "L1915050.npy",
        "L1831075.npy",
        "L2861066.npy",
        "L4138659.npy",
        "L3109497.npy",
        "L2712544.npy",
        "L1096078.npy",
        "L453412.npy",
        "L638541.npy",
        "L551077.npy",
        "L2415913.npy",
        "L5771874.npy",
        "L624725.npy",
        "L1335789.npy",
        "L3354466.npy",
        "L338888.npy",
        "L624723.npy",
        "L132797.npy",
        "L5861442.npy",
        "L280690.npy",
    ]

    for code in codes:
        r = np.load(f"./ebird_data/r/{code}")
        g = np.load(f"./ebird_data/g/{code}")
        b = np.load(f"./ebird_data/b/{code}")

        Image.fromarray(np.moveaxis(cv2.normalize(np.concatenate([r, g, b]), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8), 0, -1) + 20).save(f"{code}.jpg")

if __name__ == "__main__":
    main()