# -*- coding: utf-8 -*-

import os
import re
import numpy as np
from scipy import stats
import cv2
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir")
    parser.add_argument("save_dir")
    args = parser.parse_args()

    SOURCE_DIR = args.source_dir
    SAVE_DIR = args.save_dir

    regx = re.compile(r'^\..+')

    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    convert(SOURCE_DIR, SAVE_DIR)


class Parameter:
    def __init__(self):
        self.n = 0
        self.average = 0
        self.sum = 0
        self.sum_s2 = 0
        self.sd = 0

    def __str__(self):
        return "n: {}, average: {}, sum: {}, sum_s2: {}, sd: {}".format(
            self.n, self.average, self.sum, self.sum_s2, self.sd
        )


def convert(base_dir, save_dir, show_progress=True):
    regx = re.compile(r'^\..+')
    imnames = [n for n in os.listdir(base_dir) if not regx.search(n)]
    if show_progress: print("Target directory: {}".format(base_dir))
    n_trashed = 0

    if show_progress: print("Checking size")
    imnames_original = imnames[:]
    for fn in imnames_original:
        fn_path = os.path.join(base_dir, fn)
        img = cv2.imread(fn_path)
        if img.shape != (300, 300, 3):
            n_trashed += 1
            imnames.remove(fn)

    if show_progress:
        print("{} removed".format(
            len(imnames_original) - len(imnames)
        ))
    del imnames_original
    n_img = len(imnames)

    parameters = [Parameter() for _ in range(3)]

    if show_progress: print("Calculating average")

    for fn in imnames:
        fn_path = os.path.join(base_dir, fn)
        img = cv2.imread(fn_path)
        img_bgr = [img[:, :, i] for i in range(3)]
        for i, p in enumerate(parameters):  #B, G, R
            p.n += img_bgr[i].size
            p.sum += img_bgr[i].sum()
    for p in parameters:
        p.average = p.sum / p.n
        if show_progress: print(p)

    if show_progress: print("Calculating standard deviation")

    for fn in imnames:
        # 要素数で割る前の分散を求める
        fn_path = os.path.join(base_dir, fn)
        img = cv2.imread(fn_path)
        img_bgr = [img[:, :, i] for i in range(3)]
        for i, p in enumerate(parameters):
            p.sum_s2 += ((img_bgr[i] - p.average) ** 2).sum()
    for p in parameters:
        p.sd = (p.sum_s2 / p.n) ** 0.5
        if show_progress: print(p)

    # z-value化して保存していく
    if show_progress: print("Saving files")

    for fn in imnames:
        fn_path = os.path.join(base_dir, fn)
        img = cv2.imread(fn_path)
        img_bgr = [img[:, :, i] for i in range(3)]
        img_bgr_zval = [None for _ in range(3)]
        img_bgr_normalized = [None for _ in range(3)]
        for i, p in enumerate(parameters):
            # 平均を引き標準偏差で割る
            img_bgr_zval[i] = (img_bgr[i] - p.average) / p.sd
            # 0 - 255 の間にノーマライズ
            img_bgr_normalized[i] = ((np.clip(img_bgr_zval[i], -2, 2) + 2) / 4) * 255

        z_img = np.dstack(img_bgr_normalized)
        cv2.imwrite(os.path.join(save_dir, 'z-{}'.format(fn)), z_img)

    if show_progress: print("Completed\n")


if __name__ == '__main__':
    main()
