# -*- coding: utf-8 -*-

import os
import shutil
import argparse
from PIL import Image
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir")
    parser.add_argument("save_dir")
    args = parser.parse_args()

    SOURCE_DIR = args.source_dir
    SAVE_DIR = args.save_dir

    if not os.path.exists(SAVE_DIR):
      os.mkdir(SAVE_DIR)

    eliminate(SOURCE_DIR, SAVE_DIR)

def eliminate(base_dir, save_dir):
    x_dirs = [n for n in os.listdir(base_dir)]
    for x_dir in x_dirs:
        imnames = [n for n in os.listdir(base_dir + '/' + x_dir)]
        for fn in imnames:
            fn_path = os.path.join(base_dir + '/' + x_dir, fn)
            im = np.array(Image.open(fn_path))
            if im.size < 270000:
                continue
            px_count = 0
            for x in range(300):
                for y in range(300):
                    if all([val < 220 for val in im[x][y]]):
                        px_count += 1
                if px_count >= 45000:
                    break
                if (x + 1) * 300 - px_count > 45000:
                    break
            if px_count >= 45000:
                shutil.copyfile(fn_path, save_dir + '/' + fn)

if __name__ == '__main__':
    main()
