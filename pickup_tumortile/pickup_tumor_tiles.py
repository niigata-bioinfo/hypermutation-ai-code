import os
import re
import shutil
from sorter_inceptionV3 import Sorter
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir")
    parser.add_argument("save_dir")
    parser.add_argument('weights_path')
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()

    classes = ["tumor", "non-tumor"]
    source_dir = args.source_dir
    save_dir = args.save_dir
    weights_path = args.weights_path

    sorter = Sorter(
        classes=classes,
        finetuning_weights_path=weights_path,
        img_size=(300, 300),
        n_gpus=args.gpus
    )

    regx = re.compile(r"^\.")
    tilenames = [name for name in os.listdir(source_dir) if not regx.match(name)]

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    counter = 0
    n_tiles = len(tilenames)

    for i, tilename in enumerate(tilenames):
        tile_path = os.path.join(source_dir, tilename)

        evaluated_classname = sorter.detect(tile_path)
        if evaluated_classname == "tumor":
            out_path = os.path.join(save_dir, tilename)
            shutil.copyfile(tile_path, out_path)
            counter += 1
        print("{0}/{1}. {2} tiles are regarded as a tumor tile".format(i + 1, n_tiles, counter), "\r", end="")

    print("")
    print("{} tiles were tumor. They picked up and save at {}".format(counter, save_dir))

if __name__ == "__main__":
    main()
