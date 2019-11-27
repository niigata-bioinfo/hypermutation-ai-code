from sorter_inceptionV3 import Sorter
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir")
    parser.add_argument("weights_path")
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()
    source_dir = args.source_dir
    weigths_path = args.weights_path

    classes = ["hyper", "non-hyper"]

    base_path = Path(source_dir)
    validation_dirs = [
        str(p) for p in base_path.iterdir() if p.is_dir()
    ]

    sorter = Sorter(
        classes=classes,
        finetuning_weights_path=weigths_path,
        img_size=(300, 300),
        n_gpus=args.gpus
    )
    # evaluate
    sorter.evaluate(validation_dirs=validation_dirs)

if __name__ == "__main__":
    main()
