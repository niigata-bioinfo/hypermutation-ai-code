from sorter_inceptionV3 import Sorter
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dir")
    parser.add_argument("validation_dir")
    parser.add_argument("save_weights_path")
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()
    classes = ["tumor", "non-tumor"]
    train_dir = args.train_dir
    validation_dir = args.validation_dir

    sorter = Sorter(
        classes=classes,
        train_dir=train_dir,
        validation_dir=validation_dir,
        save_weights_path=args.save_weights_path,
        img_size=(300, 300),
        n_gpus=args.gpus
    )

    # train
    sorter.train()

if __name__ == "__main__":
    main()
