import argparse
import random
import json
import tqdm
import os

from PIL import Image


def main(caption_path, dataset_path):
    train_dir = os.path.join(dataset_path, "train", "images")
    os.makedirs(train_dir, exist_ok=True)

    id_ = 0
    with open(os.path.join(train_dir, "metadata.jsonl"), "w") as metacsv:
        with open(caption_path, "r") as f:
            lines = list(f)
            random.shuffle(lines)
            for line in tqdm.tqdm(lines):
                data = json.loads(line)
                img = Image.open(data["rgb_fn"])
                save_fn = f"{id_:06d}.png"
                img.save(os.path.join(train_dir, save_fn))

                meta = dict(file_name=save_fn, text=data["caption"])
                metacsv.write(f"{json.dumps(meta)}\n")
                id_ += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption_path", type=str)
    parser.add_argument("--dataset_path", type=str)
    args = parser.parse_args()
    main(args.caption_path, args.dataset_path)
