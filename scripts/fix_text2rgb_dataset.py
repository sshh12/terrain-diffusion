import argparse
import random
import tqdm
import json
import os

from PIL import Image


def main(dataset_path, max_items, rewrite_images):
    train_dir = os.path.join(dataset_path, "train", "images")

    items = []
    with open(os.path.join(train_dir, "metadata.jsonl"), "r") as metacsv:
        lines = list(metacsv)
        for line in tqdm.tqdm(lines):
            data = json.loads(line)
            img_fn = os.path.join(train_dir, data["file_name"])
            if not os.path.exists(img_fn):
                print(f"Removing {data['file_name']}")
            else:
                if rewrite_images:
                    img = Image.open(img_fn)
                    img.convert("RGB").save(img_fn)
                items.append(data)

    if len(items) > max_items:
        print(f"Reducing from {len(items)} to {max_items}")
        items = random.sample(items, max_items)

    file_names = set()
    for item in items:
        file_names.add(item["file_name"])
    for fn in os.listdir(train_dir):
        if fn.endswith(".png") and fn not in file_names:
            print(f"Removing unused {fn}")
            os.remove(os.path.join(train_dir, fn))

    with open(os.path.join(train_dir, "metadata.jsonl"), "w") as meta:
        for item in items:
            meta.write(f"{json.dumps(item)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--max_items", type=int, default=10_000 - 1)
    parser.add_argument("--rewrite_images", action="store_true", default=False)
    args = parser.parse_args()
    main(args.dataset_path, args.max_items, args.rewrite_images)
