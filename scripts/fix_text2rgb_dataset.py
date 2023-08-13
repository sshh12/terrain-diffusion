import argparse
import tqdm
import json
import os


def main(dataset_path):
    train_dir = os.path.join(dataset_path, "train", "images")

    items = []
    with open(os.path.join(train_dir, "metadata.jsonl"), "r") as metacsv:
        lines = list(metacsv)
        for line in tqdm.tqdm(lines):
            data = json.loads(line)
            if not os.path.exists(os.path.join(train_dir, data["file_name"])):
                print(f"Removing {data['file_name']}")
            else:
                items.append(data)

    with open(os.path.join(train_dir, "metadata.jsonl"), "w") as meta:
        for item in tqdm.tqdm(items):
            meta.write(f"{json.dumps(item)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str)
    args = parser.parse_args()
    main(args.dataset_path)
