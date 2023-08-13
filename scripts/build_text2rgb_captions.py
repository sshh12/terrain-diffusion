from functools import partial
from concurrent.futures import ThreadPoolExecutor
import threading
import argparse
import glob
import queue
import json
import os

from transformers import CLIPProcessor, CLIPModel
from PIL import Image


def _score_image(model, processor, texts, img):
    inputs = processor(text=texts, images=img, return_tensors="pt", padding=True)
    inputs = inputs.to("cuda")
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    scores = logits_per_image.cpu().detach().numpy()[0]
    return dict(zip(texts, scores))


def _get_existing_paths(caption_path):
    if not os.path.exists(caption_path):
        return set()
    else:
        with open(caption_path, "r") as f:
            return set(json.loads(line)["meta_fn"] for line in f)


def _process_load(meta_fn, q, apply_meta_filter, skip_list):
    if meta_fn in skip_list:
        print(f"Skipping {meta_fn}")
        return
    elif not apply_meta_filter:
        q.put(meta_fn)
        return
    else:
        with open(meta_fn, "r") as f:
            meta = json.load(f)
        # prefilter
        if meta["rgb_pct_non_zero"] > 90 and meta["rgb_stitch_lines"] == 0:
            print(f"Adding {meta_fn}")
            q.put(meta_fn)


def _process(q, caption_path, captions, clip_model):
    model = CLIPModel.from_pretrained(clip_model).to("cuda")
    processor = CLIPProcessor.from_pretrained(clip_model)
    with open(caption_path, "a") as f:
        while True:
            meta_fn = q.get()
            img_fn = meta_fn.replace(".json", ".rgb.png")
            img = Image.open(img_fn)
            scores = _score_image(model, processor, captions, img)
            caption = max(scores, key=scores.get)
            print("Built Caption", img_fn, caption)
            f.write(
                json.dumps({"meta_fn": meta_fn, "rgb_fn": img_fn, "caption": caption})
                + "\n"
            )
            f.flush()


def main(workers, rgb_path, caption_vocab, caption_path, apply_meta_filter, clip_model):
    fn_iter = glob.iglob(os.path.join(rgb_path, "*.json"))
    skip_list = _get_existing_paths(caption_path)

    with open(caption_vocab, "r") as f:
        captions = [line.strip() for line in f.readlines() if len(line.strip()) > 0]

    q = queue.Queue(1000)

    thread = threading.Thread(
        target=_process, args=(q, caption_path, captions, clip_model), daemon=True
    )
    thread.start()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        executor.map(
            partial(
                _process_load,
                q=q,
                apply_meta_filter=apply_meta_filter,
                skip_list=skip_list,
            ),
            fn_iter,
        )

    thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_path", type=str)
    parser.add_argument(
        "--caption_vocab", type=str, default="./scripts/captions_text2rgb.txt"
    )
    parser.add_argument("--caption_path", type=str)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--apply_meta_filter", type=bool, default=True)
    parser.add_argument(
        "--clip_model", type=str, default="openai/clip-vit-large-patch14"
    )

    args = parser.parse_args()
    main(
        args.workers,
        args.rgb_path,
        args.caption_vocab,
        args.caption_path,
        args.apply_meta_filter,
        args.clip_model,
    )
