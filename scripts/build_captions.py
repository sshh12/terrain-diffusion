from functools import partial
from concurrent.futures import ThreadPoolExecutor
import threading
import itertools
import argparse
import glob
import queue
import json
import os

from transformers import CLIPProcessor, CLIPModel
from PIL import Image

BASE_CAPTION = "a satellite image"
HIGH_LEVEL_LANDFORM = [
    "island",
    "ocean",
    "desert",
    "beach",
    "mountains",
    "forest",
    "urban area",
    "farmland",
    "river",
    "lake",
    "volcano",
    "rainforest",
    "coral reef",
    "swamp",
    "tundra",
    "savannah",
    "wetlands",
    "grassland",
    "plateau",
    "canyon",
    "ice cap",
    "archipelago",
    "reef",
    "lagoon",
    "bay",
    "isthmus",
    "straits",
    "gulf",
    "marsh",
    "coastline",
    "valley",
    "plain",
    "atoll",
    "steppe",
]
LANDFORM_ITEMS = [
    "islands",
    "oceans",
    "deserts",
    "mountains",
    "forests",
    "urban areas",
    "farmlands",
    "rivers",
    "lakes",
    "volcanoes",
    "glaciers",
    "rainforests",
    "coral reefs",
    "sand dunes",
    "swamps",
    "tundras",
    "savannahs",
    "deltas",
    "wetlands",
    "grasslands",
    "plateaus",
    "canyons",
    "ice caps",
    "archipelagos",
    "reefs",
    "lagoons",
    "cliffs",
    "bays",
    "peninsulas",
    "isthmuses",
    "straits",
    "gulfs",
    "marshes",
    "coastlines",
    "peaks",
    "valleys",
    "plains",
    "ridges",
    "fjords",
    "atolls",
    "steppes",
    "craters",
    "clouds",
    "agricultural fields",
    "towns",
    "villages",
]

COUNT_PHRASES = ["a few", "several", "some", "full of"]

SINGLE_PHRASES = ["a large", "a small"]


def _get_landform_phrases():
    phrases = {}
    for ph, item in itertools.product(COUNT_PHRASES, LANDFORM_ITEMS):
        phrases[ph + " " + item] = item
    for ph, item in itertools.product(SINGLE_PHRASES, LANDFORM_ITEMS):
        phrases[ph + " " + item[:-1]] = item
    return phrases


def _max(scores):
    return max(scores, key=scores.get)


def _score_image(model, processor, texts, img):
    inputs = processor(text=texts, images=img, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    scores = logits_per_image.detach().numpy()[0]
    return dict(zip(texts, scores))


def _build_caption(model, processor, img, max_phrases):
    phrases = _get_landform_phrases()

    base_scores = _score_image(
        model, processor, [BASE_CAPTION + ", " + t for t in HIGH_LEVEL_LANDFORM], img
    )
    cur = _max(base_scores)

    landforms = set()
    for _ in range(max_phrases):
        scores = _score_image(
            model,
            processor,
            [cur] + [cur + ", " + t for t in phrases.keys() if t not in landforms],
            img,
        )
        best = _max(scores)
        if best != cur:
            landforms.add(phrases[best.split(",")[-1].strip()])
            cur = best
        else:
            break
    return cur


def _process_load(meta_fn, q):
    with open(meta_fn, "r") as f:
        meta = json.load(f)
    # prefilter
    if (
        meta["rgb_pct_zero"] > 90
        and meta["rgb_stitch_lines"] == 0
        and "caption" not in meta
    ):
        print(f"Adding {meta_fn}")
        q.put(meta_fn)


def _process(q, max_phrases, caption_path, clip_model):
    model = CLIPModel.from_pretrained(clip_model)
    processor = CLIPProcessor.from_pretrained(clip_model)
    with open(caption_path, "w") as f:
        while not q.empty():
            meta_fn = q.get()
            img_fn = meta_fn.replace(".json", ".rgb.png")
            img = Image.open(img_fn)
            caption = _build_caption(model, processor, img, max_phrases)
            print("Built Caption", img_fn, caption)
            f.write(json.dumps({"fn": img_fn, "caption": caption}) + "\n")
            f.flush()


def main(workers, rgb_path, caption_path, max_phrases, clip_model):
    fn_iter = glob.iglob(os.path.join(rgb_path, "*.json"))

    q = queue.Queue(1000)

    thread = threading.Thread(
        target=_process, args=(q, max_phrases, caption_path, clip_model), daemon=True
    )
    thread.start()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        executor.map(partial(_process_load, q=q), fn_iter)

    thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_path", type=str)
    parser.add_argument("--caption_path", type=str)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--max_phrases", type=int, default=5)
    parser.add_argument(
        "--clip_model", type=str, default="openai/clip-vit-large-patch14"
    )

    args = parser.parse_args()
    main(
        args.workers,
        args.rgb_path,
        args.caption_path,
        args.max_phrases,
        args.clip_model,
    )
