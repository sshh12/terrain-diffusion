from concurrent.futures import ThreadPoolExecutor
from functools import partial
import argparse
import tempfile
import urllib
import zipfile
import glob
import json
import os

from PIL import Image
import numpy as np
import rasterio
import rasterio.features
import rasterio.warp
import ee
import cv2


RGB_SIZE = (512, 512)
DEPTH_ICON_SIZE = (512, 512)


def _regions_from_depth(depth_fn, subdivisions):
    with rasterio.open(depth_fn) as dataset:
        depth = dataset.read(1)
        height, width = depth.shape
        h_range = np.linspace(0, height, subdivisions + 1, dtype=np.int16)
        w_range = np.linspace(0, width, subdivisions + 1, dtype=np.int16)

        depth_slices = []
        for h in reversed(range(subdivisions)):
            for w in range(subdivisions):
                depth_slices.append(
                    depth[h_range[h] : h_range[h + 1], w_range[w] : w_range[w + 1]]
                )

        mask = dataset.dataset_mask()
        for geom, _ in rasterio.features.shapes(mask, transform=dataset.transform):
            geom = rasterio.warp.transform_geom(
                dataset.crs, "EPSG:4326", geom, precision=6
            )

        coords = geom["coordinates"][0]
        Xmin = min([coord[0] for coord in coords])
        Xmax = max([coord[0] for coord in coords])
        Ymin = min([coord[1] for coord in coords])
        Ymax = max([coord[1] for coord in coords])

        x_range = np.linspace(Xmin, Xmax, subdivisions + 1)
        y_range = np.linspace(Ymin, Ymax, subdivisions + 1)

        region_slices = []
        for h in range(subdivisions):
            for w in range(subdivisions):
                region = ee.Geometry.Polygon(
                    [
                        [x_range[w], y_range[h]],
                        [x_range[w + 1], y_range[h]],
                        [x_range[w + 1], y_range[h + 1]],
                        [x_range[w], y_range[h + 1]],
                    ]
                )
                region_slices.append(region)

        return zip(region_slices, depth_slices)


def _mask_clouds(image):
    return image


def _download_image_of_region(region, scale):
    ee_img = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterDate("2020-01-01", "2020-05-30")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
        .map(_mask_clouds)
        .select(["TCI_R", "TCI_G", "TCI_B"])
    )
    ee_img = ee_img.median().visualize(
        bands=["TCI_R", "TCI_G", "TCI_B"], min=0.0, max=256, gamma=1
    )
    url = ee_img.getDownloadUrl(
        {
            "scale": scale,
            "region": region,
            "bestEffort": True,
            "crs": "EPSG:4326",
        }
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, "rgb.zip")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmp_dir)
        blue = Image.open(os.path.join(tmp_dir, "download.vis-blue.tif"))
        green = Image.open(os.path.join(tmp_dir, "download.vis-green.tif"))
        red = Image.open(os.path.join(tmp_dir, "download.vis-red.tif"))
    return Image.merge("RGB", (red, green, blue))


def _count_stitch_lines(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 5, 50, 3)

    rho = 1
    theta = np.pi / 180
    thresh = 200
    min_line_length = 50
    max_line_gap = 20

    lines = cv2.HoughLinesP(
        edges,
        rho,
        theta,
        thresh,
        np.array([]),
        min_line_length,
        max_line_gap,
    )
    return len(lines) if lines is not None else 0


def _process(fn, rgb_path, scale, subdivision):
    depth_key = os.path.split(fn)[1].split("_")[1]

    mark_path = os.path.join(rgb_path, f"{depth_key}.success")
    if os.path.exists(mark_path):
        print(f"Skipping {fn}")
        return
    print(f"Processing {fn}")

    regions = _regions_from_depth(fn, subdivision)
    for i, (region, depth) in enumerate(regions):
        key = f"{depth_key}-{i}_{subdivision*subdivision}"
        rgb_path = os.path.join(rgb_path, f"{key}.rgb.png")
        depth_path = os.path.join(rgb_path, f"{key}.depth.npy")
        depth_icon_path = os.path.join(rgb_path, f"{key}.depth.png")
        meta_path = os.path.join(rgb_path, f"{key}.json")

        depth_icon = Image.fromarray((depth / (depth.max() + 1) * 255).astype(np.uint8))
        depth_icon.resize(DEPTH_ICON_SIZE, Image.Resampling.LANCZOS).save(
            depth_icon_path
        )
        rgb = _download_image_of_region(region, scale)
        rgb.resize(RGB_SIZE, Image.Resampling.LANCZOS).save(rgb_path)
        np.save(depth_path, depth, allow_pickle=False)

        meta = dict(
            scale=scale,
            subdivision=subdivision,
            i=i,
            coordinates=region["coordinates"],
            region=depth_key,
            depth_max=int(depth.max()),
            depth_min=int(depth.min()),
            depth_pct_zero=int((depth < 0.01).mean() * 100),
            rgb_stitch_lines=_count_stitch_lines(rgb_path),
            rgb_pct_zero=int(
                np.array(rgb).any(axis=-1).mean() * 100
            ),  # oops this is inverted
        )
        with open(meta_path, "w") as f:
            json.dump(meta, f)

    with open(mark_path, "w") as f:
        f.write("1")


def main(workers, depth_path, rgb_path, scale, subdivision):
    ee.Initialize()
    os.makedirs(rgb_path, exist_ok=True)

    func = partial(_process, rgb_path=rgb_path, scale=scale, subdivision=subdivision)

    fn_iter = glob.iglob(os.path.join(depth_path, "**", "*.tif"), recursive=True)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        executor.map(func, fn_iter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth_path", type=str)
    parser.add_argument("--rgb_path", type=str)
    parser.add_argument("--scale", type=int)
    parser.add_argument("--subdivision", type=int)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.workers, args.depth_path, args.rgb_path, args.scale, args.subdivision)
