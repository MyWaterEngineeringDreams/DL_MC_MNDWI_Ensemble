#2. reflectance_utils.py
import os
import glob
import re
import numpy as np
import pandas as pd
from datetime import datetime
import rasterio
from rasterio.mask import mask


def get_acquisition_date(mtl_file):
    with open(mtl_file, "r") as f:
        content = f.read()
    match = re.search(r"DATE_ACQUIRED\s=\s\"?(\d{4}-\d{2}-\d{2})\"?", content)
    return datetime.strptime(match.group(1), "%Y-%m-%d") if match else None


def determine_flood_type(date):
    m = date.month
    if 6 <= m <= 11:
        return "White Flood"
    if m in [12, 1, 2, 3]:
        return "Black Flood"
    return "Other"


def compute_mean_reflectance(scene_folder, reservoir):
    values = []
    band_list = [1, 2, 3, 4, 5, 6, 7]

    for b in band_list:
        tif = glob.glob(os.path.join(scene_folder, f"*SR_B{b}.TIF"))
        if not tif:
            values.append(np.nan)
            continue
        try:
            with rasterio.open(tif[0]) as src:
                arr, _ = mask(src, reservoir.geometry, crop=True)
                img = arr[0].astype("float32")
                img[img == 0] = np.nan
                values.append(np.nanmean(img))
        except Exception:
            values.append(np.nan)
    return values


def process_folder(path, reservoir):
    rows = []
    scenes = glob.glob(os.path.join(path, "*"))

    for s in scenes:
        mtl = glob.glob(os.path.join(s, "*_MTL.txt"))
        if not mtl:
            continue

        date = get_acquisition_date(mtl[0])
        if not date:
            continue

        flood = determine_flood_type(date)
        reflectance = compute_mean_reflectance(s, reservoir)

        if np.isnan(reflectance).any():
            continue

        rows.append({
            "Scene": os.path.basename(s),
            "Date": date.strftime("%Y-%m-%d"),
            "Flood Type": flood,
            "Coastal": reflectance[0],
            "Blue": reflectance[1],
            "Green": reflectance[2],
            "Red": reflectance[3],
            "NIR": reflectance[4],
            "SWIR1": reflectance[5],
            "SWIR2": reflectance[6]
        })

    return pd.DataFrame(rows)
