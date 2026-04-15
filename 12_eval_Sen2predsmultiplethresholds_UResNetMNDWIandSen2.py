
"""
Evaluate U-Net/MNDWI flood predictions against Sentinel-2 and ISO reference maps
using multiple thresholds: Threshold params {0.10, 0.20, 0.22, 0.40, 0.90} .
"""

import os
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import numpy as np
from skimage.transform import resize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, confusion_matrix


# Paths
SEN2_DIR = r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\DLSeg\GIS Files_DLSeg\00Sen2GrndTruthFromGEE\EarthEngineExports"
ISO_DIR = r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\DLSeg\IsoCF"
URESMNDWI_DIR = r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\DLSeg\Final UResNetMNDWI rasters"
CLIP_SHP = r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\ManuKainji to Professor\General format\ManuKainjiPhd\DLSeg\GIS Files_DLSeg\ClipAfterPred.shp"

# Load clip boundary
clip = gpd.read_file(CLIP_SHP)
clip = clip.to_crs("EPSG:4326")

# Sample raster to extract correct CRS
sample_raster_path = os.path.join(SEN2_DIR, "WF2020.tif")
with rasterio.open(sample_raster_path) as src:
    target_crs = src.crs

clip = clip.to_crs(target_crs)
geometry = clip.geometry

# Helper function: Clip raster

def clip_raster(raster_path, geometry):
    with rasterio.open(raster_path) as src:
        out_image, out_transform = mask(src, geometry, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        return out_image[0], out_meta

# Load and clip all raster types
years = ["2019", "2020", "2021", "2022", "2023", "2024"]
globals_data = {}

for year in years:

    # Before Flood (BF)
    for prefix, directory, fname in [
        ("Sen2_BF", SEN2_DIR, f"BF{year}.tif"),
        ("ISO_BF",  ISO_DIR, f"iso3BF{year}.tif"),
        ("UNet_BF", URESMNDWI_DIR, f"BF{year}.tif")
    ]:
        path = os.path.join(directory, fname)
        if os.path.exists(path):
            try:
                clipped, _ = clip_raster(path, geometry)
                globals_data[f"{prefix}{year}"] = clipped
                print(f"Clipped {prefix}{year}")
            except Exception as e:
                print(f"Error clipping {prefix}{year}: {e}")

    # Wet Flood (WF)
    for prefix, directory, fname in [
        ("Sen2_WF", SEN2_DIR, f"WF{year}.tif"),
        ("ISO_WF",  ISO_DIR, f"iso3WF{year}.tif"),
        ("UNet_WF", URESMNDWI_DIR, f"WF{year}.tif")
    ]:
        path = os.path.join(directory, fname)
        if os.path.exists(path):
            try:
                clipped, _ = clip_raster(path, geometry)
                globals_data[f"{prefix}{year}"] = clipped
                print(f"Clipped {prefix}{year}")
            except Exception as e:
                print(f"Error clipping {prefix}{year}: {e}")

# Resize Sentinel-2 to 30 m
target_shape = (3019, 1356)

for year in years:
    for prefix in [f"Sen2_BF{year}", f"Sen2_WF{year}"]:
        if prefix in globals_data:
            arr_10m = globals_data[prefix]
            arr_30m = resize(arr_10m, target_shape, order=1, mode="reflect", anti_aliasing=True)
            globals_data[prefix] = arr_30m
            print(f"{prefix} resized to {arr_30m.shape}")

# Load U-ResNet-MNDWI predictions
for year in years:
    try:
        globals_data[f"BF{year}"] = rasterio.open(
            os.path.join(URESMNDWI_DIR, f"BF{year}.tif")
        ).read(1)

        globals_data[f"WF{year}"] = rasterio.open(
            os.path.join(URESMNDWI_DIR, f"WF{year}.tif")
        ).read(1)

        print(f"Loaded UResMNDWI BF{year} WF{year}")

    except Exception as e:
        print(f"Error loading UResMNDWI for {year}: {e}")

# Evaluation utilities
def evaluate_model(pred, ref):
    mask = (~np.isnan(pred)) & (~np.isnan(ref)) & (ref != -9999) & (pred != -9999)
    pred_f = pred[mask].astype(int).flatten()
    ref_f = ref[mask].astype(int).flatten()

    return {
        "Accuracy": accuracy_score(ref_f, pred_f),
        "Precision": precision_score(ref_f, pred_f, average="macro", zero_division=0),
        "Recall": recall_score(ref_f, pred_f, average="macro", zero_division=0),
        "F1": f1_score(ref_f, pred_f, average="macro", zero_division=0),
        "IoU": jaccard_score(ref_f, pred_f, average="macro", zero_division=0),
        "Confusion": confusion_matrix(ref_f, pred_f)
    }


def threshold_array(arr, thr):
    return (arr > thr).astype(np.uint8)


def binarize_iso(arr):
    return np.where(arr == 1, 1, 0)

# Threshold set
THRESHOLDS = [0.10, 0.20, 0.22, 0.40, 0.90]

# Full evaluation
results = {}

for flood_type in ["WF", "BF"]:
    for year in years:
        try:
            pred = globals_data[f"{flood_type}{year}"]
            sen2_raw = globals_data[f"Sen2_{flood_type}{year}"]
            iso_raw = globals_data[f"ISO_{flood_type}{year}"]

            year_results = {}

            for thr in THRESHOLDS:
                sen2_bin = threshold_array(sen2_raw, thr)
                iso_bin = binarize_iso(iso_raw)

                eval_sen2 = evaluate_model(pred, sen2_bin)
                eval_iso = evaluate_model(pred, iso_bin)

                year_results[f"thr_{thr}"] = {
                    "UResMNDWI vs Sen2": eval_sen2,
                    "UResMNDWI vs ISO": eval_iso
                }

            results[f"{flood_type}{year}"] = year_results
            print(f"Evaluated {flood_type}{year}")

        except Exception as e:
            print(f"Error evaluating {flood_type}{year}: {e}")
# Save results
import json
output_json = "evaluation_results.json"

with open(output_json, "w") as f:
    json.dump(results, f, indent=4)

print(f"Saved evaluation results to {output_json}")
