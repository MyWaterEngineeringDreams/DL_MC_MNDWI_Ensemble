#16_UResNetMNDWI_Preds_Sen2_GRY_LKJ

"""
Predict flood extent from Sentinel-2 RGB and compare with MNDWI groundtruth.
Groundtruth MNDWI is thresholded at 0.22 with masking.
"""

import os
import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, jaccard_score, confusion_matrix
)
import arcpy

# Paths – update these to your GEE export folders
RGB_DIR = r"D:\GEEExports\Gerinya_LKJ\RGB"
MNDWI_DIR = r"D:\GEEExports\Gerinya_LKJ\MNDWI"
JRC_DIR = r"D:\JRCWaterExtentData"
MODEL_PATH = r"D:\Models\UNet\UResNet34.dlpk"
CLIP_SHP = r"D:\ClipBoundaries\clip.shp"
OUTPUT_DIR = r"D:\FloodOutputs\UResNet_vs_MNDWI"


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Load clip boundary
clip = gpd.read_file(CLIP_SHP)
clip = clip.to_crs("EPSG:4326")
geometry = clip.geometry

# Helper: read raster
def read_raster(path):
    with rasterio.open(path) as src:
        arr = src.read()
        meta = src.meta
    return arr, meta

# Helper: clip raster to AOI
def clip_raster(path, geometry):
    with rasterio.open(path) as src:
        out_image, out_transform = mask(src, geometry, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
    return out_image, out_meta

# Helper: evaluate two binary maps
def evaluate(pred, ref):
    mask_valid = (~np.isnan(pred)) & (~np.isnan(ref))
    p = pred[mask_valid].astype(int).flatten()
    r = ref[mask_valid].astype(int).flatten()

    return {
        "Accuracy": accuracy_score(r, p),
        "Precision": precision_score(r, p, zero_division=0),
        "Recall": recall_score(r, p, zero_division=0),
        "F1": f1_score(r, p, zero_division=0),
        "IoU": jaccard_score(r, p, zero_division=0),
        "Confusion": confusion_matrix(r, p)
    }

# Process each pair of RGB + MNDWI images
files = [f for f in os.listdir(RGB_DIR) if f.endswith("_RGB.tif")]

for rgb_file in files:

    print(f"\nProcessing: {rgb_file}")

    base = rgb_file.replace("_RGB.tif", "")
    rgb_path = os.path.join(RGB_DIR, rgb_file)
    mndwi_path = os.path.join(MNDWI_DIR, f"{base}_MNDWI.tif")
    jrc_path = os.path.join(JRC_DIR, "JRC_Extent_10m.tif")

    if not os.path.exists(mndwi_path):
        print(f"MNDWI missing for {base}, skipping.")
        continue

    # Clip RGB
    rgb_arr, _ = clip_raster(rgb_path, geometry)
    rgb_arr = rgb_arr.astype("float32")

    tmp_rgb_path = os.path.join(OUTPUT_DIR, f"tmp_{base}_rgb.tif")

    with rasterio.open(
        tmp_rgb_path, "w",
        driver="GTiff",
        height=rgb_arr.shape[1],
        width=rgb_arr.shape[2],
        count=3,
        dtype="float32"
    ) as dst:
        dst.write(rgb_arr)

    # Deep learning prediction
    print("Running U-ResNet34 model prediction...")

    with arcpy.EnvManager(cellSize=10, processorType="GPU"):
        out = arcpy.ia.ClassifyPixelsUsingDeepLearning(
            in_raster=tmp_rgb_path,
            in_model_definition=MODEL_PATH,
            arguments="padding 32;batch_size 8;predict_background True",
            processing_mode="PROCESS_AS_MOSAICKED_IMAGE"
        )

    pred_path = os.path.join(OUTPUT_DIR, f"{base}_UResNetMNDWI_pred.tif")
    out.save(pred_path)
    os.remove(tmp_rgb_path)

    pred_arr = rasterio.open(pred_path).read(1)

    # Clip MNDWI and threshold at 0.22
    mndwi_arr, _ = clip_raster(mndwi_path, geometry)
    mndwi_arr = mndwi_arr[0]

    mndwi_bin = (mndwi_arr > 0.22).astype(np.uint8)

    # Load and clip JRC mask
    jrc_arr, _ = clip_raster(jrc_path, geometry)
    jrc_arr = jrc_arr[0]

    # JRC extent
    jrc_mask = (jrc_arr > 0).astype(np.uint8)

    # Apply mask to both prediction and groundtruth
    pred_masked = np.where(jrc_mask == 1, pred_arr, 0)
    mndwi_masked = np.where(jrc_mask == 1, mndwi_bin, 0)

    # Threshold prediction at 0.5
    pred_bin = (pred_masked > 0.5).astype(np.uint8)

    # Evaluate
    metrics = evaluate(pred_bin, mndwi_masked)

    print(f"Evaluation for {base}:")
    for k, v in metrics.items():
        print(k, v)

    out_masked_path = os.path.join(OUTPUT_DIR, f"{base}_UResNetMNDWIFExt.tif")
    out_gt_path = os.path.join(OUTPUT_DIR, f"{base}_Sen2FextGTruth.tif")

    with rasterio.open(pred_path) as src:
        meta = src.meta.copy()

    meta.update({"dtype": "uint8", "count": 1})

    with rasterio.open(out_masked_path, "w", **meta) as dst:
        dst.write(pred_bin, 1)

    with rasterio.open(out_gt_path, "w", **meta) as dst:
        dst.write(mndwi_masked, 1)

print("\nProcessing done by Kola.")
