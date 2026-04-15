#8_mndwi_MonteCarlo_UResNetMNDWI.py
import os
import numpy as np
import rasterio

folder_path = r"Final L"
uresnet_pred_folder = r"Uresnetpreds2018_2024"
output_folder = r"UResNetMNDWIEns"

GREEN_BAND = 3
SWIR_BAND = 6

MC_RUNS = 1000
THRESH_MIN = 0.2
THRESH_MAX = 0.4

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Compute MNDWI

def compute_mndwi(green, swir):
    numerator = green - swir
    denominator = green + swir
    mndwi = np.divide(
        numerator, denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=(denominator != 0)
    )
    return mndwi

# Monte Carlo Thresholding
# Creates 1000 realizations across 0.2–0.4
# Returns probability surface: freq(water) / MC_RUNSs

def monte_carlo_mndwi_probability(mndwi):
    thresholds = np.linspace(THRESH_MIN, THRESH_MAX, MC_RUNS)

    water_count = np.zeros(mndwi.shape, dtype=np.uint16)

    for th in thresholds:
        mask = mndwi >= th
        water_count += mask.astype(np.uint16)

    probability = water_count.astype(np.float32) / MC_RUNS
    return np.clip(probability, 0, 1)

tif_files = [f for f in os.listdir(input_folder) if f.endswith(".tif")]

for tif_name in tif_files:
    print(f"Processing: {tif_name}")

    tif_path = os.path.join(input_folder, tif_name)
    uresnet_prob_path = os.path.join(
        uresnet_pred_folder,
        tif_name.replace(".tif", "_prob.tif")
    )

    if not os.path.exists(uresnet_prob_path):
        print(f"Missing UNet probability raster for: {tif_name}")
        continue


    with rasterio.open(tif_path) as src:
        green = src.read(GREEN_BAND).astype(np.float32)
        swir = src.read(SWIR_BAND).astype(np.float32)
        profile = src.profile.copy()

    mndwi = compute_mndwi(green, swir)

    # Monte Carlo MNDWI probability
    print("Running Monte Carlo thresholding...")
    mndwi_prob = monte_carlo_mndwi_probability(mndwi)

    # UNet-ResNet34 probability raster
    with rasterio.open(uresnet_prob_path) as up:
        uresnet_prob = up.read(1).astype(np.float32)

    # Maximum positive ensemble
    ensemble_prob = np.maximum(uresnet_prob, mndwi_prob)

    # Final binary mask using 0.5 threshold (This will get rid of water artefacts and mixed LU water pixels)
    ensemble_binary = (ensemble_prob >= 0.5).astype(np.uint8)

    prob_output = os.path.join(
        output_folder,
        tif_name.replace(".tif", "_UResNetMNDWI_MC_prob.tif")
    )

    profile.update(dtype=rasterio.float32, count=1)

    with rasterio.open(prob_output, "w", **profile) as dst:
        dst.write(ensemble_prob.astype(np.float32), 1)

    # Save final mask
    mask_output = os.path.join(
        output_folder,
        tif_name.replace(".tif", "_UResNetMNDWI_MC_mask.tif")
    )

    profile.update(dtype=rasterio.uint8)

    with rasterio.open(mask_output, "w", **profile) as dst:
        dst.write(ensemble_binary, 1)

    print(f"Saved probability: {prob_output}")
    print(f"Saved mask: {mask_output}")
