#1. metadata_utils.py
import os
import glob
import pandas as pd

def parse_mtl_file(mtl_path):
    meta = {}
    with open(mtl_path, "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split(" = ")
                meta[key.strip()] = value.strip().strip('"')
    return meta


def assign_flood_type(date_str):
    try:
        month = int(date_str.split("-")[1])
        if month in [12, 1, 2, 3]:
            return "BF"
        if month in [6, 7, 8, 9, 10, 11]:
            return "WF"
        return "Unknown"
    except Exception:
        return "Unknown"


def extract_metadata_from_folder(base_path):
    records = []
    folders = [f.path for f in os.scandir(base_path) if f.is_dir()]

    for folder in folders:
        mtl_files = glob.glob(os.path.join(folder, "*MTL.txt"))
        if not mtl_files:
            continue

        meta = parse_mtl_file(mtl_files[0])
        date = meta.get("DATE_ACQUIRED", "0000-00-00")
        flood = assign_flood_type(date)

        row = {
            "FILENAME": os.path.basename(folder),
            "Flow Regime": flood,
            "SPACECRAFT_ID": meta.get("SPACECRAFT_ID"),
            "SENSOR_ID": meta.get("SENSOR_ID"),
            "DATE_ACQUIRED": date,
            "COLLECTION_CATEGORY": meta.get("COLLECTION_CATEGORY"),
            "DATA_TYPE": meta.get("DATA_TYPE"),
            "CLOUD_COVER": float(meta.get("CLOUD_COVER", -1)),
            "SUN_ELEVATION": float(meta.get("SUN_ELEVATION", -1)),
            "SUN_AZIMUTH": float(meta.get("SUN_AZIMUTH", -1)),
            "WRS_PATH": meta.get("WRS_PATH"),
            "WRS_ROW": meta.get("WRS_ROW"),
        }
        records.append(row)

    return pd.DataFrame(records)
