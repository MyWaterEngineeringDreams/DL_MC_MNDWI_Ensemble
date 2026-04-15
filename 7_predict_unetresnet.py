#7_predict_unetresnet.py
#Iterate through all L9 files and ignore the first WBF17 training data
import os

folder_path = r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\DLSeg\SWE Prediction\Final L" 
tif_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.tif') and file != 'WetBlackL2017.tif']

for tif_file in tif_files:
    print(f"Processing: {tif_file}")
    
    t0 = time.time()
    
    with arcpy.EnvManager(cellSize=30, processorType="GPU"):
        out_classified_raster = arcpy.ia.ClassifyPixelsUsingDeepLearning(
            in_raster=tif_file,
            in_model_definition=r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\DLSeg\SWE Prediction\Models\UNet\RESNET34\RESNET34.dlpk",
            arguments="padding 56;batch_size 16;predict_background True;test_time_augmentation False;tile_size 224",
            processing_mode="PROCESS_AS_MOSAICKED_IMAGE",
            out_classified_folder=None,
            out_featureclass=None
        )
    
    output_path = os.path.join(r"Uresnetpreds2018_2024", f"SWEUNetRESNET34_{os.path.basename(tif_file)[:-4]}")
    out_classified_raster.save(output_path)
    
    t1 = time.time()
    print(f'Optimal UNetRESNET34 Prediction Runtime for {os.path.basename(tif_file)}: %.2f s' % (t1 - t0))