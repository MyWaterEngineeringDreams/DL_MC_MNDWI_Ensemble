#5. run_preprocessing
"""
Model Training Script for Kainji Reservoir Deep Learning Segmentation
Author: [Kola]

This script trains multiple architectures (UNet, DeepLab, PSPNet) with
ResNet34 and VGG16 backbones using ArcGIS Image Analyst's deep learning tools.
"""

import time
import arcpy

# Training Utility Function
def train_model(
    in_folder,
    out_folder,
    model_type,
    backbone,
    max_epochs=100,
    batch_size=16,
    validation_percentage=20,
):
    """Wrapper for ArcGIS TrainDeepLearningModel."""

    params = (
        "class_balancing False;"
        "mixup False;"
        "focal_loss False;"
        "ignore_classes #;"
        "chip_size 224;"
        "monitor valid_loss"
    )

    t0 = time.time()

    with arcpy.EnvManager(processorType="GPU"):
        arcpy.ia.TrainDeepLearningModel(
            in_folder=in_folder,
            out_folder=out_folder,
            max_epochs=max_epochs,
            model_type=model_type,
            batch_size=batch_size,
            arguments=params,
            learning_rate=None,
            backbone_model=backbone,
            pretrained_model=None,
            validation_percentage=validation_percentage,
            stop_training="STOP_TRAINING",
            freeze="FREEZE_MODEL",
            augmentation="DEFAULT",
            monitor="VALID_LOSS",
        )

    t1 = time.time()
    print(
        f"{model_type}-{backbone} training completed in {(t1 - t0):.2f} seconds."
    )

# Paths
TRAIN_DATA = r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\DLSeg\New Model\1TrainSR"
OUTPUT_BASE = r"D:\Document folder\Projects\Research_Nigeria\Kainji Lake\DLSeg\New Model\2TrainedModels"


# Training Execution
def main():

    # UNET Models
    train_model(
        in_folder=TRAIN_DATA,
        out_folder=f"{OUTPUT_BASE}\\UResNet",
        model_type="UNET",
        backbone="RESNET34",
    )

    train_model(
        in_folder=TRAIN_DATA,
        out_folder=f"{OUTPUT_BASE}\\UVGG16",
        model_type="UNET",
        backbone="TIMM:VGG16_BN",
    )

    # DEEPLAB Models
    train_model(
        in_folder=TRAIN_DATA,
        out_folder=f"{OUTPUT_BASE}\\DpLResNet",
        model_type="DEEPLAB",
        backbone="RESNET34",
    )

    train_model(
        in_folder=TRAIN_DATA,
        out_folder=f"{OUTPUT_BASE}\\DpLVGG16",
        model_type="DEEPLAB",
        backbone="TIMM:VGG16_BN",
    )

    # PSPNET Models
  
    train_model(
        in_folder=TRAIN_DATA,
        out_folder=f"{OUTPUT_BASE}\\PSPResNet",
        model_type="PSPNET",
        backbone="RESNET34",
    )

    train_model(
        in_folder=TRAIN_DATA,
        out_folder=f"{OUTPUT_BASE}\\PSPVGG16",
        model_type="PSPNET",
        backbone="TIMM:VGG16_BN",
    )

# Entrypoint
if __name__ == "__main__":
    main()
