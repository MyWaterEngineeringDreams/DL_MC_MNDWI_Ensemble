# DL_MC_MNDWI_Ensemble

# DeepLearningMonteCarloSpectralIndices_Ensemble_FloodMapping

Code repository for my manuscript submission titled:

**Hybrid Deep Learning - Monte Carlo-based MNDWI Ensemble for Probabilistic Water Extent Mapping under Dynamic Spectral Flow Conditions in Large Waterbodies**

Submitted to *Remote Sensing (MDPI)*.

## Summary

Accurate surface water mapping in river–reservoir systems remains challenging due to mixed pixels, turbidity, cloud contamination, and variable water spectral behavior under rapidly varying inflows. Spectral index approaches often fail in transition zones due to manual thresholding problems, while deep learning models also struggle to generalize under dynamic hydrologic conditions.

This repository implements a probabilistic ensemble framework that integrates deep learning segmentation with Monte Carlo–based MNDWI pertubations to improve delineation under complex flow regimes. The framework was developed and tested over the Kainji Reservoir and flood events along the transboundary Niger River system.

## Instruction

- Run code accordingly with adequate input data 

## Repository Structure
```text
Scripts/
├── *.py                  # Executable workflows
├── requirements.txt      # Preserved dependency list
├── Figs/                 # Figures for analysis and manuscript
├── GIS Files/            # Raster and vector datasets

## Environment

All dependencies are listed in `requirements.txt`.  
Deep learning inference relies on the ArcGIS Pro Python environment and ArcPy.

## Notes

- Large raster datasets are not included and must be supplied locally.
- File paths in scripts may require adjustment depending on the local setup.
- The repository is intended for research reproducibility and peer review.

## Citation

Please cite the associated Remote Sensing (MDPI) submission when using this code.

Repository initialized on 2025-12-18 by Authors.
