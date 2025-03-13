# Workflow Description  

This workflow preprocesses MERRA2 datasets and WRF idealized storm simulations to generate **multi-channel NumPy arrays** and **corresponding cyclone labels**. Outputs are structured for direct use in machine learning pipelines.  

---

## MERRA2 Data Preprocessing  
- **Input**:  
  - MERRA2 atmospheric reanalysis data.  
  - IBTrACS historical cyclone tracks (CSV format).  
- **Output**:  
  - Variables: storm-centered multi-channel NumPy images (user-defined MERRA2 variables in original units).  
  - Labels: Tropical cyclone intensity parameters extracted from IBTrACS:  
    - `VMAX`: Maximum wind speed (**knots**)  
    - `PMIN`: Minimum sea level pressure (**millibars**)  
    - `RMW`: Radius of maximum wind (**nautical miles**)  

---

## WRF Idealized Data Preprocessing  
- **Input**:  
  - WRF simulation outputs (e.g., idealized tropical cyclones).  
- **Output**:  
  - Variables: storm-centered multi-channel NumPy images (user-specified WRF channels in original units).  
  - Labels extracted from:  
    - `VMAX`: Maximum wind speed (**m/s**)  
    - `PMIN`: Minimum sea level pressure (**Pa**)  
    - `RMW`: Radius of maximum wind (**km**)  

---

## WRF Cross-Experiment Variable-Label Mapping  
This script enables pairing variables from one experiment with labels from another. Cross-experimenting allows users to investigate complex nested-grid feedback mechanism from the WRF model.
**Example**:  
- **Input Variables**: Extracted from a WRF 18km × 18km resolution simulation.  
- **Output Labels**: Derived from a nested WRF experiment (18km outer grid with 6km/2km nested grids).  

---

# Data & Output Configuration  
**Location**: Line 28 of the job script  

Configure input data paths, preprocessing outputs, and model results for tropical cyclone prediction workflows.  

---

## Input Data Paths  
| Parameter       | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `besttrack`     | Path to IBTrACS dataset (CSV format) containing ground truth labels (e.g., storm intensity, location). |
| `datapath`      | Directory storing raw MERRA2 data files.                                   |
| `wrf_base`      | Root folder for WRF model output files (alternative data source to MERRA2). |
| `config`        | Path to CNN configuration file (defines model architecture/hyperparameters). |

---

## Working Directory (`workdir`)  
Main directory for workflow outputs. Automatically generates these subfolders:  

| Subdirectory         | Purpose                                                                                   |
|----------------------|-------------------------------------------------------------------------------------------|
| **`TC_domain`**      | Output from **first MERRA2 preprocessing step**:<br> - Stores storm-centered domains with *all* MERRA2 variables/levels.<br> - Organized as `TC_domain/[basin]/[year]/` (e.g., `TC_domain/NA/2017/`). |
| **`Domain_data`**    | Output from **second MERRA2 preprocessing step**:<br> - Contains subsetted variables/levels for training.<br> - Structured as `Domain_data/[experiment_name]/data/`. |
| **`wrf_data`**       | Preprocessed WRF data (resampled resolution, selected variables/levels).                 |
| **`text_report`**    | Evaluation outputs:<br> - Graphs comparing predictions vs. labels.<br> - CSV reports (`prediction` and `label` columns). |

---

## Temporary Folder (`temporary_folder`)  
**Purpose**: Intermediate storage for unified model input files.  

### Workflow Integration  
1. Converts preprocessed MERRA2/WRF data into a standardized format.  
2. Ensures cross-source compatibility before model ingestion.  

**NOTE**: Files here are **automatically deleted** after workflow execution.  

---

## Key Workflow Notes  
- **Multi-Source Support**: Handles MERRA2 and WRF data via dedicated pipelines.  
- **Model Compatibility**: The `temporary_folder` guarantees consistent input formatting.  

---

