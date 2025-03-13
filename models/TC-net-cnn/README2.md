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
| **`TC_domain`**      | Output from **first MERRA2 preprocessing step**:<br> - Stores storm-centered domains with *all* MERRA2 variables/levels.<br> - Organized as `TC_domain/[basin]/[year]/` (e.g., `TC_domain/atlantic/2023/`). |
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
- **Reproducibility**: Directory structures (e.g., `TC_domain/basin/year/`) ensure traceable preprocessing.  
- **Multi-Source Support**: Handles MERRA2 and WRF data via dedicated pipelines.  
- **Model Compatibility**: The `temporary_folder` guarantees consistent input formatting.  

---

This setup enables organized, scalable deep learning experiments for storm prediction tasks.  
