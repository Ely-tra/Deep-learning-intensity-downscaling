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

# How to Run the Workflow  

To run the entire workflow, execute the workflow using the `job_br200.sh` script, which centralizes configuration parameters. Advanced users may adjust additional parameters within individual Python scripts, though this is generally not required.  

---

## Execution Steps  
1. **Job Script**: `job_br200.sh` defines configurable workflow parameters.  
2. **Preprocessing Stages**:  
   - **MERRA2**: 3 sequential preprocessing steps.  
   - **WRF**: 1 preprocessing step.  
3. **Model Pipeline**:
   - Load preprocessed data.   
   - Build model.   
   - Save results.  

---

## Workflow Control  
### Execution Sequence (Line 21 of `job_br200.sh`)  
```bash
# Example control sequence  
merra=(1 1 1)  # MERRA2 preprocessing steps (1=enabled, 0=disabled)  
wrf=1           # WRF preprocessing (1=enabled)  
model_steps=(1 1 1)  # Model building, data loading, result saving
```
## Key Rules  

### **Mutually Exclusive Execution**  
- MERRA2 and WRF preprocessing cannot run concurrently.  
- Submit separate jobs for each dataset.  

### **WRF Default Behavior**  
- MERRA2 preprocessing is automatically disabled when the data source is set to `WRF`, and vice versa (handled at line 49).  

---

## Modifying Workflow Steps  

### **Example: Run Only the Final MERRA2 Step**  
Edit line 23 in `job_br200.sh`:  
```bash
merra=(0 0 1)  # Skips MERRA2 Steps 1 and 2
```
---
## NOTE
	•	Output Conflicts: Ensure no overlapping jobs are running to prevent file corruption.
	•	WRF-MERRA2 toggle: Line 49
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
# Experiment Naming Configuration  
**Location**: Line 38 of the job script  

Configure experiment-specific naming conventions for outputs and models.  

---

## Parameters  

| Parameter                   | Description                                                                                  |
|-----------------------------|----------------------------------------------------------------------------------------------|
| `text_report_name`          | Name of the CSV report comparing predictions and labels. Saved to `workdir/text_report/[name].csv`. Example: `vmax_validation` creates `vmax_validation.csv`. |
| `experiment_identification` | Placeholder for human-readable experiment notes (no functional impact). Example: `merra2_vmax_test1`. |
| `model_name`                | Base name for saved models. Final name includes:<br> - This base name<br> _ Data source (`MERRA2`, `WRF`)<br> _ Target label (`VMAX`, `PMIN`, `RMW`)<br> - Spatiotemporal enabled. <br> Example: `baseline` → `baseline_MERRA2_VMAX`. |
| `plot_unit`                 | Explicit unit for evaluation plots (no auto-detection). Match to label source:<br> - IBTrACS: `knots` (VMAX), `millibars` (PMIN)<be>, `nmile` (RMW) - WRF: `m/s` (VMAX), `Pa` (PMIN), `km` (RMW). |

---

## Usage Examples  
1. **Basic Configuration**:  
   ```bash
   text_report_name="pmain_predictions.txt"  
   model_name="hurricane_intensity"  
   plot_unit="millibars"
   ```
Generates: `workdir/text_report/pmin_predictions.txt`

Model saved as: `hurricane_intensity_MERRA2_PMIN`
Or `hurricane_intensity_MERRA2_PMIN_st` if `st_embed` is set to 1 (line 68)
