# Tropical Cyclone Intensity Retrieval using CNN

## DESCRIPTION:
This workflow processes MERRA2 datasets around historical Tropical Cyclone centers using  CNN architectures to correct TC intensities of in climate reanalysis dataset. This addresses  the limitations of grided climate data outputs from numerical models like WRF or renalaysis data like MERRA, which often could not capture true TC intensity.

The workflow includes all data processing such as extraction and preprocessing of MERRA2, data normalization, NaN value handling... Additionally, CNN can help  build models that can correct TC intensities, using convolutional NN.

Please note that while the workflow is designed to ensure a stable and seamless workflow, it requires carefully check the input and output paths, model parameters between scripts. These paths are provided at the beginning of each script to minimize complications; however, minor issues may still arise. This workflow is specifically tested and run with Python/GPU/3.10.10 on the IU's native BigRed200 HPC environment.

## HISTORY:
+ Jan 26, 2024: created by Khanh Luong from the CNN workflow in CK's previous TCG model
+ May 14, 2024: cross-checked and cleaned up by CK
+ Oct 19, 2024: added a job script, further test, and cleaned up by CK to be consistent with VIT model

## CONTRIBUTION:
+ Khanh Luong: IU Earth and Atmospheric Sciences (kmluong@iu.edu or mk223338@gmail.com)
+ Chanh Kieu: IU Earth and Atmospheric Sciences (ckieu@iu.edu)

## HOW TO RUN:
To run the entire workflow, use the job script `job_br200.sh`, which lists out all parameters and steps to run this system. Note that many of the parameters are currently hardwired to each script, which needed to be changed later on. Below are the full steps of running this workflow manually, using MERRA-2 dataset in NETCDF and IBTrACs data in CVS format: 

**Step 1**: Run `MERRA2TC_domain.py` to generate TC domains from IBTracts CSV data and MERRA2 data, 
        given a specific basins, years, and TC names.  

_Notes_:
- Input: MERRA2 model outputs and IbTrACs data. Output: NETCDF files containing  MERRA2 data centered on each TC center, with a given dimension.
- Domain files are organized by basin, year, and storm name in the following structure: BASIN/YEAR/NAME. 
- Files are named using the convention: MERRA_TC{domain size}YYYYMMDDHH_{suffix}, where the domain size is denoted as SSxSS degrees, YYYYMMDDHH represents the timestamp, and the suffix is a unique identifier to distinguish multiple active storms recorded at the same time. For example, the file NA/2001/MERRA_TC18x182001010112_4 corresponds to one  (the fourth recorded one) of many TCs active at 12Z on January 1, 2001, in the North American basin, with a domain size of 18x18 degrees.   
- If any pathway related problem arises, it is from MERRA2TC_domain.py.

**Step 2**: Run `TC-extract_data.py` to extract some meteorological fields such as wind, temp, or RH for a small domain from Step 1. A related script `TC-extract_data_TSU.py` will produce other files containing additional TC intensity information such as the location of the TC center, the day of the year in the form of sines and cosines (embedding position) that indicate when the frame is taken. Files are separated into months. Save as NumPy files.

_Notes_:
- Input: data from Step 1 outputs.  
- Output: NumPy files with the same dimensions as Step 1 outputs but contain only specific variables and levels.
- Naming convention: CNNfeatures{number of channel used}{basin}.{domain size}{month}.npy
- If users want to run without additional information, use `TC-extract_data.py`. Need to revise this
- This script is currently hard-wired to some specific variable/levels. Need to revise this. 
- There is some warning related to cfgrib, but it should be ok

**Step 3**: Run `TC-CA_NaN_filling.py` to eliminate NaN values from Step 2 outputs.

_Notes_:
- Input: Step 2 outputs; Output: NaN-free datasets.
- Naming convention: Step 2 names with suffix "fixed" before .npy, but with the same CNNfeatures{number of channel used}{basin}.{domain size}{month}fixed.npy

**Step 4**: (optional) Merge data from different basins into a single dataset if not training across multiple basins. Skip if 'regionize' parameter in TC-extract_data.py is False. Inputs: Basin-named files; Output: Unified features and labels with adjusted naming.

_Notes_:
- Naming convention: change {basin} to AL
- CNNfeatures{number of channel used}AL.{domain size}fixed.npy

**Step 5**: Run `TC-Split.py` to separate data into two training/test datasets. Users need to set all path/sizes within the script. 

_Notes_:
- Input: Features and labels files; Output: Training and testing sets in .npy format.
- For this step 5, if one wants to check for each season, use the script `TC-Split_seasonal.py` to generate (x,y) test data for each season (month). This seasonal mode is however not fully tested.

**Step 6**: Run `retrieval_model_vmax_ctl.py` VMAX to train a VIT model with Step 5 output. Note there are separate script for Vmax, Pmin, and RMW. All model parameters are given inside the scripts. Need to manually edit these parameter and data paths for now.

**Step 7**: Run `TC-test_plot.py` VMAX/PMIN/RMW to evaluate model performance on a test set for Vmax.  Note that all test sets are named according to the convention test{number_of_channel}x/y.{domain_size}.npy.

## DISCLAIMER:
This project is funded by the US National Science Foundation (NSF, PI: Chanh Kieu). Any dissemination of information, code, or data must be in accordance with NSF guidelines and regulations.
