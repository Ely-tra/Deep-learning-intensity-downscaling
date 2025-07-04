# Tropical Cyclone Intensity Retrieval using Vision Transformer (VIT)

## DESCRIPTION:
This workflow processes MERRA2 datasets around historical Tropical Cyclone centers using 
VIT architectures to correct TC intensities of in climate reanalysis dataset. This addresses 
the limitations of grided climate data outputs from numerical models like WRF or renalaysis 
data like MERRA, which often could not capture true TC intensity.

The workflow includes all data processing such as extraction and preprocessing of MERRA2, 
data normalization, NaN value handling... Additionally, VIT can help  build models that can 
correct TC intensities, using encoder attention mechanism instead of convolutional NN.

Please note that while the workflow is designed to ensure a stable and seamless workflow, it 
requires carefully check the input and output paths, model parameters between scripts. 
These paths are provided at the beginning of each script to minimize complications; 
however, minor issues may still arise. This workflow is specifically tested and run 
with Python/GPU/3.10.10 on the IU's native BigRed200 HPC environment.

## HISTORY:
+ Aug 27, 2024: VIT main function created from previous CK's TCG project by TN
+ Sep 10, 2024: Implemented for TC intensity retrieval using the CNN workflow by KL 
+ Oct 16, 2024: packaged by TN and KL
+ Oct 17, 2024: sanity checked and noted by CK  
+ Nov 17, 2024: revised workflow by TN and KL.
+ Nov 19, 2024: sanity checked by CK
+ Mar 22, 2025: added support for ALL in a single workflow and support for WRF output
                similar to the TCNN version by CK.

## CONTRIBUTION:
+ Khanh Luong: IU Earth and Atmospheric Sciences (kmluong@iu.edu or mk223338@gmail.com)
+ Tri Nguyen: IU Luddy School of Informatics and Computer Science (trihnguy@iu.edu)
+ Chanh Kieu: IU Earth and Atmospheric Sciences (ckieu@iu.edu)

## HOW TO RUN:
**Step 1**: Run `MERRA2TC_domain.py` to generate TC domains from IBTracts CSV data and MERRA2 data, 
        given a specific basins, years, and TC names.  

_Notes_:
- Input:
  + MERRA2 model outputs and IbTrACs data.
  + Windowsize: 19x19 [default]
  + regions: EP NA WP
  + Minimum Latitude and Maximum Latitude
  + Minimum Wind Speed and Maximum Wind Speed
  + Minimum pressure and Maximum Pressure
  + Minimum Radius Maximum Wind and Maximum Radius Maximum Wind
- Output: NETCDF files containing  MERRA2 data centered on each TC center, with a given dimension.
- Domain files are organized by basin, year, and storm name in the following structure: BASIN/YEAR/NAME. 
- Files are named using the convention: MERRA_TC{domain size}YYYYMMDDHH_{suffix}, where the domain size is denoted as SSxSS degrees, YYYYMMDDHH represents the timestamp, and the suffix is a unique identifier to distinguish multiple active storms recorded at the same time. For example, the file NA/2001/MERRA_TC18x182001010112_4 corresponds to one  (the fourth recorded one) of many TCs active at 12Z on January 1, 2001, in the North American basin, with a domain size of 18x18 degrees.   
- If any pathway related problem arises, it is from MERRA2TC_domain.py.

**Step 2**: Run `TC-extract_data_TSU.py` to extract some meteorological fields such as wind, temp, or RH for a small domain from Step 1. This script will also produce other files containing additional TC intensity information such as the location of the TC center, the day of the year in the form of sines and cosines (embedding position) that indicate when the frame is taken. Files are separated into months. Save as NumPy files.

_Notes_:
- Input:
   + Data directory from Step 1 output(Example: /N/project/Typhoon-deep-learning/output-Tri/TC_domain/)
   + Working Directory: (Example: /N/project/Typhoon-deep-learning/output-Tri/)
   + Windowsize: 19x19 [default]
   + List of variables to extract
- Output: NumPy files with the same dimensions as Step 1 outputs but contain only specific variables and levels.
- Naming convention: CNNfeatures{number of channel used}{basin}.{domain size}{month}.npy

**Step 3**: Run `TC-CA_NaN_filling_kfold.py` to eliminate NaN values from Step 2 outputs.

_Notes_:
- Input:
   + Working directory from Step2 output
   + Windowsize: 19x19[default]
   + Number of variables
- Output: NaN-free datasets.
- Naming convention: Step 2 names with suffix "fixed" before .npy, but with the same CNNfeatures{number of channel used}{basin}.{domain size}{month}fixed.npy


**Step 4**: Run `TC-build_model.py` 
- Input:
   + Mode: VMAX PMIN RMW
   + Working Directory
   + Windowsize: 19x19 [default]
   + Number of variables
   + Width and Height of Input
   + Space-Time embedding
   + Validation and Test year
   + Vision Transformer configurable parameters
- VMAX/PMIN/RMW to train a VIT model with Step 3 output.

**Step 5**: Run `TC-test_plot.py` 
- Input:
   + Mode : VMAX PMIN RMW
   + Working Directory
   + Windowsize: 19x19 [default]
   + Number of variables
   + Width and Height of Input
   + Space-Time embedding
   + Validation and Test year
- VMAX/PMIN/RMW to evaluate model performance on a test set for Vmax.  Note that all test sets are named according to the convention test{number_of_channel}x/y.{domain_size}.npy.

## DISCLAIMER:
This project is funded by the US National Science Foundation (NSF, PI: Chanh Kieu). Any 
dissemination of information, code, or data must be in accordance with NSF guidelines and 
regulations.
