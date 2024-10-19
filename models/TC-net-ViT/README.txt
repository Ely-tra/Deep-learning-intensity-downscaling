Tropical Cyclone Intensity Retrieval using Vision Transformer (VIT)

DESCRIPTION:
This workflow processes MERRA2 datasets around historical Tropical Cyclone centers using 
VIT architectures to correct TC intensities of in climate reanalysi dataset. This addresses 
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

HISTORY:
+ Aug 27, 2024: VIT main function created from previous CK's TCG project by TN
+ Sep 10, 2024: Implemented for TC intensity retrieval using the CNN workflow by KL 
+ Oct 16, 2024: packaged by TN and KL
+ Otc 17, 2024: sanity checked and noted by CK  

CONTRIBUTION:
+ Khanh Luong: IU Earth and Atmospheric Sciences (kmluong@iu.edu or mk223338@gmail.com)
+ Tri Nguyen: IU Luddy School of Informatics and Computer Science (trihnguy@iu.edu)
+ Chanh Kieu: IU Earth and Atmospheric Sciences (ckieu@iu.edu)

HOW TO RUN:
Step 1: Run MERRA2TC_domain.py to generate TC domains from IBTracts CSV data and MERRA2 data, 
        given a specific basins, years, and TC names.  

        Notes:
        - Input: MERRA2 model outputs and IbTrACs data. Output: NETCDF files containing 
	  MERRA2 data centered on each TC center, with a given dimension.
        - Domain files are organized by basin, year, and storm name in the following
	  structure: BASIN/YEAR/NAME. 
        - Files are named using the convention: MERRA_TC{domain size}YYYYMMDDHH_{suffix}, 
	  where the domain size is denoted as SSxSS degrees, YYYYMMDDHH represents the 
	  timestamp, and the suffix is a unique identifier to distinguish multiple active 
          storms recorded at the same time.
          For example, the file NA/2001/MERRA_TC18x182001010112_4 corresponds to one 
	  (the fourth recorded one) of many TCs active at 12Z on January 1, 2001, in the 
	  North American basin, with a domain size of 18x18 degrees. 
        - If any pathway related problem arises, it is from MERRA2TC_domain.py.

Step 2: Run TC-extract_data_TSU.py to extract some meteorological fields such as wind, temp, 
        or RH for a small domain from Step 1. This script will also produce other files 
        containing additional TC intensity information such as location of the TC center,
        the day of the year in the form of sines and cosines (embedding position) that 
        indicates when the frame is taken. Files are separeted into months. Save as NumPy 
        files.

        Notes:
        - Input: data from Step 1 outputs.  
	- Output: NumPy files with the same dimensions as Step 1 outputs but contains only
	  specific variables and levels.
        - Naming convention: CNNfeatures{number of channel used}{basin}.{domain size}{month}.npy

Step 3: Run TC-CA_NaN_filling_kfold.py to eliminate NaN values from Step 2 outputs.

        Notes:
	- Input: Step 2 outputs; Output: NaN-free datasets.
	- Naming convention: Step 2 names with suffix "fixed" before .npy, but with the same
	  CNNfeatures{number of channel used}{basin}.{domain size}{month}fixed.npy

Step 5: Run TC-Split_KFold.py to separate data into two training/test datasets. Users need to 
	set all path/sizes within the script. 

        Notes:
	- Input: Features and labels files; Output: Training and testing sets in .npy format.

Step 6: Run TC-build_model.py VMAX/PMIN/RMW to train a VIT model with Step 5 output. Note there are 
	separate script for Vmax, Pmin, and RMW. All model parameters are given inside the 
  	scripts. Need to manually edit these parameter and data paths for now.

Step 7: Run TC-test_plot.py VMAX/PMIN/RMW to evaluate model performance on a test set for Vmax.  
	Note that all test sets are named according to the convention 
        test{number_of_channel}x/y.{domain_size}.npy.

DISCLAIMER:
This project is funded by the US National Science Foundation (NSF, PI: Chanh Kieu). Any 
dissemination of information, code, or data must be in accordance with NSF guidelines and 
regulations.
