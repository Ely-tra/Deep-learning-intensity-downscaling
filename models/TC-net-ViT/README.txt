Step: MERRA2TCdomain, TC-extract_data_TSU, TC-CA_nan filling, TC-build_model (before I have the time to write the full readme file)

Tropical Cyclone Intensity Correction using CNN

This collection of scripts processes MERRA2 datasets around historical Tropical Cyclone centers. The primary goal is to develop a Convolutional Neural Network (CNN) that corrects the predicted intensities of tropical cyclones. This addresses limitations of numerical models like WRF and MERRA, which often fail to capture the true intensity of these storms.

The scripts cover data processing, which includes the extraction and preprocessing of data from specified MERRA2 domains. Essential tasks such as data normalization and intelligent NaN value handling are integral to preparing the data for modeling. Additionally, the development of the CNN utilizes advanced machine learning techniques to build models that can correct TC intensities more accurately.

Please note that while the scripts are designed to ensure a stable and seamless workflow, it is advisable to carefully check the input and output paths. These paths are set in a straightforward manner to minimize complications; however, minor issues may still arise.

Framework: Utilizing Python 3.10.10 within the native BigRed200 (Indiana University High Performance Cluster) environment.

Step 1: Employ MERRA2TC_domain.py to generate TC domains from IBTracts CSV data and NETCDF model outputs, specifying basins, years, and TC names. Ensure correct file naming for workflow integrity. Input: MERRA2 model outputs; Output: .nc files containing MERRA2 data around each TC center, with even dimensions.

=== NOTE ===
Domain files are organized by basin, year, and storm name in the following structure: BASIN/YEAR/NAME. 

Files are named using the convention: MERRA_TC{domain size}YYYYMMDDHH_{suffix}, where the domain size is denoted as SSxSS degrees, YYYYMMDDHH represents the timestamp, and the suffix is a unique identifier to distinguish multiple active storms recorded at the same time.

For example, the file NA/2001/MERRA_TC18x182001010112_4 corresponds to one (the fourth recorded one) of many tropical cyclones active at 12Z on January 1, 2001, in the North American basin, with a domain size of 18x18 degrees. 

If any pathway related problem arises, it is from MERRA2TC_domain.py.
============

Step 2: Use TC-extract_data_TSU.py to extract wind field, temperature, and humidity from the Step 1 outputs, along with the label files containing TC intensity information and another space-time file that includes the location of the TC center and the day of the year in the form of sines and cosines, indicating when the frame is taken. Files are separeted into months. Save the extracted data as NumPy files following the established naming convention.

Input: Step 1 outputs
Output: NumPy save files with matching dimensions
=== NOTE ===
Naming convention: CNNfeatures{number of channel used}{basin}.{domain size}{month}.npy
============

Step 3: Apply TC-CA_NaN_filling_KFold.py to eliminate NaN values from datasets, ensuring compatibility with subsequent modeling scripts. Input: Step 2 outputs; Output: NaN-free datasets.

=== NOTE ===
Naming convention: Step 2 names with suffix "fixed" before .npy
CNNfeatures{number of channel used}{basin}.{domain size}{month}fixed.npy
============

Step 5: Execute /kfold/TC-Split_KFold.py to generate separate training and testing datasets, configuring size and details within the script. Ensure proper file naming for manual setup in model building. Input: Features and labels files; Output: Training and testing sets in .npy format.

Step 6: Train models with prepared datasets using retrieval scripts. Refer to the script's last section for adjustments to model or data paths.

Step 7 (optional): To evaluate model performance on a test set, execute the script TC-test_set.py and follow the provided instructions. Note that all test sets are named according to the convention test{number_of_channel}x/y.{domain_size}.npy.

================================================================
The end.
================================================================

This project is funded by the United States National Science Foundation (NSF), and therefore, any dissemination of information or data must be in accordance with NSF guidelines and regulations.

It is executed by Minh Khanh Luong from the Department of Earth and Atmospheric Science at Indiana University Bloomington, under the advisement of Assoc. Prof. Chanh Kieu, also from the same department. For further details, please contact kmluong@iu.edu or mk223338@gmail.com.

