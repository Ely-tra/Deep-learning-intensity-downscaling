#
# NOTE: This machine learning script is for validating TCG prediction, using
#       input dataset in the NETCDF format. The program treats different
#       2D input fields as different channels of an image. This specific
#       program requires a set of 12 2D-variables (12-channel image) and
#       consists of three stages
#       - Stage 1: reading NETCDF input and generating (X,y) data with a
#                  given image sizes, which are then saved by pickle;
#       - Stage 2: import the saved pickle (X,y) pair and build a CNN model
#                  with a given training/validation ratio, and then save
#                  the train model under tcg_CNN.model.
#       - Stage 3: Validation of the model by using the trained model from 
#                  Stage 2, and a list pos/neg input.
#
# INPUT: This Stage 3 script reads in the CNN trained model "tcg_CNN.model"
#        that is generated from Step 2.
#
#        Remarks: Note that the input data for this script must be on the
#        same as in Step 1 with standard 19 vertical
#        levels 1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600,
#        550, 500, 450, 400, 350, 300, 250, 200. Also, all field vars must
#        be resize to cover an area of 30x30 around the TC center for the
#        positive data cases.
#        Similar to Step 2, this Step 3 needs to also have a large mem
#        allocation so that it can be run properly.
#
# OUTPUT: A list of probability forecast with the same dimension as the
#        number of input 12-channel images.
#
# HIST: - 01, Nov 22: Created by CK
#       - 02, Nov 22: Modified to optimize it
#       - 05, Jun 23: Rechecked and added F1 score function for a list of models
#       - 10, Oct 23: Re-worked on the code structure for better flow and future
#                     upgrades with different channels
#
# AUTH: Chanh Kieu (Indiana University, Bloomington. Email: ckieu@iu.edu)
#
#==========================================================================
import cv2
import tensorflow as tf
import os
import sys
from tqdm import tqdm
import netCDF4
import numpy as np
import libtcg_netcdfloader as tcg_loader
import libtcg_utils as tcg_utils
#
# main fucntion that loops thru all best-saved CNN trained models and make 
# a prediction. Note that prediction is applied one by one instead of a batch input. 
#
def main(datadir,bestmodels,test_sample=1000):
    #
    # Test data with an input structure of pos/neg dirs
    #
    categories = ["neg", "pos"]
    F1_performance = []
    for bestmodel in bestmodels:
        model = tf.keras.models.load_model(bestmodel)
        prediction_total = 0
        prediction_yes = 0
        prediction_history = []
        truth_history = []
        for category in categories:
            path = os.path.join(datadir,category)
            for img in tqdm(os.listdir(path)):    
                try:
                    img_dir = datadir + '/' + category + '/' + img
                    print('Processing image:', img_dir)
                    #print('Input image dimension is: ',prepare(img_dir).shape)
                    indata, _ = tcg_loader.prepare12channels(img_dir,IMG_SIZE=30)
                    prediction = model.predict([indata])
                    print("TC formation prediction is",prediction,round(prediction[0][0]),categories[round(prediction[0][0])])
                    prediction_history.append(prediction[0][0])
                    if round(prediction[0][0]) == 1:
                        prediction_yes += 1
                    if category == "pos":
                        truth_history.append(1)
                    else:
                        truth_history.append(0)
                    prediction_total += 1    
                    if prediction_total > test_sample:
                        prediction_total = 0
                        break
                except Exception as e:
                    pass   
        #
        # Compute F1 score for each best model now
        #
        #print(prediction_history)
        F1_performance.append([bestmodel,tcg_utils.F1_score(truth_history,prediction_history,1,0.5)]) 
    return F1_performance

if __name__ == '__main__':
    n = len(sys.argv)
    print("Total arguments input are:", n)
    print("Name of Python script:", sys.argv[0])
    if n < 3:
       print("Need a forecast lead time and datapath to process...Stop")
       print("+ Example: tcg_CNN_p2.py 00 /N/project/hurricane-deep-learning/data/ncep_extracted_binary_30x30/ncep_EP_binary_0h/")
       exit()
    leadtime = str(sys.argv[1])
    datadir = str(sys.argv[2])
    print("Forecast lead time to run is: ",leadtime)
    print("Datapath is: ",datadir)
    #sys.exit()

    #datadir = "/N/project/hurricane-deep-learning/data/ncep_extracted_binary_30x30/ncep_EP_binary_0h/"
    #bestmodels = ["3-conv-32-layer-0-dense.model_00h","3-conv-32-layer-1-dense.model_00h","3-conv-32-layer-2-dense.model_00h",
    #              "5-conv-32-layer-0-dense.model_00h","5-conv-32-layer-1-dense.model_00h","5-conv-32-layer-2-dense.model_00h"]
    basemodels = ["3-conv-32-layer-0-dense.model_ZZh","3-conv-32-layer-1-dense.model_ZZh","3-conv-32-layer-2-dense.model_ZZh",
                  "5-conv-32-layer-0-dense.model_ZZh","5-conv-32-layer-1-dense.model_ZZh","5-conv-32-layer-2-dense.model_ZZh"]
    bestmodels = []
    for model in basemodels:
        a=model.replace('ZZ', leadtime)
        bestmodels.append(a)
    print(bestmodels)

    performance = main(datadir,bestmodels,1000)
    print("========================================")
    print("Summary of the CNN model performance for forecast lead time: ",leadtime)
    for i in range(len(bestmodels)):
        print("Model:",performance[i][0]," --- F1, Recall, Presision:",np.round(performance[i][1],2))
