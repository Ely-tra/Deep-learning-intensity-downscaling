import cv2
import tensorflow as tf
import os
from tqdm import tqdm
import netCDF4
import numpy as np
import libtcg_netcdfloader as tcg_loader
import libtcg_utils as tcg_utils
import pickle
import sys
from tcg_VIT_p2_2 import Patches, PatchEncoder
#
# Visualize the output of the training model (work for jupyter notebook only)
#
def view_history(histories):
    import matplotlib.pyplot as plt
    val_accuracy1 = histories[0]['val_binary_accuracy']
    accuracy1 = histories[0]['binary_accuracy']
    # val_accuracy2 = histories[1]['val_binary_accuracy']
    # accuracy2 = histories[1]['binary_accuracy']
    # val_accuracy3 = histories[2]['val_binary_accuracy']
    # accuracy3 = histories[2]['binary_accuracy']
    epochs = np.arange(len(val_accuracy1))
    plt.plot(epochs,val_accuracy1,'r',label="val binary_accuracy Vision Transformer")
    plt.plot(epochs,accuracy1,'r--',label="train binary_accuracy Vision Transformer")
    # plt.plot(epochs,val_accuracy2,'b',label="val binary_accuracy resnet22")
    # plt.plot(epochs,accuracy2,'b--',label="train binary_accuracy resnet22")
    # plt.plot(epochs,val_accuracy3,'g',label="val binary_accuracy resnet40")
    # plt.plot(epochs,accuracy3,'g--',label="train binary_accuracy resnet40")
    # plt.legend()

    plt.figure()
    val_loss1 = histories[0]['val_loss']
    loss1 = histories[0]['loss']
    # val_loss2 = histories[1]['val_loss']
    # loss2 = histories[1]['loss']
    # val_loss3 = histories[2]['val_loss']
    # loss3 = histories[2]['loss']
    plt.plot(epochs,val_loss1,'r',label="val loss resnet20")
    plt.plot(epochs,loss1,'r--',label="train loss resnet20")
    # plt.plot(epochs,val_loss2,'b',label="val loss resnet22")
    # plt.plot(epochs,loss2,'b--',label="train loss resnet22")
    # plt.plot(epochs,val_loss3,'g',label="val loss resnet40")
    # plt.plot(epochs,loss3,'g--',label="train loss resnet40")
    plt.legend()
    plt.show()
#
# loop thru all best-saved ResNet-trained models and make a prediction. Note that prediction is applied one by one instead 
# of a batch input. 
#


def main(DATADIR="",bestmodels=[],test_sample=10):
    #
    # Test data with an input structure of pos/neg dirs
    #
    CATEGORIES = ["neg", "pos"]
    F1_performance = []
    for bestmodel in bestmodels:
        print("Best model aaaa", bestmodel)
        model = tf.keras.models.load_model(bestmodel, custom_objects={'Patches': Patches, 'PatchEncoder': PatchEncoder})
        prediction_total = 0
        prediction_yes = 0
        prediction_history = []
        truth_history = []
        for category in CATEGORIES:
            path = os.path.join(DATADIR,category)
            for img in tqdm(os.listdir(path)):
                print("img aaaa", img)
                try:
                    img_dir = DATADIR + '/' + category + '/' + img
                    print('Processing image:', img_dir)
                    indata, _ = tcg_loader.prepare12channels(img_dir,IMG_SIZE=32)
                    prediction = model.predict([indata])
                    print("TC formation prediction is",prediction,round(prediction[0][0]),CATEGORIES[round(prediction[0][0])])
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
        # Compute F1 score for each best model and save it
        #
        print("len truth history", len(truth_history))
        print("len prediction history", len(prediction_history))
        F1_performance.append([bestmodel,tcg_utils.F1_score(truth_history,prediction_history,1,0.5)])
    return F1_performance

if __name__ == '__main__':
    n = len(sys.argv)
    print("Total arguments input are:", n)
    print("Name of Python script:", sys.argv[0])
    if n < 3:
       print("Need a forecast lead time and datapath to process...Stop")
       print("+ Example: tcg_ResNet_p3.py 00 /N/project/hurricane-deep-learning/data/ncep_extracted_binary_30x30/EP/")
       exit()
    leadtime = str(sys.argv[1])
    datadir = str(sys.argv[2]) + "/" + leadtime + "/"
    print("Forecast lead time to run is: ",leadtime)
    print("Datapath is: ",datadir)

    history_check = "no"
    if history_check == "yes":
        hist = open("tcg_histories_vit.pickle","rb")
        histories = pickle.load(hist)
        view_history(histories)

    basemodels = ["tcg_VIT.model_ZZ.keras"]
    bestmodels = []
    for model in basemodels:
        a=model.replace('ZZ', leadtime)
        bestmodels.append(a)
    print(bestmodels)

    performance = main(DATADIR=datadir,bestmodels=bestmodels,test_sample=300)
    print("Summary of the ResNet model performance for forecast lead time: ",leadtime)
    for i in range(len(bestmodels)):
        print("Model:",bestmodels[i]," --- F1, Recall, Presision:",np.round(performance[i][1],2))

