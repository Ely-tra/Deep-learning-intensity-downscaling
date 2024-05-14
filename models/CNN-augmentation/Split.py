import numpy as np
from sklearn.utils import shuffle
def split_data(datax, datay, percentage=5):
    datax, datay = shuffle(datax, datay, random_state=0)
    testx , testy  = datax[:int(len(datax)/(100/percentage))], datay[:int(len(datax)/(100/percentage))]
    trainx, trainy = datax[int(len(datax)/(100/percentage)):], datay[int(len(datax)/(100/percentage)):]
    return trainx, trainy, testx, testy
root='/N/slate/kmluong/Training_data/'
datax=np.load(root+'CNNfeatures13.25x25fixed.npy')
datay=np.load(root+'CNNlabels13.25x25.npy')
trainx, trainy, testx, testy=split_data(datax,datay, percentage=10)
np.save(root+'Split/data/train13x.25x25.npy', trainx)
np.save(root+'Split/data/test13x.25x25.npy', testx)
np.save(root+'Split/data/train13y.25x25.npy', trainy)
np.save(root+'Split/data/test13y.25x25.npy', testy)


