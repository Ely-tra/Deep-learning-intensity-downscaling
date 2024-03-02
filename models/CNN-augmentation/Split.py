import numpy as np
from sklearn.utils import shuffle
def split_data(datax, datay, percentage):
    datax, datay = shuffle(datax, datay, random_state=0)
    testx , testy  = datax[:int(len(datax)/(100/percentage))], datay[:int(len(datax)/(100/percentage))]
    trainx, trainy = datax[int(len(datax)/(100/percentage)):], datay[int(len(datax)/(100/percentage)):]
    return trainx, trainy, testx, testy
root='/N/slate/kmluong/Training_data/'
datax=np.load(root+'CNNfeatures9ALfixed.npy')
datay=np.load(root+'CNNlabels9AL.npy')
trainx, trainy, testx, testy=split_data(datax,datay, percentage=10)
np.save(root+'Split/data/train9x.npy', trainx)
np.save(root+'Split/data/test9x.npy', testx)
np.save(root+'Split/data/train9y.npy', trainy)
np.save(root+'Split/data/test9y.npy', testy)


