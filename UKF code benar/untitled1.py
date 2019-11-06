import pandas as pd
#from pykalman import UnscentedKalmanFilter as ukF
import numpy as np
import matplotlib.pyplot as plt
#import time
#from numpy.linalg import inv
from filterpy.kalman import UnscentedKalmanFilter as ukf
from filterpy.kalman import MerweScaledSigmaPoints #,unscented_transform
import scipy 

#%%

#persiapan data
data = pd.read_csv('data.csv',
                    usecols=[1],
                    engine='python',
                    delimiter=',',
                    decimal=".",
                    thousands=',',
                    header=None,
                    names=['date','value'] )
data['value'] = data['value'].values
data['value'] = data['value'].astype('float32')

def pisahData(data,a,b):
     if((a+b)!=1):
            print("pemisahan tidak valid")
     else:
        train = []
        test = []
        train_size = int(len(data)*a)
        train = data[0:train_size-1]
        test = data[train_size-1:len(data)]
     return np.array(train),np.array(test)

def normalize(data, scale):
    normalized = []
    for i in range(len(data)):
        a = (min(scale))+(data[i]-min(data))*(max(scale)-min(scale))/(max(data)-min(data))
        normalized.append(a)
    return np.array(normalized)
    scale = (-1,1)
    normalized = normalize(data,scale)

data_raw = normalize(data['value'],(-1,1))

train_data, test_data = pisahData(data_raw, 0.7, 0.3)
train_data = train_data.reshape(-1,1) #reshape data dengan range -1,1 -> satu kolom kebawah
test_data = test_data.reshape(-1,1)

zs = train_data
#Xs, Ps = ukf.batch_filter(zs=train_data)

plt.plot(xs);
plt.plot(zs, marker='x', ls ='')