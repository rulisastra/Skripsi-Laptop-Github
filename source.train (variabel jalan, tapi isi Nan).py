import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from numpy.linalg import inv

def splitData(data, a, b):
    if((a+b)!=1):
        print("Tidak Valid")
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

def denormalize(normalized, data, scale):
    denormalized = []
    for i in range(len(normalized)):
        a = ((normalized[i]-min(scale))*(max(data)-min(data)))/(max(scale)-min(scale))+min(data)
        denormalized.append(a)
    return np.array(denormalized)

def createDataset(data, window):
    dataX, dataY = [], []
    for i in range(len(data)-window):
        a = []
        for j in range(i, i+window):
            a.append(data[j,0])
        dataX.append(a)
        dataY.append(data[i+window,0])
    return np.array(dataX), np.reshape(np.array(dataY),(-1,1))

def tanh(x):
    return (1-np.exp(-2*x))/(1+np.exp(-2*x))

def dtanh(x):
    return 1 - tanh(x)**2

def mse(x,y):
    mse = []
    for i in range(len(y)):
        a = (x[i]-y[i])**2
        mse.append(a)
    mse = float((sum(mse)/len(y)))
    return mse

def mae(x,y):
    mae = []
    for i in range(len(y)):
        a = abs(y[i]-x[i])
        mae.append(a)
    mae = float(sum(mae)/len(y))
    return mae

def rmse(x,y):
    rmse = []
    for i in range(len(y)):
        a = (x[i]-y[i])**2
        rmse.append(a)
    rmse = float((sum(rmse)/len(y))**0.5)
    return rmse

def dstat(x,y):
    dstat = 0
    n = len(y)
    for i in range(n-1):
        if(((x[i+1]-y[i])*(y[i+1]-y[i]))>0):
            dstat += 1
    Dstat = (1/float(n-2))*float(dstat)*100
    return float(Dstat)

#prepare data
data = pd.read_csv('data.csv', usecols=[1], engine='python', delimiter=';', decimal=",", header=None, names=['date','value'])
data['value'] = data['value'].values
data['value'] = data['value'].astype('float32')
data_raw = normalize(data['value'],(-1,1))

train_data, test_data = splitData(data_raw, 0.7, 0.3)
train_data = train_data.reshape(-1,1)
test_data = test_data.reshape(-1,1)

#bikin data input sesuai window
window = 3
trainX, trainY = createDataset(train_data, window)
testX, testY = createDataset(test_data, window)

#initialize neuron size
batch_dim = trainX.shape[0]
input_dim = window
hidden_dim = 6
output_dim = 1

np.random.seed(4)

#initialize weight
synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1
synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1
synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

#log mse tiap epoch
mse_all = []

#inisialisasi sebelum train
jumlah_w = (input_dim*hidden_dim)+(hidden_dim*hidden_dim)+(hidden_dim*output_dim)
Q = 0.01*np.identity(jumlah_w)
R = 1*np.identity(output_dim)
P = 1*np.identity(jumlah_w)

context_layer = np.full((batch_dim,hidden_dim),0)
layer_h_deltas = np.zeros(hidden_dim)

epoch = 1270
start_time = time.time()
for i in range(epoch):
    index = 0
    layer_2_value = []
    context_layer = np.full((batch_dim,hidden_dim),0)
    layer_h_deltas = np.zeros(hidden_dim)
    while(index+batch_dim<=trainX.shape[0]):
        X = trainX[index:index+batch_dim,:]
        Y = trainY[index:index+batch_dim,:]
        index = index+batch_dim
    
        #input to hidden
        layer_1 = tanh(np.dot(X,synapse_0)+np.dot(context_layer,synapse_h))
    
        #hidden to output
        layer_2 = tanh(np.dot(layer_1,synapse_1))
        layer_2_value.append(layer_2)
    
        #hitung error output
        layer_2_error = layer_2 - Y
    
        #layer 2 deltas
        layer_2_delta = layer_2_error*dtanh(layer_2)
    
        #layer 1 delta
        layer_1_delta = (np.dot(layer_h_deltas,synapse_h.T) + np.dot(layer_2_delta,synapse_1.T)) * dtanh(layer_1)
    
        #calculate weight update
        synapse_1_update = np.dot(np.atleast_2d(layer_1).T,(layer_2_delta))
        synapse_h_update = np.dot(np.atleast_2d(context_layer).T,(layer_1_delta))
        synapse_0_update = np.dot(X.T,(layer_1_delta))
        
        #concatenate weight
        synapse_0_c = np.reshape(synapse_0,(-1,1))
        synapse_h_c = np.reshape(synapse_h,(-1,1))
        synapse_1_c = np.reshape(synapse_1,(-1,1))
        w_concat = np.concatenate((synapse_0_c,synapse_h_c,synapse_1_c), axis=0)
        
        #jacobian
        dsynapse_0 = np.reshape(synapse_0_update,(1,-1))
        dsynapse_h = np.reshape(synapse_h_update,(1,-1))
        dsynapse_1 = np.reshape(synapse_1_update,(1,-1))
        H = np.concatenate((dsynapse_0,dsynapse_h,dsynapse_1), axis=1)
        H_transpose = H.T
        
        #Kalman Gain
        K1 = np.dot(H,P)
        K2 = np.dot(K1,H_transpose)+R
        K3 = inv(K2)
        K4 = np.dot(P,H_transpose)
        K = np.dot(K4,K3)
        
        #update weight
        innovation = ((Y-layer_2).sum()/len(layer_2_error))
        w_concat_new = w_concat + np.dot(K,innovation)
        
        #update P
        P1 = np.dot(K,H)
        P2 = np.dot(P1,P)+Q
        P = P-P2
        
        #assign bobot
        synapse_0 = w_concat_new[0:(input_dim*hidden_dim),0]
        synapse_h = w_concat_new[(input_dim*hidden_dim):(input_dim*hidden_dim)+(hidden_dim*hidden_dim),0]
        synapse_1 = w_concat_new[(input_dim*hidden_dim)+(hidden_dim*hidden_dim):w_concat_new.shape[0],0]
        
        #reshape balik bobot
        synapse_0 = np.reshape(synapse_0,(input_dim,hidden_dim))
        synapse_h = np.reshape(synapse_h,(hidden_dim,hidden_dim))
        synapse_1 = np.reshape(synapse_1,(hidden_dim,output_dim))
    
        #reset update
        synapse_0_update *= 0
        synapse_1_update *= 0
        synapse_h_update *= 0
    
#        update context layer
        layer_h_deltas = layer_1_delta
        context_layer = layer_1
    
    layer_2_value = np.reshape(layer_2_value,(-1,1))
    mse_epoch = mse(trainY,layer_2_value)
    mse_all.append(mse_epoch)
run_time = time.time() - start_time
#%%
plt.plot(mse_all,label='loss')
plt.title('Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()

#%%

#coba predict

batch_predict = testX.shape[0]
context_layer_p = np.full((batch_predict,hidden_dim),0)
y_pred = []
index = 0
while(index+batch_predict<=testX.shape[0]):
    X = testX[index:index+batch_predict,:]
    layer_1p = tanh(np.dot(X,synapse_0)+np.dot(context_layer_p,synapse_h))
    layer_2p = tanh(np.dot(layer_1p,synapse_1))
    y_pred.append(layer_2p)
    context_layer_p = layer_1p
    index = index+batch_predict
    
y_pred = denormalize(np.reshape(y_pred,(-1,1)), data['value'], (-1,1))
testY = denormalize(testY, data['value'], (-1,1))
mse_pred = mse(testY,y_pred)
rmse_pred = rmse(testY,y_pred)
mae_pred = mae(testY,y_pred)
dstat_pred = dstat(testY,y_pred)
scoring = [mse_pred,rmse_pred,mae_pred,dstat_pred, run_time]

plt.plot(testY, label='true')
plt.plot(y_pred, label='prediction')
plt.title('RNN-EKF Elman')
plt.legend()
plt.show()
print(scoring)

#%%
plt.plot(testY[0:50], label='true')
plt.plot(y_pred[0:50], label='prediction')
plt.title('RNN-EKF Elman')
plt.legend()
plt.show()

#%%
np.savetxt('bobot_input.csv', synapse_0, delimiter=',')
np.savetxt('bobot_hidden.csv', synapse_h, delimiter=',')
np.savetxt('bobot_output.csv', synapse_1, delimiter=',')
np.savetxt('loss_ekf.csv', mse_all, delimiter=';')
