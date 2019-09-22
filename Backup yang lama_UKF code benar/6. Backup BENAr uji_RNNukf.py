import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    Dstat = (1/float(n-1))*float(dstat)*100
    return float(Dstat)

data_coba = pd.read_csv('data_coba_1bulan.csv',usecols=[1],
                        engine='python',
                        delimiter=',',
                        decimal=".",
                        thousands=',',
                        header=None,
                        names=['date','value'] )
data_coba['value'] = data_coba['value'].values
data_coba['value'] = data_coba['value'].astype('float32')
coba_data_raw = data_coba['value'].values.reshape(-1,1)
coba_data = normalize(coba_data_raw,(-1,1))
cobaX, cobaY = createDataset(coba_data, 3)

synapse_0 = np.loadtxt('bobot_input.csv', delimiter=',')
synapse_h = np.loadtxt('bobot_hidden.csv', delimiter=',')
synapse_1 = np.loadtxt('bobot_output.csv', delimiter=',')

batch_predict = cobaX.shape[0]
context_layer_p = np.full((batch_predict,6),0)
y_pred = []
index = 0
while(index+batch_predict<=cobaX.shape[0]):
    X = cobaX[index:index+batch_predict,:]
    layer_1p = tanh(np.dot(X,synapse_0)+np.dot(context_layer_p,synapse_h))
    layer_2p = tanh(np.dot(layer_1p,synapse_1))
    y_pred.append(layer_2p)
    context_layer_p = layer_1p
    index = index+batch_predict
    
y_pred = denormalize(np.reshape(y_pred, (-1,1)), coba_data_raw, (-1,1))
cobaY = denormalize(cobaY, coba_data_raw, (-1,1))
mse_pred = mse(cobaY,y_pred)
rmse_pred = rmse(cobaY,y_pred)
mae_pred = mae(cobaY,y_pred)
dstat_pred = dstat(cobaY, y_pred)
scoring = [mse_pred,rmse_pred,mae_pred,dstat_pred]

plt.plot(cobaY, label='true')
plt.plot(y_pred, label='prediction')
plt.title('RNN-EKF Elman')
plt.legend()
plt.show()
print(scoring)