import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.linalg import inv
#from filterpy.kalman import UnscentedKalmanFilter as ukf_
#from pykalman import UnscentedKalmanFilter

#persiapan data
data = pd.read_csv('BTC_USD_2018-04-06_2019-09-23-CoinDesk.csv',
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

def denormalize(normalized, data, scale):
    denormalized = []
    for i in range(len(normalized)):
        a = ((normalized[i]-min(scale))*(max(data)-min(data)))/(max(scale)-min(scale))+min(data)
        denormalized.append(a)
    return np.array(denormalized)


data_raw = normalize(data['value'],(-1,1))

train_data, test_data = pisahData(data_raw, 0.7, 0.3)
train_data = train_data.reshape(-1,1) #reshape data dengan range -1,1 -> satu kolom kebawah
test_data = test_data.reshape(-1,1)

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

def mape(x,y):
    mape = []
    for i in range(len(y)):
        a = abs((x[i]-y[i])/x[i])
        mape.append(a)
    mape = float((sum(mse))/len(y))*100
    return mape

def dstat(x,y):
    dstat = 0
    n = len(y)
    for i in range(n-1):
        if(((x[i+1]-y[i])*(y[i+1]-y[i]))>0):
            dstat += 1
    Dstat = (1/float(n-2))*float(dstat)*100
    return float(Dstat)

#createwindowSize =3 untuk input
def createDataset(data, windowSize):
    dataX, dataY = [],[]
    for i in range(len(data)-windowSize):
        a = []
        for j in range(i, i+windowSize):
            a.append(data[j,0])
        dataX.append(a)
        dataY.append(data[i+windowSize,0])
    return np.array(dataX), np.array(dataY)
windowSize = 3
trainX, trainY = createDataset(train_data,windowSize)
testX, testY = createDataset(test_data, windowSize)

#initialize neuron size
batch_dim = trainX.shape[0] #mengambil banyak baris (n) dari trainX(n,m)
input_dim = windowSize
hidden_dim = 6
output_dim = 1

np.random.seed(4) #random tetap walaupun kamu mengiterasi beruulang kali

#fungsi aktivasi dan turunannya
def tanh(x):
    return (1-np.exp(-2*x))/(1+np.exp(-2*x))

def dtanh(x):
    return (1-tanh(x)**2)

#inisialisasi random BOBOOOTT awal JST dengan random.random ->> interval [0,1]
synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1 #inisialisasi
synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1 #bobotjaringan dg
synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1 #interval[-1,1]

synapse_0_update = np.zeros_like(synapse_0) #meng-0 kan semua isi array sesuai shape dr variabel (synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

#log mse tiap epoch
mse_all = []

#inisialisasi sebelum train
jumlah_w = (input_dim*hidden_dim)+(hidden_dim*hidden_dim)+(hidden_dim*output_dim)
Q = 1*np.identity(jumlah_w) #kovarian Noise process
R = 1*np.identity(output_dim) #Kovarian Noise measurement(observasi)
P = 1*np.identity(jumlah_w) #kovarian estimasi vektor state

#%% UKF

alpha=1e-1
beta=2
kappa=0
lambda_ = alpha**2 * (data + kappa) - data


def total_sigmas(data): #semua n diganti (data)
        return 2*data + 1 #Number of sigma points for each variable in the state x

def sigma_points(x, data, P, sqrt_method=None, subtract=None):
        if trainX != np.size(x):
            raise ValueError("expected size(x) {}, but size is {}".format(
                trainX, np.size(x)))

        data = X

        if np.isscalar(x):
            x = np.asarray([x])

        if  np.isscalar(P):
            P = np.eye(data)*P
        else:
            P = np.atleast_2d(P)

        lambda_ = alpha**2 * (data + kappa) - data
        U = np.sqrt((lambda_ + data)*P)
        c = .5 / (data + lambda_) #sama dengan Wc = Wm
        Wc = np.full(2*data + 1, c)
        Wm = np.full(2*data + 1, c)
        Wc[0] = lambda_ / (data + lambda_) + (1 - alpha**2 + beta)
        Wm[0] = lambda_ / (data + lambda_)
        sigmas = np.zeros((2*data+1, data))
        sigmas[0] = x
        
        for k in range(data):
            # pylint: disable=bad-whitespace
            sigmas[k+1]   = subtract(x, -U[k])
            sigmas[data+k+1] = subtract(x, U[k])

        return sigmas

def state_mean(sigmas, Wm):
    x = np.zeros(3)
    sum_sin, sum_cos = 0., 0.
    
    for i in range(len(sigmas)):
        s = sigmas[i]
        x[0] += s[0] * Wm[i]
        x[1] += s[1] * Wm[i]
        sum_sin += np.sin(s[2])*Wm[i]
        sum_cos += np.cos(s[2])*Wm[i]
        x[2] = np.tan(sum_sin, sum_cos) #atan2
        return x
   
#%%

epoch = 5
start_time = time.time()
for i in range(epoch):
    index = 0
    layer_2_value = []
    context_layer = np.full((batch_dim,hidden_dim),0)
    layer_h_deltas = np.zeros(hidden_dim)
    while(index+batch_dim<=trainX.shape[0]):
        X = trainX[index:index+batch_dim,:]
        Y = trainY[index:index+batch_dim]
        index = index+batch_dim

        # forward pass -> input to hidden
        layer_1 = tanh(np.dot(X,synapse_0)+np.dot(context_layer,synapse_h))
    
        #hidden to output
        layer_2 = tanh(np.dot(layer_1,synapse_1))
        layer_2_value.append(layer_2)
    
        #hitung error output
        layer_2_error = layer_2 - Y[:,None] #problemnya, y diganti dr Y matrix
    
        #layer 2 deltas (masuk ke context layer dari hidden layer)
        layer_2_delta = layer_2_error*dtanh(layer_2)
    
        #layer 1 delta (masuk ke hidden layer dari context layer)
        layer_1_delta = (np.dot(layer_h_deltas,synapse_h.T) + np.dot(layer_2_delta,synapse_1.T)) * dtanh(layer_1)
    
        #calculate weight update
        synapse_1_update = np.dot(np.atleast_2d(layer_1).T,(layer_2_delta))
        synapse_h_update = np.dot(np.atleast_2d(context_layer).T,(layer_1_delta))
        synapse_0_update = np.dot(X.T,(layer_1_delta))
        
        #concatenate weight
        synapse_0_c = np.reshape(synapse_0,(-1,1))
        synapse_h_c = np.reshape(synapse_h,(-1,1))
        synapse_1_c = np.reshape(synapse_1,(-1,1))
        w_concat = np.concatenate((synapse_0_c,synapse_h_c,synapse_1_c), axis=0) #rentetan bobot
        
        #jacobian
        dsynapse_0 = np.reshape(synapse_0_update,(1,-1))
        dsynapse_h = np.reshape(synapse_h_update,(1,-1))
        dsynapse_1 = np.reshape(synapse_1_update,(1,-1))
        H = np.concatenate((dsynapse_0,dsynapse_h,dsynapse_1), axis=1) # T_ sama dengan H di EKF
        H_transpose = H.T
        
        #inisialisasi UKF
        Myu = np.mean(X)
        stdDev = np.std(X)
        Q = np.dot((stdDev**2),np.eye(jumlah_w))
        R = stdDev**2
        P = np.eye(jumlah_w)
        
        #Kalman Gain
        K = np.zeros((batch_dim, output_dim))    # Kalman gain
        y = np.zeros((output_dim))           # residual
        z = np.array([[None]*output_dim]).T  # measurement
        S = np.zeros((output_dim, output_dim))    # system uncertainty
        SI = np.zeros((output_dim, output_dim))   # inverse system uncertainty
#       Zet = ukf.sigma(P)# transformed sigma points i measurements space
        
        c = .5 / (data + lambda_) #sama dengan Wc = Wm

        K1 = np.dot(H,P)
        K2 = np.dot(K1,H_transpose)+R
        K3 = inv(K2)
        K4 = np.dot(P,H_transpose) #jangan diubah karena sama
        
        K0 = np.substract(sigmas,Myu)
        K1 = np.sum(Wm)
        K2 = np.subtract(H,z)
        K2_t = np.transpose(K2)
        K3 = np.sum()
        K4_t = np.sum(np.dot(c,K4)) # T kapital
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
    
        #update context layer
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

#%%  mari kita coba prediksiiiiiii

batch_predict = testX.shape[0] # mengambil banyaknya baris (n) dari testX(n,m)
context_layer_p = np.full((batch_predict,hidden_dim),0) # return full array yg isinya (0) sebesar dimensi [batch_predict x hidden_dim]
y_pred = [] # hasil output y prediksi
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
scoring = [mse_pred,rmse_pred,mae_pred,dstat_pred,run_time]

plt.plot(testY, label='true') #testY[0:50] buat plotting coba liat di catatan
plt.plot(y_pred, label='prediction')
plt.title('RNN-UKF')
plt.legend()
plt.show()

print(scoring)

#%%
plt.subplot(121)
plt.scatter(data[:X], range(X), alpha=.2, s=1)
plt.title('Input')
plt.subplot(122)
plt.title('Output')
plt.scatter(w_concat_new(data[:X]), range(X), alpha=.2, s=1);


#%%
np.savetxt('bobot_input.csv', synapse_0, delimiter=',')
np.savetxt('bobot_hidden.csv', synapse_h, delimiter=',')
np.savetxt('bobot_output.csv', synapse_1, delimiter=',')
np.savetxt('loss_ukf.csv', mse_all, delimiter=';')

