import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.linalg import inv

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
hidden_dim = 2
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
Q = 0.1*np.identity(jumlah_w) #kovarian Noise process
R = 1*np.identity(output_dim) #Kovarian Noise measurement(observasi)
P = 1*np.identity(jumlah_w) #kovarian estimasi vektor state
# V = np.zeros((windowSize), dtype=int)

# =============================================================================
# #%% ----------------------------------------------------------
# #UNSCENTED KALMAN FILTER (ALL)
# 
# #inisialisasi faktor skalar
# #m = np.sum(trainY)
# #L = np.sum(trainX)
# 
# #inisialisasi PARAMETER -> van der merwe suggest using beta =2 kappa = 3-n
# beta = 2
# alpha = np.random.random(1)
# n = 1 #n = dimensi dari x
# kappa = 3-n #atau 0 oleh dash 2014
# 
# ##WEIGHTS
# lambda_ = alpha**2 * (n + kappa) - n
# Wc = np.full(2*n + 1,  1. / (2*(n + lambda_)))
# Wm = np.full(2*n + 1,  1. / (2*(n + lambda_)))
# Wc[0] = lambda_ / (n + lambda_) + (1. - alpha**2 + beta)
# Wm[0] = lambda_ / (n + lambda_)
# 
# #SIGMA POINTS masih salaah
# #a
# s = np.sqrt(data) #data atau x??? data kok
# np.dot(s, s.T)
# 
# sigmas = np.zeros((2*n+1, n))
# U = np.sqrt((n+lambda_)*P) # sqrt
# 
# def sigmaPoint(data,X,k):
#     sigmas[0] = X
#     for k in range(n):
#         sigmas[k+1]   = X + U[k]
#         sigmas[n+k+1] = X - U[k]
#         x = np.dot(Wm, sigmas) #jumlah sigma mean atau means
#         return np.dot(x,n)
#     
# x = np.dot(Wm, sigmas)
# n = sigmas.shape
# sg = sigmaPoint
# 
# def P_(kmax,n,k):
#     P = np.zeros((n, n))
#     for i in range(kmax):
#         y = sigmas[i] - x
#         P += Wc[i] * np.outer(y, y) 
#         P += Q
#         y = (sigmas[k] - x).reshape(kmax, 1) # convert into 2D array
#         P += Wc[k] * np.dot(y, y.T) #P += Wc[K] * np.dot(y, y.T)
#         return()
# 
# #PREDIKSI (STEP)
# def predict(sigma_points_fn):
#     """ Performs the predict step of the UKF. On return, 
#     self.xp and self.Pp contain the predicted state (xp) 
#     and covariance (Pp). 'p' stands for prediction.
#     """
# 
#     # calculate sigma points for given mean and covariance
#     sigmas = sigma_points_fn(x,P)
# 
#     for i in range(sigmas):
#         sigmas_f[i] = fx(sigmas[i])
# 
#     xp,Pp = (sigmas_f,Wm,Wc,Q).T #transform
#     return(predict)
#     
# predict
# 
# #UPDATE STEP
# def update(z):
#     # rename for readability
#     sigmas_f = sigmas_f
#     sigmas_h = sigmas_h
# 
#     # transform sigma points into measurement space
#     for i in range(self._num_sigmas):
#         sigmas_h[i] = self.hx(sigmas_f[i])
# 
#     # mean and covariance of prediction passed through UT
#     zp, Pz = (sigmas_h, self.Wm, self.Wc, self.R).T #transform
# 
#     # compute cross variance of the state and the measurements
#     Pxz = np.zeros((self._dim_x, self._dim_z))
#     for i in range(self._num_sigmas):
#         Pxz += self.Wc[i] * np.outer(sigmas_f[i] - self.xp,
#                                     sigmas_h[i] - zp)
# 
#     K = np.dot(Pxz, np.linalg.inv(Pz)) # Kalman gain
# 
#     self.x = self.xp + np.dot(K, z-zp)
#     self.P = self.Pp - np.dot(np.dot(K, Pz),np.dot(K.T))
#     return np.array(self.x),np.array(self.P)
# 
# =============================================================================
#%%

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
    mape = float((sum(mape))/len(y))*100
    return mape

def dstat(x,y):
    dstat = 0
    n = len(y)
    for i in range(n-1):
        if(((x[i+1]-y[i])*(y[i+1]-y[i]))>0):
            dstat += 1
    Dstat = (1/float(n-2))*float(dstat)*100
    return float(Dstat)

#%%

epoch = 100
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
mape_pred = mape(testY,y_pred)
mae_pred = mae(testY,y_pred)
dstat_pred = dstat(testY,y_pred)
#scoring = [mse_pred,rmse_pred,mae_pred,dstat_pred,run_time]


plt.plot(testY[0:50], label='true', marker ='o') #testY[0:50] untuk plotting (catatan)
plt.plot(y_pred[0:50], label='prediction', marker ='x')
plt.title('RNN-UKF')
plt.legend()
plt.show()

print("runtime : ", run_time)
print("mse : " , mse_pred)
print("rmse : ", rmse_pred) 
print("mape : ", mape_pred) 
print("mae: " , mae_pred)
print("dstat : " , dstat_pred)

#%%
np.savetxt('bobot_input.csv', synapse_0, delimiter=',')
np.savetxt('bobot_hidden.csv', synapse_h, delimiter=',')
np.savetxt('bobot_output.csv', synapse_1, delimiter=',')
np.savetxt('loss_ukf.csv', mse_all, delimiter=';')

#%%

