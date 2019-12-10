import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.linalg import cholesky
#%% persiapan data

# IDRUSD.csv = 64%
# data baru  mulai 27sep2017.csv = 0.0
# data baru all = 21%
# data = 56%
# Ether mulai 1jul2017 = 52%
# Ether = 55%
# Litecoin mulai 1jan2017.csv =     55%
# NEO jul2017 = 62%
# NEO all = 63

data = pd.read_csv('IDRUSD.csv',
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

#fungsi aktivasi dan turunannya


def sigmoid(x): #52% dengan IDRUSD
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def dsigmoid(output):
    return output*(1-output)

# =============================================================================
# def tanh(x): # 54% dengan IDRUSD
#     return (1-np.exp(-2*x))/(1+np.exp(-2*x))
# 
# def dtanh(x):
#     return (1-tanh(x)**2)
# =============================================================================

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
data_raw = np.reshape(data_raw,(-1,1))

# pembagian data latih dan test
train_data, test_data = pisahData(data_raw, 0.7, 0.3)
train_data = train_data.reshape(-1,1) #satu kolom kebawah
test_data = test_data.reshape(-1,1)

plt.plot(data_raw, c='b', label='test', ls=None)
plt.plot(train_data[0:1366], c='r', label='training', ls=None)
plt.title('DATA ternormalisasi 7:3')
plt.legend()
plt.show()

# createwindowSize =3 untuk input
def createDataset(data, windowSize):
    dataX, dataY = [],[]
    for i in range(len(data)-windowSize):
        a = []
        for j in range(i, i+windowSize):
            a.append(data[j,0])
        dataX.append(a)
        dataY.append(data[i+windowSize,0])
    return np.array(dataX), np.array(dataY)
windowSize = 5 # 5
trainX, trainY = createDataset(train_data,windowSize)
testX, testY = createDataset(test_data, windowSize)

trainY = np.reshape(trainY,(-1,1))
testY = np.reshape(testY,(-1,1))

#%% PELATIHAN ====  gunain trainX & trainY ====
# INISIALISASI banyaknya neuron setiap layer 
alpha = .1
batch_dim = trainX.shape[0] # ambil jumlah baris (n) dari trainX(n,m)
input_dim = windowSize
hidden_dim = 7 # 7,9,
hidden_dim2 = 7
output_dim = 1

np.random.seed(1) # 1 =72%

# BOBOT === inisialisasi bobot awal (baris,kolom)
synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1 # inisialisasi random bobot awal
synapse_h = 2*np.random.random((hidden_dim,hidden_dim2)) - 1 # dengan interval [-1,1]
synapse_1 = 2*np.random.random((hidden_dim2,output_dim)) - 1 # random.random ->> interval [0,1]

synapse_0_update = np.zeros_like(synapse_0) #meng-0 kan semua isi array sesuai shape dr variabel (synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

# log mse tiap epoch
mse_all = []

# inisialisasi sebelum train
jumlah_w = (input_dim*hidden_dim)+(hidden_dim*hidden_dim2)+(hidden_dim2*output_dim)
Q = 1*np.identity(jumlah_w) #kovarian Noise process
R = 1*np.identity(output_dim) #Kovarian Noise measurement(observasi)
P = 1*np.identity(jumlah_w) #kovarian estimasi vektor state

#%% EVALUASI ====
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

#%% MULAI EPOCH ============ TRAINING ===================

epoch = 100 # 100
start_time = time.time()
for i in range(epoch):
    index = 0
    layer_3_value = []
    layer_h_value = np.zeros(hidden_dim)
    while(index+batch_dim<=trainX.shape[0]):        
        # input dan output
        X = trainX[index:index+batch_dim,:]
        Y = trainY[index:index+batch_dim]
        index = index+batch_dim

        # forwardpass input ~> hidden
        layer_1 = sigmoid(np.dot(X,synapse_0))

        # hidden ~> hidden2
        layer_2 = sigmoid(np.dot(layer_1,synapse_h))

        # hidden2 ~> output
        layer_3 = dsigmoid(np.dot(layer_2,synapse_1))
        layer_3_value.append(layer_3)

        # hitung error output
        layer_3_error = layer_3 - Y[:,None] 

        # error di hidden layer_2
        layer_2_error = layer_3_error*dsigmoid(layer_2)

        # error di hidden layer 
        layer_1_error = (np.dot(layer_h_value,synapse_h.T) + np.dot(layer_3_error,synapse_1.T)) * dsigmoid(layer_1)

        # calculate weight update
        synapse_1_update = np.dot(np.atleast_2d(layer_1).T,(layer_3_error))
        synapse_h_update = np.dot(np.atleast_2d(layer_2).T,(layer_1_error))
        synapse_0_update = np.dot(X.T,(layer_1_error))

        # concatenate weight
        synapse_0_c = np.reshape(synapse_0,(-1,1))
        synapse_h_c = np.reshape(synapse_h,(-1,1))
        synapse_1_c = np.reshape(synapse_1,(-1,1))
        w_concat = np.concatenate((synapse_0_c,synapse_h_c,synapse_1_c), axis=0)
        
        synapse_0_masuk = np.reshape(synapse_0_update,(1,-1)) # satu baris kesamping
        synapse_h_masuk = np.reshape(synapse_h_update,(1,-1))
        synapse_1_masuk = np.reshape(synapse_1_update,(1,-1))
        masuk = np.concatenate((synapse_0_masuk,synapse_h_masuk,synapse_1_masuk), axis=1)
        
        #%% Unscented Kalman Filter without filterpy
        
        X_ = masuk # myu
        beta = 2.
        kappa = 0
        lambda_ = 2
        # lambda_ = alpha**2 * (n + kappa) - n
        n = X_.size # julier versi masalah 'dimension of problem'
        
        #%% SIGMA POINTS around mean
        mean = np.sum(w_concat) / n # mean secara keseluruhan
        U = cholesky((n + lambda_)*P) # sama dg np.sqrt
# =============================================================================
#         implements U'*U = (n+kappa)*P. Returns lower triangular matrix.
#         Take transpose so we can access with U[i]
#         dikali P biar hold the value
# =============================================================================
        # filterpy version dengan shape (121,60) karena dikali dg P!
        sigmas = np.zeros((2*n+1, n))
        sigmas[0] = X_
        # maka....
        for k in range(n):
            sigmas[k+1] = np.subtract(X_, -U[k])
            sigmas[n+k+1] = np.subtract(X_, U[k])
        # gabung (121,60) kebawah.. jadinya 121,60
        
        #%% BOBOT SIGMA dari Merwe
        c_ = .5 / (n + lambda_)
        Wm = np.full(2*n+1, c_)
        Wc = Wm # size (121,) atau (n,)
        Wc[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)
        Wm[0] = lambda_ / (n + lambda_)
        
        # SIGMA Unscented Transform  (biar hold value dg eyeP) ke measurement
        
        Mz = np.dot(Wm,sigmas)
        # Mz = np.sum(np.dot(Wm,sigmas)) # yang bener
        
        # KOVARIAN ke measurement juga
        kmax, n = sigmas.shape
        Pz = np.zeros(n)
        for k in range(kmax):
            c = np.subtract(sigmas[k],Mz)
            Pz = Wc[k] * np.outer(c, c) 
            # makin besar nilainya dengan +=
        Pz += Q
        Pz = Pz.sum()
        # Pz = np.round(Pz,2)
        
        # Kalman gain
        Pxz = np.zeros(sigmas.shape) # Tut
        for k in range(kmax):
            cc = np.subtract(X_,Mz)
            Pxz = Wc[k] * np.outer(cc, c) # makin besar nilainya dengan +=
        # Pxz = np.round(Pxz,2)
        Pxz = Pxz.sum()
        
        # Kalman gain
        # K = np.dot(Pxz,inv(Pz))
        K = Pz * Pxz # harus inv
        
        P = P - np.dot(K,np.dot(Pz,K.T))

        #%%
        innovation =  ((Y-layer_2).sum()/len(layer_2_error))
        # w_concat_new = w_concat + np.dot(K,innovation)
        K_inn = K*innovation
        w_concat_new = w_concat + np.dot(K,innovation)
        
        #assign bobot
        synapse_0 = w_concat_new[0:(input_dim*hidden_dim),0]
        synapse_h = w_concat_new[(input_dim*hidden_dim):(input_dim*hidden_dim)+(hidden_dim*hidden_dim),0]
        synapse_1 = w_concat_new[(input_dim*hidden_dim)+(hidden_dim*hidden_dim):w_concat_new.shape[0],0]
        
        #reshape balik bobot
        synapse_0 = np.reshape(synapse_0,(input_dim,hidden_dim))
        synapse_h = np.reshape(synapse_h,(hidden_dim,hidden_dim2))
        synapse_1 = np.reshape(synapse_1,(hidden_dim2,output_dim))
        
        synapse_0 += synapse_0_update * alpha
        synapse_1 += synapse_1_update * alpha
        synapse_h += synapse_h_update * alpha  

        # reset update
        synapse_0_update *= 0
        synapse_1_update *= 0
        synapse_h_update *= 0

        # update hidden
        layer_h_value = layer_1_error

    # layer_2_value = np.rot90(np.reshape(layer_2_value,(-1,1)),2)
    layer_3_value = np.reshape(layer_3_value,(-1,1))
    mse_epoch = mse(trainY,layer_3_value)
    mse_all.append(mse_epoch)
    
run_time = time.time() - start_time
#%% seberapa besar lossnya???

plt.plot(mse_all,label='loss')
plt.title('Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()

plt.plot(trainY, marker='o', label='true') #testY[0:50] buat plotting coba liat di catatan
plt.plot(layer_3_value, marker='o', label='prediction')
plt.title('MLP')
plt.legend()
plt.show()

#%% mari coba ============ PREDIKSI ===================

batch_predict = testX.shape[0] # mengambil banyaknya baris (n) dari testX(n,m)
y_pred = [] # hasil akhir Y prediksi
index = 0
while(index+batch_predict<=testX.shape[0]):
    X = testX[index:index+batch_predict,:]
    layer_1p = sigmoid(np.dot(X,synapse_0))
    layer_2p = sigmoid(np.dot(layer_1p,synapse_h))
    layer_3p = sigmoid(np.dot(layer_2p,synapse_1))
    y_pred.append(layer_3p)
    index = index+batch_predict
    
y_pred = denormalize(np.reshape(y_pred,(-1,1)), data['value'], (-1,1))
testYseb = testY.reshape(-1,1)
testY = denormalize(testY, data['value'], (-1,1))
mse_pred = mse(testY,y_pred)
rmse_pred = rmse(testY,y_pred)
mae_pred = mae(testY,y_pred)
mape_pred = mape(testY,y_pred)
dstat_pred = dstat(testY,y_pred)
scoring = [mse_pred,rmse_pred,mae_pred,mape_pred,dstat_pred,run_time]


plt.plot(testY, marker='o', label='true') #testY[0:50] buat plotting coba liat di catatan
plt.plot(y_pred, marker='o', label='prediction')
plt.title('RNN-UKF')
plt.legend()
plt.show()

#scoring = [mse_pred,rmse_pred,mae_pred,dstat_pred,run_time]
print("mse : " , mse_pred)
print("rmse : ", rmse_pred) 
print("mape : ", mape_pred) 
print("mae: " , mae_pred)
print("dstat : " , dstat_pred)
print("runtime : ", run_time)

#%%
np.savetxt('bobot_input.csv', synapse_0, delimiter=',')
np.savetxt('bobot_hidden.csv', synapse_h, delimiter=',')
np.savetxt('bobot_output.csv', synapse_1, delimiter=',')
np.savetxt('loss_MLP.csv', mse_all, delimiter=';')