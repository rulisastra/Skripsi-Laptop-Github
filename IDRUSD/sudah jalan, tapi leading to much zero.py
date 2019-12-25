import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.linalg import cholesky
from scipy.stats import norm

#%% persiapan data

# IDRUSD.csv = 64% | 71#
# data baru  mulai 27sep2017.csv = 0.0
# data baru all = 21%
# data = 56%
# Ether mulai 1jul2017 = 52%
# Ether = 55%
# Litecoin mulai 1jan2017.csv = 55%
# NEO jul2017 = 62%
# NEO all = 63
# BTCIDR 1sept2017 | 74% 
# Currency Converter.csv = 70%
# Currency Converter - copy.csv 73% malah jadi 54% dengan model yang baru
# 1000 data.csv = 72% tapi mse naik

# USDIDR 2009_2019 = 48% dengan model yang baru

data = pd.read_csv('USDIDR 2009_2019.csv', # USDIDR 2009_2019   ///  USDIDR 2000_2010 cut   /// USDIDR 2000_2019 all
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
def tanh(x):
    return (1-np.exp(-2*x))/(1+np.exp(-2*x))

def dtanh(x):
    return (1-tanh(x)**2)

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

def norm(x, scale):
    normalized = []
    for i in range(len(x)):
        a = (min(scale))+(x[i]-min(x))*(max(scale)-min(scale))/(max(x)-min(x))
        normalized.append(a)
    return np.array(normalized)
    scale = (-1,1)
    normalized = normalize(x,scale)

# normalisasi dengan data satu kolom kebawah
data_raw = np.reshape(normalize(data['value'],(-1,1)),(-1,1))

# pembagian data latih dan test
train_data, test_data = pisahData(data_raw, 0.8, 0.2) #8:2 = 71%
# =============================================================================
# train_data = train_data.reshape(-1,1) #satu kolom kebawah
# test_data = test_data.reshape(-1,1)
# =============================================================================

plt.plot(data, c='g', ls=None)
plt.xlabel('Data ke-')
plt.ylabel('Harga (USD)')
plt.title('Harga Nilai Tukar IDR/USD')
plt.show()

plt.plot(data_raw, c='y', ls=None)
plt.title('Nilai Tukar IDR/USD [Ternormalisasi]')
plt.xlabel('Data ke-')
plt.ylabel('Jangkauan Harga (-1,1)')
plt.show()

plt.plot(data, c='b', label='test', ls=None)
plt.plot(data[1:3499], c='g', label='training', ls=None)
plt.title('Pembagian Data [7:3]')
plt.xlabel('Data ke-')
plt.ylabel('Harga (USD)')
plt.legend()
plt.show()

plt.plot(data_raw, c='b', label='test', ls=None)
plt.plot(train_data[0:-1], c='r', label='training', ls=None)
plt.title('Pembagian Data [7:3]')
plt.xlabel('Data ke-')
plt.ylabel('Jangkauan Harga')
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

# ===================================================
#%%
windowSize = 5 # 5 70%
epoch = 10 # 100
hidden_dim = 7 # 7,9, ---->>> 73% dengan wn=5 dan hd = 9 dan 7:3

#%%
trainX, trainY = createDataset(train_data,windowSize)
testX, testY = createDataset(test_data, windowSize)

#%% PELATIHAN ====  gunain trainX & trainY ====
# INISIALISASI banyaknya neuron setiap layer 
batch_dim = trainX.shape[0] # ambil jumlah baris (n) dari trainX(n,m)
input_dim = windowSize
output_dim = 1
alpha = .001

np.random.seed(1) # 1 =72%

# BOBOT === inisialisasi bobot awal (baris,kolom)
synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1 # inisialisasi random bobot awal
synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1 # dengan interval [-1,1]
synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1 # random.random ->> interval [0,1]

synapse_0_update = np.zeros_like(synapse_0) #meng-0 kan semua isi array sesuai shape dr variabel (synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

# log mse tiap epoch
mse_all = []
rmse_all = []

# inisialisasi sebelum train
jumlah_w = (input_dim*hidden_dim)+(hidden_dim*hidden_dim)+(hidden_dim*output_dim)
Q = 1*np.identity(jumlah_w) #kovarian Noise process
R = 1*np.identity(output_dim) #Kovarian Noise measurement(observasi) output_dim
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
epsilon = 0.98 # sbg forgetting factor berdasarkan dash 2014
gamma = 1.98 # untuk minimize error karena fluktuasi (dash)
start_time = time.time()
for i in range(epoch):
    index = 0
    layer_2_value = []
    context_layer = np.full((batch_dim,hidden_dim),0) 
    layer_h_deltas = np.zeros(hidden_dim) # context layer (sebelumnya)
    sigmas_concat = []
    while(index+batch_dim<=trainX.shape[0]):        
        # input dan output
        X = trainX[index:index+batch_dim,:]
        Y = trainY[index:index+batch_dim]
        index = index+batch_dim

        # bawa ke input ~> prev hidden
        layer_1 = tanh(np.dot(X,synapse_0) + np.dot(context_layer,synapse_h))
    
        # hidden ~> output
        layer_2 = tanh(np.dot(layer_1,synapse_1))
        layer_2_value.append(layer_2)
    
        # hitung error output
        layer_2_error = layer_2 - Y[:,None] #problemnya, y diganti dr Y matrix
        
        # error di output layer -> layer 2 deltas (masuk ke context layer dari hidden layer)
        layer_2_delta = layer_2_error*dtanh(layer_2)
        
        # seberapa besar error berpengaaruh terhadap layer
        layer_1_error = np.dot(layer_2_delta,synapse_1.T)
        
        # error di hidden layer -> layer 1 delta (masuk ke hidden layer dari context layer)
        layer_1_delta = (layer_1_error + np.dot(layer_h_deltas,synapse_h.T)) * dtanh(layer_1)
    
        # calculate weight update
        synapse_1_update = np.dot(np.atleast_2d(layer_1).T,(layer_2_delta))
        synapse_h_update = np.dot(np.atleast_2d(context_layer).T,(layer_1_delta))
        synapse_0_update = np.dot(X.T,(layer_1_delta))
        
        #%% concatenate weight
        synapse_0_c = np.reshape(synapse_0,(-1,1))
        synapse_h_c = np.reshape(synapse_h,(-1,1))
        synapse_1_c = np.reshape(synapse_1,(-1,1))
        w_concat = np.concatenate((synapse_0_c,synapse_h_c,synapse_1_c), axis=0)
        w_concat_transpose = w_concat.T
        
        synapse_0_masuk = np.reshape(synapse_0_update,(1,-1)) # satu baris kesamping
        synapse_h_masuk = np.reshape(synapse_h_update,(1,-1))
        synapse_1_masuk = np.reshape(synapse_1_update,(1,-1))
        masuk = np.concatenate((synapse_0_masuk,synapse_h_masuk,synapse_1_masuk), axis=1)
        
        #%% Unscented Kalman Filter without filterpy
        # X_ = masuk # myu
        X_ = w_concat_transpose
        
        n = X_.size # julier versi masalah 'dimension of problem'
        L = X_.ndim #2
        beta = 2.
        kappa = 0 # dash
        # lambda_ = 0.001
        # lambda_ = 1 # ngaruh, menurunkan dstat 3%
        lambda_ = alpha**2 * (n + kappa) - n # bisoi, dash
        
        #%% SIGMA POINTS around mean
        U = cholesky((n + lambda_)*P) # sama dg np.sqrt
        # filterpy version dengan shape (121,60) karena dikali dg P!
        
        sigmas = np.zeros((2*n+1, n))
        sigmas[0] = X_ # filterpy version dengan shape (121,60) karena dikali dg P!
        # maka....
        for k in range(n): # gabung kebawah.. jadinya 121,60
            sigmas[k+1] = np.subtract(X_, -U[k])
            sigmas[n+k+1] = np.subtract(X_, U[k])
        
        # mengasmbil nilai eye dari sigmas
        sigmas_reshape_1 = sigmas[0]
        sigmas_reshape_2 = sigmas[1:n+1,:] * np.eye(n)
        sigmas_reshape_3 = sigmas[n+1::,:] * np.eye(n)
        
        # Menjadikan satu baris
        ones = np.reshape(np.ones(n),(-1,1))
        sigmas_1_c = np.reshape(sigmas_reshape_1.T,(-1,1))
        sigmas_2_c = np.dot(sigmas_reshape_2,ones)
        sigmas_3_c = np.dot(sigmas_reshape_3,ones)
        sigmas_concat = np.concatenate((sigmas_1_c,sigmas_2_c,sigmas_3_c), axis=1)
        
        #%% BOBOT SIGMA dari Merwe
        c_ = .5 / (n + lambda_)
        # Wm = np.full(2*n+1, c_)
        Wm  = np.full(2*n+1, 1 / (2*(n + lambda_)))
        Wc = Wm # size (121,) atau (n,)
        Wc[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)
        Wm[0] = lambda_ / (n + lambda_)
        
        #%% SIGMA Unscented Transform  (biar hold value dg eyeP) ke measurement
        Mz = np.dot(Wm,sigmas)
        # Mz = np.sum(np.dot(Wm,sigmas)) # yang bener
        '''
            sigmamin = np.subtract(sigmas,Mz)
            sigmamin_t = sigmamin.T
            sigmamin_square = np.dot(sigmamin,sigmamin_t)
            sigmamin_bbt = np.dot(Wc,sigmamin_square)
            # add_R = sigmamin_square + R
            # sigmamin_bbt_sum = sigmamin_bbt.sum() + R # Kovarian Pz
            
            sigmamin_lama = sigmamin
            si_gmamin_square = np.dot(sigmamin_lama,sigmamin_t)
            si_gmamin_bbt = np.dot(Wc,si_gmamin_square)
            # si_gmamin_bbt_sum = sigmamin_bbt.sum() # Tut
            
            Kk = np.dot(si_gmamin_bbt,inv(sigmamin_bbt))# harus inv
            Pp_1 = np.dot(sigmamin_bbt,Kk.T)
            Pp = P - np.dot(Kk,Pp_1)
        
        '''

        # KOVARIAN ke measurement juga
        kmax, n = sigmas.shape
        Pz = np.zeros(n)
        for k in range(kmax):
            c = np.subtract(sigmas[k],Mz)
            Pz = Wc[k] * np.outer(c, c)
            # makin besar nilainya dengan +=
        Pz += R # sebagai S
        
        # Cross covariance Weight sebelu dan weights sesudah unscented transform
        Pxz = np.zeros(sigmas.shape) # Tut
        for k in range(kmax):
            cc = np.subtract(X_,Mz) # sebagai T
            Pxz = Wc[k] * np.outer(cc, c)

        # Kalman gain
        K1 = np.dot(Pxz,inv(Pz))
        Knorm = np.reshape(norm(K1[:,0],(-1,1)),(-1,1)) # tambahan doang
        K = np.reshape(K1[:,0],(-1,1))
        
# =============================================================================
#         # innovasi dari dash 2014
#         P1 = np.dot(gamma**-2,np.identity(n))
#         P2 = np.subtract(inv(P),P1) # inv berdaasarkan paper dash 2014
#         P3 = np.dot(Pz,K.T)
#         P = P2 - np.dot(K,P3) # besarnya error kovarian matriks
# =============================================================================
        
        P = P - np.dot(Pz,K) # np.dot(Pz,K.T))
        
        #%%
        innovation = ((Y-layer_2).sum()/len(layer_2_error))
       
        w_concat_new = w_concat + np.dot(K,innovation)
        # w_concat_new = np.reshape(w_concat_new[:,0],(-1,1))
# =============================================================================
#         w_concat_new2 = np.reshape(w_concat_new1[0],(-1,1))
#         w_concat_new = norm(w_concat_new2,(-1,1))
# =============================================================================
        
        #assign bobot
        synapse_0 = w_concat_new[0:(input_dim*hidden_dim),0]
        synapse_h = w_concat_new[(input_dim*hidden_dim):(input_dim*hidden_dim)+(hidden_dim*hidden_dim),0]
        synapse_1 = w_concat_new[(input_dim*hidden_dim)+(hidden_dim*hidden_dim):w_concat_new.shape[0],0]
        
        #reshape balik bobot
        synapse_0 = np.reshape(synapse_0,(input_dim,hidden_dim))
        synapse_h = np.reshape(synapse_h,(hidden_dim,hidden_dim))
        synapse_1 = np.reshape(synapse_1,(hidden_dim,output_dim))
        
        # reset update
        synapse_0_update *= 0
        synapse_1_update *= 0
        synapse_h_update *= 0
    
        # update context layer
        layer_h_deltas = layer_1_delta # future_layer_1_delta
        context_layer = layer_1 # prev_hidden_layer
        
    layer_2_value = np.reshape(layer_2_value,(-1,1))
    
    mse_epoch = mse(trainY,layer_2_value)
    mse_all.append(mse_epoch)
    
    rmse_epoch = rmse(trainY,layer_2_value)
    rmse_all.append(rmse_epoch)
    
run_time = time.time() - start_time
#%% seberapa besar lossnya???

plt.plot(sigmas_1_c[0:3],label='sigma utama (mean)', marker='o', ls='-')
plt.plot(sigmas_2_c[0:3],label='sigma lain', marker='x', ls=None)
plt.plot(sigmas_3_c[0:3],label='sigma lain', marker='x', ls=None)
plt.title('SIGMAS')
plt.xlabel('sigmas ke-')
plt.ylabel('value')
plt.legend()
plt.show()

plt.plot(mse_all, marker='x')
plt.title('Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.show()

np.savetxt('loss_ukf.csv', mse_all, delimiter=';')

# plt.plot(trainY, c='r', marker='o', label='true') #testY[0:50] buat plotting coba liat di catatan
plt.plot(layer_2_value, c='y', marker='o', label='layer_2_value (predict)')
plt.plot(trainY, c='r', marker='o', label='trainY (sebenarnya)')
plt.title('Prediksi Training')
plt.xlabel('data ke-')
plt.ylabel('value')
plt.legend()
plt.show()

plt.plot(layer_2_error,label='error', marker = 'x',c='g')
plt.title('ERROR')
plt.xlabel('data ke-')
plt.ylabel('value')
plt.legend()
plt.show()

mse_pred = mse(trainY,layer_2_value)
mae_pred = mae(trainY,layer_2_value)
rmse_pred = rmse(trainY,layer_2_value)
mape_pred = mape(trainY,layer_2_value)
dstat_pred = dstat(trainY,layer_2_value)
scoring = [mse_pred,rmse_pred,mae_pred,mape_pred,dstat_pred,run_time]

print("Training mse : " , mse_pred)
print("Training mape : ", mape_pred) 
print("Training mae: " , mae_pred)
print("Training rmse : ", rmse_pred) 
print("Training dstat : " , dstat_pred)
print("Training runtime : ", run_time)
#%% mari coba ============ PREDIKSI ===================

batch_predict = testX.shape[0] # mengambil banyaknya baris (n) dari testX(n,m)
context_layer_p = np.full((batch_predict,hidden_dim),0) # return full array yg isinya (0) sebesar dimensi [batch_predict x hidden_dim]
y_pred = [] # hasil akhir Y prediksi
index = 0
while(index+batch_predict<=testX.shape[0]):
    X = testX[index:index+batch_predict,:]
    layer_1p = tanh(np.dot(X,synapse_0)+np.dot(context_layer_p,synapse_h))
    layer_2p = tanh(np.dot(layer_1p,synapse_1))
    y_pred.append(layer_2p)
    context_layer_p = layer_1p
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

plt.plot(testYseb, label='true') #testY[0:50] buat plotting coba liat di catatan
plt.plot(layer_2p, label='prediction')
plt.xlabel('Data ke-')
plt.ylabel('jangkauan')
plt.title('Jangkauan data uji keseluruhan 1000 data')
plt.legend()
plt.show()

plt.plot(testY[50:100], marker='o', label='true') #testY[0:50] buat plotting coba liat di catatan
plt.plot(y_pred[50:100], marker='o', label='prediction')
plt.title('HASIL UJI dengan metode RNN-UKF 50 data awal')
plt.xlabel('Data ke-')
plt.ylabel('Harga')
plt.legend()
plt.show()

# =============================================================================
# # plt.scatter(sigmas[0], color='r', marker ='x', ls=None)
# # plt.scatter(sigmas[1:91],sigmas[92:182])
# plt.plot(sigmas[1:91], ls=':')
# plt.plot(sigmas[92:182], ls=':')
# plt.plot(sigmas[0,:], color='r', marker ='o')
# plt.xlabel('sigmas ke-')
# plt.ylabel('value')
# plt.title('SIGMAS semua')
# plt.show()
# =============================================================================

print('Kalman dim: ', K.ndim)
print('Kalman size: ', K.size)
print('Kalman shape: ', K.shape)

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
np.savetxt('loss_ukf.csv', mse_all, delimiter=';')