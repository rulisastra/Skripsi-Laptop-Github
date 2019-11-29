import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.linalg import cholesky

# =============================================================================
# from filterpy.kalman import UnscentedKalmanFilter as UKF
# from filterpy.kalman import MerweScaledSigmaPoints as Merwe
# =============================================================================

#%% persiapan data
data = pd.read_csv('data.csv',usecols=[1],engine='python',delimiter=',',decimal=".",thousands=',',header=None,names=['date','value'] )
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

# normalisasi dengan data satu kolom kebawah
data_raw = normalize(data['value'],(-1,1))

plt.plot(data_raw[0:1000], label='data mentah')
plt.title('data_raw')
plt.legend()
plt.show()

# pembagian data latih dan test
train_data, test_data = pisahData(data_raw, 0.7, 0.3)
train_data = train_data.reshape(-1,1) #satu kolom kebawah
test_data = test_data.reshape(-1,1)

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
windowSize = 3
trainX, trainY = createDataset(train_data,windowSize)
testX, testY = createDataset(test_data, windowSize)

#%% PELATIHAN ====  gunain trainX & trainY ====
# INISIALISASI banyaknya neuron setiap layer 
alpha = 0.1
batch_dim = trainX.shape[0] # ambil jumlah baris (n) dari trainX(n,m)
input_dim = windowSize
hidden_dim = 6
output_dim = 1

np.random.seed(4)

# BOBOT === inisialisasi bobot awal (baris,kolom)
synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1 # inisialisasi random bobot awal
synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1 # dengan interval [-1,1]
synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1 # random.random ->> interval [0,1]

synapse_0_update = np.zeros_like(synapse_0) #meng-0 kan semua isi array sesuai shape dr variabel (synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

# log mse tiap epoch
mse_all = []

# inisialisasi sebelum train
jumlah_w = (input_dim*hidden_dim)+(hidden_dim*hidden_dim)+(hidden_dim*output_dim)
Q = 1*np.identity(jumlah_w) #kovarian Noise process
R = 1*np.identity(output_dim) #Kovarian Noise measurement(observasi)
P = 1*np.identity(jumlah_w) #kovarian estimasi vektor state

#%% Unscented Kalman Filter without filterpy

beta = 2.
kappa = 0
lambda_ = 1. # lambda_ = alpha**2 * (n + kappa) - n
# =============================================================================
# points = Merwe(n, alpha, beta, kappa) #makin besar alpha, makin menyebar data[:train_data], range(train_data)
# kf = UKF(dim_x=1, dim_z=1, dt=.1, hx=None, fx=None, points=points) # sigma = points
# =============================================================================
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

epoch = 5
start_time = time.time()
for i in range(epoch):
    index = 0
    layer_2_value = []
    context_layer = np.full((batch_dim,hidden_dim),0) 
    layer_h_deltas = np.zeros(hidden_dim) # context layer (sebelumnya)

    while(index+batch_dim<=trainX.shape[0]):        
        # input dan output
        X = trainX[index:index+batch_dim,:]
        Y = trainY[index:index+batch_dim]
        index = index+batch_dim

        # forwardpass = propagated = bawa ke input ~> prev hidden
        layer_1 = tanh(np.dot(X,synapse_0) + np.dot(context_layer,synapse_h))
    
        # hidden ~> output
        layer_2 = tanh(np.dot(layer_1,synapse_1))
        layer_2_value.append(layer_2)
    
        # hitung error output
        layer_2_error = layer_2 - Y[:,None] #problemnya, y diganti dr Y matrix
        # layer_2_deltas.append((layer_2_error)*dtanh(layer_2))
        
        # error di output layer -> layer 2 deltas (masuk ke context layer dari hidden layer)
        layer_2_delta = layer_2_error*dtanh(layer_2)
        
        # error di hidden layer -> layer 1 delta (masuk ke hidden layer dari context layer)
        layer_1_delta = (np.dot(layer_h_deltas,synapse_h.T) + np.dot(layer_2_delta,synapse_1.T)) * dtanh(layer_1)
    
        # calculate weight update
        synapse_1_update = np.dot(np.atleast_2d(layer_1).T,(layer_2_delta))
        synapse_h_update = np.dot(np.atleast_2d(context_layer).T,(layer_1_delta))
        synapse_0_update = np.dot(X.T,(layer_1_delta))
        
        # concatenate weight
        synapse_0_c = np.reshape(synapse_0,(-1,1))
        synapse_h_c = np.reshape(synapse_h,(-1,1))
        synapse_1_c = np.reshape(synapse_1,(-1,1))
        w_concat = np.concatenate((synapse_0_c,synapse_h_c,synapse_1_c), axis=0)

        ''' ============= UKF di measurement ============= '''
        
        n = w_concat.size # julier versi masalah 'dimension of problem'
        
        # Sigma points around mean
        mean = np.sum(w_concat) / n # mean secara keseluruhan
        lambda_ = alpha**2 *(n + kappa) - n  # Merwe
# =============================================================================
#         implements U'*U = (n+kappa)*P. Returns lower triangular matrix.
#         Take transpose so we can access with U[i]
# =============================================================================        
        U = cholesky((n + lambda_)*P) # sama dg np.sqrt
        
        # filterpy version dengan shape (121,60) jadi 60 karena dikali dg P!
        sigmas = np.zeros((2*n+1, n))
        # maka....
        s_ = np.reshape(w_concat,(1,-1))
        s_n = np.subtract(w_concat, -U)
        s_n2 = np.subtract(w_concat, U)
        sigmas_concat = np.concatenate((s_,s_n,s_n2), axis=0) # (121,60)

        '''
            # filterpy version (sigmas jadi (121,60))
            for k in range(n):
                sigmas_n =  np.subtract(w_concat, -U[k])
                sigmas_2n = np.subtract(w_concat, U[k])
            
            # pykalman version (nge ravel() jadi (1,7260))
            sigmas = np.tile(w_concat.T, (1, 2 * n + 1))
            sigmas[:, 1:(n + 1)] += P * np.sqrt(U) # mu + each column of sigma2 * sqrt(c)
            sigmas[:, (n + 1):] -= P * np.sqrt(U) # mu - each column of sigma2 * sqrt(c)
            
            # ruli version (m_utama sbg w_concat)
            m_1n = np.subtract(m_utama, -U)
            m_1n2 = np.subtract(m_utama, U)
            m_concat = np.concatenate((m_utama,m_1n,m_1n2), axis=0) # Z besar
            m_concat_sum = np.reshape(np.sum(m_concat, axis=1),(-1,1))
 
            # Wm
            Wm_sum = np.reshape(np.sum(m_1concat, axis=1),(-1,1))
            Wc_sum = np.reshape(np.sum(P, axis=1),(-1,1))
            
            # menuju....
            Mz = np.dot(np.reshape(Wm_sum,(1,-1)),m_1concat) # myu z
            
            # kovarian
            Pz1 = np.subtract(m_1concat,Mz)
            Pz2 = np.dot(Pz1,Pz1.T) # (60,60)
            Pz = Wc_sum * Pz2 # + R) 
    
# =============================================================================
#                     gila gede banget coy matrixnya 
#                     HARUSNYA dot tapi matrixnya lbh gede
# =============================================================================
    
            # Kalman gain
            K1 = np.subtract(w_concat,Mz)
            K2= np.dot(K1,Pz1.T) # (60,60)     
            K3 = np.dot(np.reshape(Wm_sum,(1,-1)),inv(Pz)) # dibalik Wm_sum nya :()
            K = np.reshape(K3,(-1,1))
     
            # update weight
            innovation = ((Y-layer_2).sum()/len(layer_2_error)) # hitung error pred dan yang diharapkan
            # w_concat_new = w_concat + np.dot(K,innovation)
            w_concat_new = w_concat + np.dot(K,innovation) # bikin yang assal, asal networknya jalan
    
            # kovarian Update
            K_transpose = K.T
            # P3 = Pz[1,:]
            # P2 = np.dot(P3,K_transpose)
            
        '''
            
        # Pembobotan sigma points
        # filterpy version
        Wc = np.full(2*n + 1, 1. / (2*(n + lambda_)))
        Wm = np.full(2*n + 1, 1. / (2*(n + lambda_)))
        Wc[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)
        Wm[0] = lambda_ / (n + lambda_)
        
        # menuju....
        Mz = np.dot(np.reshape(Wm,(1,-1)),sigmas_concat) # myu z
                
        # Pz (kovarian di measurement)
        # SEMUA BERANTAKAN GARA2 Pz
        kmax, n = sigmas.shape
        Pz_ = np.zeros(sigmas.shape)
        for k in range(kmax):
            C = np.subtract(sigmas_concat[k],Mz)
            # harusnya +=
            Pz_ = Wc[k] * np.outer(C, C) # makin besar nilainya dengan +=
        Pz = Pz_ + R

        # Kalman gain
        K1 = np.subtract(w_concat,Mz)
        K2= np.dot(K1,Pz.T) # (60,60)
        Pxz_ = np.zeros(sigmas.shape)
        for i in range(kmax):
            Cc = np.subtract(sigmas,Mz)
            C = np.subtract(sigmas_concat[k],Mz)
            # harusnya +=
            Pxz_ = Wc[k] * np.outer(Cc, C) # makin besar nilainya dengan +=
        Pxz = Pxz_ + R
        K3 = np.dot(np.reshape(Wc,(1,-1)),inv(Pz)) # dibalik Wm_sum nya :()
        K = np.reshape(K3,(-1,1))
        
'''
        # update weight
        innovation = ((Y-layer_2).sum()/len(layer_2_error)) # hitung error pred dan yang diharapkan
        # w_concat_new = w_concat + np.dot(K,innovation)
        w_concat_new = w_concat + np.dot(K,innovation) # bikin yang assal, asal networknya jalan

        # kovarian Update
        K_transpose = K.T
        # P3 = Pz[1,:]
        # P2 = np.dot(P3,K_transpose)
        
        # assign bobot versi ekf
        synapse_0 = w_concat_new[0:(input_dim*hidden_dim),0]
        synapse_h = w_concat_new[(input_dim*hidden_dim):(input_dim*hidden_dim)+(hidden_dim*hidden_dim),0]
        synapse_1 = w_concat_new[(input_dim*hidden_dim)+(hidden_dim*hidden_dim):w_concat_new.shape[0],0]
        
        # reshape balik bobot
        synapse_0 = np.reshape(synapse_0,(input_dim,hidden_dim))
        synapse_h = np.reshape(synapse_h,(hidden_dim,hidden_dim))
        synapse_1 = np.reshape(synapse_1,(hidden_dim,output_dim))
    
        # reset update
        synapse_0_update *= 0
        synapse_1_update *= 0
        synapse_h_update *= 0
    
        # update context layer
        layer_h_deltas = layer_1_delta
        context_layer = layer_1
               
    layer_2_value = np.reshape(layer_2_value,(-1,1))
    mse_epoch = mse(trainY,layer_2_value)
    mse_all.append(mse_epoch)
run_time = time.time() - start_time
#%% seberapa besar lossnya???

plt.plot(mse_all,label='loss')
plt.title('Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()

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

plt.plot(testYseb[0:50], label='true') #testY[0:50] buat plotting coba liat di catatan
plt.plot(layer_2p[0:50], label='prediction')
plt.xlabel('Data ke-')
plt.ylabel('Harga')
plt.title('RNN-UKF sebelun denormalisasi')
plt.legend()
plt.show()

plt.plot(testY, label='true') #testY[0:50] buat plotting coba liat di catatan
plt.plot(y_pred, label='prediction')
plt.xlabel('Data ke-')
plt.ylabel('Harga')
plt.title('RNN-UKF')
plt.legend()
plt.show()

plt.plot(m0, label='mean bobot = 1', marker ='x', ls=None) #testY[0:50] untuk plotting (catatan)
plt.plot(m_utama, label='bobot = all titik mean', color='r', marker ='o', ls=None)
plt.xlabel('bobot ke')
plt.ylabel('value')
plt.title('Bobot semua')
plt.legend()
plt.show()

plt.plot(m_1concat[:,0], marker='o', ls=':')
plt.plot(m_1concat[:,[1,2]],'x')
plt.xlabel('bobot ke')
plt.ylabel('value')
plt.title('Bobot per index')
plt.legend()
plt.show()

#scoring = [mse_pred,rmse_pred,mae_pred,dstat_pred,run_time]
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
'''