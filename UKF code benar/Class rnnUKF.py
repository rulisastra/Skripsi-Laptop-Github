import pandas as pd
import numpy as np

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

class RnnUkf(self, object):
    
    def __init__(self, object):
        
    
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
