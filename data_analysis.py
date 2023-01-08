from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import librosa
import librosa.display

import argparse

parser = argparse.ArgumentParser(description='AWFD')

parser.add_argument('--seed', type=int, default=117, help='seed')
parser.add_argument('--data', type=str, default='SKAB', help='data_name')
parser.add_argument('--model', type=str, default='RNN', help='model:: RNN, SCINet, Informer, NLinear')
parser.add_argument('--ad_model', type=str, default='DNN', help='model:: DNN, CNN, CNN2')

parser.add_argument('--threshold_rate', type=float, default=0.2, help='threshold_rate')
parser.add_argument('--epoch', type=int, default=100, help='epoch')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
parser.add_argument('--lr', type=float, default=1e-4, help='learning_rate')
parser.add_argument('--r_model', type=str, default='LSTM', help='rnn model; model:: LSTM, GRU')
parser.add_argument('--ad_r_model', type=str, default='LSTM', help='rnn model; model:: LSTM, GRU')

parser.add_argument('--input_size', type=int, default=8, help='dimension of data')
parser.add_argument('--hidden_size', type=int, default=128, help='dimension of latent vector')
parser.add_argument('--n_layer', type=int, default=3, help='n_layers of rnn model')

parser.add_argument('--seq_len', type=int, default=120, help='input sequence length of encoder')
parser.add_argument('--pred_len', type=int, default=60, help='input sequence length of decoder')

parser.add_argument('--patience', type=int, default=3, help='patience of early_stopping')
parser.add_argument('--evaluate', action='store_false', default=True)
parser.add_argument('--gpus', type=str, default='0', help='gpu numbers')

args = parser.parse_args()


# all_files listing
dirs='valve1'
all_files = []
if dirs=='valve1':
    for z in range(1):
        all_files.append(f'./SKAB/data/{dirs}/{z}.csv')
elif dirs=='valve2':
    for z in range(4):
        all_files.append(f'./SKAB/data/{dirs}/{z}.csv')
else:
    print('dirs_name problem in SKAB_loader2.py error')
    exit()

# read data
def dataframe_from_csv(target):
    return pd.read_csv(target, index_col='datetime', sep=';', parse_dates=True).rename(columns=lambda x: x.strip())

# concat data
def dataframe_from_csvs(targets):
    return pd.concat([dataframe_from_csv(x) for x in targets])

# read train_data
vari_dict = ['Accelerometer1RMS', 'Accelerometer2RMS', 'Current', 'Pressure', 'Temperature', 'Thermocouple', 'Voltage', 'Volume Flow RateRMS']
for i in range(8):
    for j in range(2):
        scaler = StandardScaler()
        df = dataframe_from_csvs(all_files)
        df = df[df['anomaly']==j]
        df1=df.iloc[:, i:i+1].values #.apply(lambda x: x*10000).values
        scaler.fit(df1)
        df = scaler.transform(df1)
        
        fft = np.fft.fft(df)
        
        spectrum = np.abs(fft)
        
        f = np.linspace(0, 1, len(spectrum))
        
        left_spectrum = spectrum[1:int(len(spectrum)/2)]
        left_f = f[1:int(len(spectrum)/2)]
        
        plt.figure(figsize=(30,10))
        plt.plot(left_f, left_spectrum, alpha=0.4)
        plt.xlabel('Frequency')
        plt.ylabel('Magnitude')
        if j==0:
            plt.title(f'Power spectrum {vari_dict[i]} normal')
            plt.savefig(f'./data_analysis/df_fft_{i}_normal.png')
        elif j==1:
            plt.title(f'Power spectrum {vari_dict[i]} abnormal')
            plt.savefig(f'./data_analysis/df_fft_{i}_abnormal.png')
        else:
            print('no')
            exit()
        plt.close()



'''
for i in range(1):
    i=7
    df = dataframe_from_csvs(all_files)
    df=df.iloc[:, i].values #.apply(lambda x: x*10000).values
    
    D = np.abs(librosa.stft(df, n_fft=8))
    plt.figure(dpi=5000)
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='linear', x_axis='time')
    plt.title('Spectrogram')
    plt.ylim(0, 4000)
    #plt.xlim(7,12)
    plt.colorbar()
    plt.savefig(f'./data_analysis/df_librosa_{i}.png')
    plt.close()
'''
exit()

print(df0.describe())
print(df.describe())
exit()

train_len = int(len(df)*0.5)
valid_len = int(len(df)*0.2)

ad_train_data = df.iloc[:train_len, :-1]
ad_train_data = ad_train_data[ad_train_data['anomaly']==0]
ad_threshold_data = df.iloc[train_len:train_len+valid_len, :-1]
ad_threshold_data = ad_threshold_data[ad_threshold_data['anomaly']==0]
ad_test_data = df.iloc[train_len+valid_len:, :-1]
lstfad_train_data = df.iloc[:train_len, :-1]
lstfad_valid_data = df.iloc[train_len-args.seq_len:train_len+valid_len, :-1]
lstfad_test_data = df.iloc[train_len+valid_len-args.seq_len:, :-1]

# StandardScaler
scaler = StandardScaler()
scaler.fit(ad_train_data.iloc[:, :-1])

# dataset configure
AD_train_dataset = ADDataset(ad_train_data, scaler, args.pred_len)
AD_threshold_dataset = ADDataset(ad_threshold_data, scaler, args.pred_len)
AD_test_dataset = ADtestDataset(ad_test_data, scaler, args.pred_len)
LSTFAD_train_dataset = LSTFADDataset(lstfad_train_data, scaler, args.seq_len, args.pred_len)
LSTFAD_valid_dataset = LSTFADDataset(lstfad_valid_data, scaler, args.seq_len, args.pred_len)
LSTFAD_test_dataset = LSTFADtestDataset(lstfad_test_data, scaler, args.seq_len, args.pred_len)

# data_loader configure
AD_train_loader = DataLoader(AD_train_dataset, batch_size=args.batch_size, shuffle=True)
AD_threshold_loader = DataLoader(AD_threshold_dataset, batch_size=1)
AD_test_loader = DataLoader(AD_test_dataset, batch_size=1)
LSTFAD_train_loader = DataLoader(LSTFAD_train_dataset, batch_size=args.batch_size, shuffle=True)
LSTFAD_valid_loader = DataLoader(LSTFAD_valid_dataset, batch_size=args.batch_size)
LSTFAD_test_loader = DataLoader(LSTFAD_test_dataset, batch_size=1)

