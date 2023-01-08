import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

# ADDataset
class ADDataset(Dataset):
    def __init__(self, data, scaler, pred_len):
        self.seq = data.iloc[:, :-1]
        self.data = scaler.transform(self.seq)
        self.target = data.iloc[:, -1]
        self.pred_len = pred_len

    def __getitem__(self, index):
        begin = index
        end = begin + self.pred_len

        return self.data[begin:end-1], self.data[end-1]

    def __len__(self):
        return len(self.data) - self.pred_len + 1

# ADtestDataset
class ADtestDataset(Dataset):
    def __init__(self, data, scaler, pred_len):
        self.seq = data.iloc[:, :-1]
        self.label = data.iloc[:, -1]                
        self.data = scaler.transform(self.seq)
        self.pred_len = pred_len

    def __getitem__(self, index):
        begin = index
        end = begin + self.pred_len

        return self.data[begin:end-1], self.data[end-1], self.label[end-1]

    def __len__(self):
        return len(self.data) - self.pred_len + 1

# LSTFADDataset
class LSTFADDataset(Dataset):
    def __init__(self, data, scaler, seq_len, pred_len):
        self.seq = data.iloc[:, :-1]
        self.data = scaler.transform(self.seq)
        self.target = data.iloc[:, -1]
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        
        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]

        return seq_x, seq_y

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

# LSTFADtestDataset
class LSTFADtestDataset(Dataset):
    def __init__(self, data, scaler, seq_len, pred_len):
        self.seq = data.iloc[:, :-1]
        self.data = scaler.transform(self.seq)
        self.target = data.iloc[:, -1]
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        
        seq_x = self.data[s_begin:s_end]

        return seq_x, self.target[r_end-1]

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

# read_data
def data_read(args, dirs):
    # all_files listing
    all_files = []
    if dirs=='valve1':
        for z in range(s):
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
    df = dataframe_from_csvs(all_files)
    
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

    return AD_train_loader, AD_threshold_loader, AD_test_loader, LSTFAD_train_loader, LSTFAD_valid_loader, LSTFAD_test_loader


