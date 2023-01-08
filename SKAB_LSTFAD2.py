import os
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt

#from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score

from util.metric import TSMetric

import datetime

import numpy as np
import pandas as pd

from data_process.SKAB_loader2 import data_read

from models.lstf_model.RNN_forecasting import RNN_forecasting
from models.lstf_model.SCINet import SCINet
from models.lstf_model.Informer import Informer
from models.lstf_model.NLinear import NLinear
from models.lstf_model.DLinear import DLinear
from models.lstf_model.Liformer import Liformer

from models.ad_model.Classification import DNN, CNN

import argparse

parser = argparse.ArgumentParser(description='AWFD')

parser.add_argument('--seed', type=int, default=117, help='seed')
parser.add_argument('--data', type=str, default='SKAB', help='data_name')
parser.add_argument('--model', type=str, default='RNN', help='model:: RNN, SCINet, Informer, NLinear')
parser.add_argument('--ad_model', type=str, default='DNN', help='model:: DNN, CNN')

parser.add_argument('--threshold_rate', type=float, default=1, help='threshold_rate')
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

#seed
if args.seed is not None:
    import random
    import numpy as np
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Let's use", torch.cuda.device_count(), "GPUs!")
print('device:', device)

# AD_train
def ADtrain(args, train_loader, valid_loader, model, criterion, optimizer, dirs):
    # AD_train
    total_loss = 0
    train_loss = []
    i = 1
    stop_loss = np.inf
    count = 0
    for epoch in range(args.epoch):
        #train
        model.train()
        for data, target in train_loader:
            data = data.float().to(device)
            target = target.long().to(device)
    
            output = model(data)
            loss = criterion(output, target)
    
            total_loss += loss
            train_loss.append(total_loss/i)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            if i % 10 == 0:
                print(f'Epoch: {epoch+1}\t Train Step: {i:3d}\t Loss: {loss:.4f}')
            i += 1
        print(f'Epoch: {epoch+1} finished')

        #validation
        with torch.no_grad():
            model.eval()
            valid_loss = []
            for data, target in valid_loader:
                data = data.float().to(device)
                target = target.long().to(device)

                output = model(data)
                loss = criterion(output, target)
                valid_loss.append(loss.detach().cpu().numpy())
    
        #early_stopping
        if not os.path.exists('./path_SKAB_AD2'):
            os.makedirs('./path_SKAB_AD2')
        if np.mean(valid_loss) < stop_loss:
            stop_loss = np.mean(valid_loss)
            print('best_loss:: {:.4f}'.format(stop_loss))
            torch.save(model.state_dict(), f'./path_SKAB_AD2/{args.data}_{dirs}_{args.ad_model}_batch_{args.batch_size}_lr_{args.lr}.pth')
            count = 0
        else:
            count += 1
            print(f'EarlyStopping counter: {count} out of {args.patience}')
            print(f'best_loss:: {stop_loss:.4f}\t valid_loss:: {np.mean(valid_loss):.4f}' )
            if count >= args.patience:
                print('Ealry stopping')
                break
    return

# LSTF_train
def LSTFtrain(args, train_loader, valid_loader, model, criterion, optimizer, dirs):
    # AD_train
    total_loss = 0
    train_loss = []
    i = 1
    stop_loss = np.inf
    count = 0
    for epoch in range(args.epoch):
        #train
        model.train()
        for data_x, data_y in train_loader:
            data_x = data_x.float().to(device)
            data_y = data_y.float().to(device)
            
            if args.model == 'Informer':
                dec_inp = torch.zeros([data_x.shape[0], args.pred_len, data_x.shape[-1]]).float().to(device)
                output = model(data_x, dec_inp)
            else:
                output = model(data_x)

            loss = criterion(output, data_y)
    
            total_loss += loss
            train_loss.append(total_loss/i)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            if i % 10 == 0:
                print(f'Epoch: {epoch+1}\t Train Step: {i:3d}\t Loss: {loss:.4f}')
            i += 1
        print(f'Epoch: {epoch+1} finished')
    
        #validation
        with torch.no_grad():
            model.eval()
            valid_loss = []
            for data_x, data_y in valid_loader:
                data_x = data_x.float().to(device)
                data_y = data_y.float().to(device)
                
                if args.model == 'Informer':
                    dec_inp = torch.zeros([data_x.shape[0], args.pred_len, data_x.shape[-1]]).float().to(device)
                    output = model(data_x, dec_inp)
                else:
                    output = model(data_x)

                loss = criterion(output, data_y)
                valid_loss.append(loss.detach().cpu().numpy())
    
        #early_stopping
        if not os.path.exists('./path_SKAB_LSTF2'):
            os.makedirs('./path_SKAB_LSTF2')
        if np.mean(valid_loss) < stop_loss:
            stop_loss = np.mean(valid_loss)
            print('best_loss:: {:.4f}'.format(stop_loss))
            torch.save(model.state_dict(), f'./path_SKAB_LSTF2/{args.data}_{dirs}_{args.model}_{args.r_model}_{args.seq_len}_{args.pred_len}_batch_{args.batch_size}_lr_{args.lr}.pth')
            count = 0
        else:
            count += 1
            print(f'EarlyStopping counter: {count} out of {args.patience}')
            print(f'best_loss:: {stop_loss:.4f}\t valid_loss:: {np.mean(valid_loss):.4f}' )
            if count >= args.patience:
                print('Ealry stopping')
                break
    return

def ADeval(args, test_loader, model, criterion, dirs):
    # AD_test
    test_acc_sum = 0
    y_true = []
    y_pred = []
    tp_loss = []
    tn_loss = []
    fp_loss = []
    fn_loss = []
    with torch.no_grad():
        model.eval()
        for i, (data, target) in enumerate(test_loader):
            data = data.float().to(device)
            target = target.long().to(device)
    
            output = model(data)
            loss = criterion(output, target)
    
            prediction = output.max(1)[1]
            test_acc_sum += prediction.eq(target).sum()

            y_true.append(target.detach().cpu().numpy())
            y_pred.append(prediction.detach().cpu().numpy())

        print(f'  Test Accuracy: {100 * test_acc_sum / len(test_loader.dataset):.4f}')

    values_real = np.array(y_true, dtype=int)
    values_pred = np.array(y_pred, dtype=int)

    recall_classic = recall_score(values_real, values_pred)
    precision_classic = precision_score(values_real, values_pred)
    f1_classic = f1_score(values_real, values_pred)
    print("classic metric")
    print("precision: ", precision_classic, "recall: ", recall_classic, "f1: ", f1_classic)

    # log
    with open('SKAB_log2.txt', 'a') as f:
        f.write(f'precision: {precision_classic:.4f}\t recall: {recall_classic:.4f}\t f1: {f1_classic:.4f}\n')

    m = detection_margine(values_pred, 5)
    m = np.array(m, dtype=int)

    recall_classic = recall_score(values_real, m)
    precision_classic = precision_score(values_real, m)
    f1_classic = f1_score(values_real, m)
    print("classic metric_m")
    print("precision: ", precision_classic, "recall: ", recall_classic, "f1: ", f1_classic)

    # log
    with open('SKAB_log2_m.txt', 'a') as f:
        f.write(f'precision: {precision_classic:.4f}\t recall: {recall_classic:.4f}\t f1: {f1_classic:.4f}\n')

    if not os.path.exists('./picture/SKAB2'):
        os.makedirs('./picture/SKAB2')
    if not os.path.exists('./picture/SKAB2_m'):
        os.makedirs('./picture/SKAB2_m')

    plt.figure(figsize=(20, 10))
    plt.title(f'{dirs}_{args.ad_model}')
    plt.plot(values_real*1.25)
    plt.plot(values_pred)
    plt.legend(["real anomalies", "predicted anomalies"])
    plt.savefig(f'./picture/SKAB2/AD2_{dirs}_{args.ad_model}_{args.ad_r_model}.png')
    plt.close()

    plt.figure(figsize=(20, 10))
    plt.title(f'{dirs}_{args.ad_model}')
    plt.plot(values_real*1.25)
    plt.plot(m)
    plt.legend(["real anomalies", "predicted anomalies"])
    plt.savefig(f'./picture/SKAB2_m/AD2_{dirs}_{args.ad_model}_{args.ad_r_model}.png')
    plt.close()

    return

def LSTFADeval(args, test_loader, LSTFmodel, ADmodel, criterion, dirs):
    # AD_test
    test_acc_sum = 0
    y_true = []
    y_pred = []
    tp_loss = []
    tn_loss = []
    fp_loss = []
    fn_loss = []
    with torch.no_grad():
        LSTFmodel.eval()
        ADmodel.eval()
        for i, (data, target) in enumerate(test_loader):
            data = data.float().to(device)
            target = target.long().to(device)
            
            if args.model == 'Informer':
                dec_inp = torch.zeros([data.shape[0], args.pred_len, data.shape[-1]]).float().to(device)
                pred = LSTFmodel(data, dec_inp)
            else:
                pred = LSTFmodel(data)  # pred.shape: [batch, pred_len, input_size]

            seq = pred[:, -1, :]
            
            output = ADmodel(seq)
            loss = criterion(output, target)

            prediction = output.max(1)[1]
            test_acc_sum += prediction.eq(target).sum()

            y_true.append(target.detach().cpu().numpy())
            y_pred.append(prediction.detach().cpu().numpy())

            #print(f'\nTest set: Loss: {loss:.4f}')

        print(f'  Test Accuracy: {100 * test_acc_sum / len(test_loader.dataset):.4f}')

    values_real = np.array(y_true, dtype=int)
    values_pred = np.array(y_pred, dtype=int)

    recall_classic = recall_score(values_real, values_pred)
    precision_classic = precision_score(values_real, values_pred)
    f1_classic = f1_score(values_real, values_pred)
    print("classic metric")
    print("precision: ", precision_classic, "recall: ", recall_classic, "f1: ", f1_classic)

    # log
    with open('SKAB_log2.txt', 'a') as f:
        f.write(f'precision: {precision_classic:.4f}\t recall: {recall_classic:.4f}\t f1: {f1_classic:.4f}\n')

    m = detection_margine(values_pred, 5)
    m = np.array(m, dtype=int)

    recall_classic = recall_score(values_real, m)
    precision_classic = precision_score(values_real, m)
    f1_classic = f1_score(values_real, m)
    print("classic metric_m")
    print("precision: ", precision_classic, "recall: ", recall_classic, "f1: ", f1_classic)

    # log
    with open('SKAB_log2_m.txt', 'a') as f:
        f.write(f'precision: {precision_classic:.4f}\t recall: {recall_classic:.4f}\t f1: {f1_classic:.4f}\n')

    if not os.path.exists('./picture/SKAB2'):
        os.makedirs('./picture/SKAB2')
    if not os.path.exists('./picture/SKAB2_m'):
        os.makedirs('./picture/SKAB2_m')

    plt.figure(figsize=(20, 10))
    plt.title(f'{dirs}_{args.model}_{args.ad_model}')
    plt.plot(values_real*1.25)
    plt.plot(values_pred)
    plt.legend(["real anomalies", "predicted anomalies"])
    plt.savefig(f'./picture/SKAB2/LSTFAD2_{dirs}_{args.model}_{args.r_model}_{args.ad_model}_{args.ad_r_model}.png')
    plt.close()
    
    plt.figure(figsize=(20, 10))
    plt.title(f'{dirs}_{args.model}_{args.ad_model}')
    plt.plot(values_real*1.25)
    plt.plot(m)
    plt.legend(["real anomalies", "predicted anomalies"])
    plt.savefig(f'./picture/SKAB2_m/LSTFAD2_{dirs}_{args.model}_{args.r_model}_{args.ad_model}_{args.ad_r_model}.png')
    plt.close()

    return

def detection_margine(pred, margine):
    result = []
    tmp = []
    for p in pred:
        tmp.append(p)
        if p==1:
            pass
        else:
            if len(tmp)>margine:
                result.extend(tmp)
            else:
                result.extend([0]*len(tmp))
            tmp = []
    result.extend(tmp)
    return result

with open('SKAB_log2.txt', 'a') as f:
    f.write(f'{datetime.datetime.now()}\n')
with open('SKAB_log2_m.txt', 'a') as f:
    f.write(f'{datetime.datetime.now()}\n')
# all_data_train_test
for dirs in ['valve1','valve2']:
    # data_loader
    AD_train_loader, AD_valid_loader, AD_test_loader, LSTFAD_train_loader, LSTFAD_valid_loader, LSTFAD_test_loader = data_read(args, dirs)
    # load_model
    
    if args.ad_model == 'DNN':
        ADmodel = DNN(args.input_size, args.hidden_size//4, 2, args.pred_len)
    elif args.ad_model == 'CNN':
        ADmodel = CNN(args.input_size, args.hidden_size//4, 2, args.pred_len)
    else:
        print(args.ad_model=='DNN')
        print('No ad_Model')
        exit()
    ADmodel.to(device)
    
    if args.model == 'RNN':
        LSTFmodel = RNN_forecasting(args.input_size, args.seq_len, args.pred_len, r_model=args.r_model)
    elif args.model == 'SCINet':
        LSTFmodel = SCINet(args.pred_len, args.seq_len, args.input_size)
    elif args.model == 'NLinear':
        LSTFmodel = NLinear(args.seq_len, args.pred_len)
    elif args.model == 'DLinear':
        LSTFmodel = DLinear(args.seq_len, args.pred_len, args.input_size)
    elif args.model == 'Informer':
        LSTFmodel = Informer(args.input_size, args.input_size, args.input_size, args.seq_len, 0, args.pred_len)
    elif args.model == 'MCF':
        LSTFmodel = Liformer(args.seq_len, args.pred_len, args.input_size)
    else:
        print('No Model')
        exit()
    LSTFmodel.to(device)
    
    # criterion, ADoptimizer
    s_criterion = nn.CrossEntropyLoss().to(device)
    criterion = nn.MSELoss().to(device)
    ADoptimizer = optim.Adam(ADmodel.parameters(), lr=args.lr)
    LSTFoptimizer = optim.Adam(LSTFmodel.parameters(), lr=args.lr)
    
    # train
    ADtrain(args, AD_train_loader, AD_valid_loader, ADmodel, s_criterion, ADoptimizer, dirs)
    LSTFtrain(args, LSTFAD_train_loader, LSTFAD_valid_loader, LSTFmodel, criterion, LSTFoptimizer, dirs)
    
    # load_best model
    ADmodel.load_state_dict(torch.load(f'./path_SKAB_AD2/{args.data}_{dirs}_{args.ad_model}_batch_{args.batch_size}_lr_{args.lr}.pth'))
    LSTFmodel.load_state_dict(torch.load(f'./path_SKAB_LSTF2/{args.data}_{dirs}_{args.model}_{args.r_model}_{args.seq_len}_{args.pred_len}_batch_{args.batch_size}_lr_{args.lr}.pth'))
    
    # evaluate
    
    with open('SKAB_log2.txt', 'a') as f:
        f.write(f'{datetime.datetime.now()}\n')
        if args.model == 'RNN':
            f.write(f'{dirs}_{args.r_model}_{args.ad_model}_{args.seq_len}_{args.pred_len}\n')
        else:
            f.write(f'{dirs}_{args.model}_{args.ad_model}_{args.seq_len}_{args.pred_len}\n')

    with open('SKAB_log2_m.txt', 'a') as f:
        f.write(f'{datetime.datetime.now()}\n')
        if args.model == 'RNN':
            f.write(f'{dirs}_{args.r_model}_{args.ad_model}_{args.seq_len}_{args.pred_len}\n')
        else:
            f.write(f'{dirs}_{args.model}_{args.ad_model}_{args.seq_len}_{args.pred_len}\n')

    AD_results = ADeval(args, AD_test_loader, ADmodel, s_criterion, dirs)
    LSTFAD_results = LSTFADeval(args, LSTFAD_test_loader, LSTFmodel, ADmodel, s_criterion, dirs)


