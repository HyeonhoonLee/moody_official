import vitaldb
import numpy as np
import pandas as pd
import os
import scipy.signal
from scipy.signal import resample
import matplotlib.pyplot as plt
import random
import itertools as it
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simps
from mne.time_frequency import psd_array_multitaper
from sklearn.metrics import roc_auc_score
import sys
sys.path.append('../')
from resnet1d.crnn1d import CRNN
from resnet1d.acnn1d import ACNN


SRATE = 128  # in hz
SEGLEN = 1 * 60 * SRATE  # samples (5min) ###########################################
BATCH_SIZE = 1024
MAX_CASES = 6000
NFFT = SRATE * 30  # FFT length
SLIDE = SRATE * 10 # 10 sec

# Column order when loading data
EEG1 = 0
EEG2 = 1
SEVO = 2
BIS = 3


cachefile = '{}sec_{}cases.npz'.format(SEGLEN // SRATE, MAX_CASES)
a = True
if os.path.exists(cachefile) & a:
    dat = np.load(cachefile)
    x, y, b, c = dat['x'], dat['y'], dat['b'], dat['c']
else:
    df_trks = pd.read_csv("https://api.vitaldb.net/trks")  # track information
    df_cases = pd.read_csv("https://api.vitaldb.net/cases")  # patient information

    # Inclusion & Exclusion criteria
    caseids = set(df_cases.loc[df_cases['age'] > 18, 'caseid']) &\
        set(df_trks.loc[df_trks['tname'] == 'BIS/EEG1_WAV', 'caseid']) &\
        set(df_trks.loc[df_trks['tname'] == 'BIS/EEG2_WAV', 'caseid']) &\
        set(df_trks.loc[df_trks['tname'] == 'BIS/BIS', 'caseid'])
        #set(df_trks.loc[df_trks['tname'] == 'Primus/EXP_SEVO', 'caseid'])
    print('total {} cases'.format(len(caseids)))
    x = []  
    y = []  # sevo
    b = []  # bis
    c = []  # caseids
    icase = 0  # number of loaded cases
    for caseid in (caseids): 
        print('loading {} ({}/{})'.format(caseid, icase, MAX_CASES), end='...', flush=True)

        # Excluding the following values
        if np.any(vitaldb.load_case(caseid, 'Primus/EXP_SEVO') > 1):
            print('sevoflurane')
            continue
        if np.any(vitaldb.load_case(caseid, 'Primus/EXP_DES') > 1):
            print('desflurane')
            continue
        if np.any(vitaldb.load_case(caseid, 'Orchestra/PPF20_CE') > 0.2):
            print('propofol')
            pass
        if np.any(vitaldb.load_case(caseid, 'Primus/FEN2O') > 2):
            print('n2o')
            continue
        if np.any(vitaldb.load_case(caseid, 'Orchestra/RFTN50_CE') > 0.2):
            print('remifentanil')
            pass


         # Extract data
        vals = vitaldb.load_case(caseid, ['BIS/EEG1_WAV', 'BIS/EEG2_WAV', 'Orchestra/PPF20_CE', 'BIS/BIS'], 1 / SRATE) 
        #vals[:, 0] = vals[:, 0] - vals[:, 3] 여기서 EEG1_WAV - EEG2_WAV하면 이론적으로 F7 - F3이 나오게됨
        if np.nanmax(vals[:, SEVO]) < 1:
            print('all sevo <= 1')
            continue

        # Convert etsevo to the age related mac
        age = df_cases.loc[df_cases['caseid'] == caseid, 'age'].values[0]
#        vals[:, SEVO] /= 1.80 * 10 ** (-0.00269 * (age - 40))

        if not np.any(vals[:, BIS] > 0):
            print('all bis <= 0')
            continue

        # Since the EEG should come out well, we start from the location where the value of bis was first calculated.
        valid_bis_idx = np.where(vals[:, BIS] > 0)[0]
        first_bis_idx = valid_bis_idx[0]
        last_bis_idx = valid_bis_idx[-1]
        vals = vals[first_bis_idx:last_bis_idx + 1, :]

        if len(vals) < 1800 * SRATE:  # Do not use cases that are less than 30 minutes
            print('{} len < 30 min'.format(caseid))
            continue

        ## Forward fill in MAC value and BIS value up to 5 seconds
        vals[:, SEVO:] = pd.DataFrame(vals[:, SEVO:]).ffill(limit=5 * SRATE).values

        # Extract data every 10 second from its start to its end and then put into the dataset
        oldlen = len(y)

        ppfon = np.where(vals[:, SEVO] > 0.3)[0]  
        if len(ppfon)==0:
            print('no ppf on')
            continue

        else:
            ppfon = ppfon[0]
        

        for irow in range(0, ppfon+2*SEGLEN, SRATE*20):

            bis = vals[irow+SEGLEN, BIS]
            ppf = vals[irow+SEGLEN, SEVO]
            if np.isnan(bis) or np.isnan(ppf) or bis == 0:
            #if np.isnan(bis) or bis == 0:
                continue
            # add dataset
            eeg1 = vals[irow:irow+SEGLEN, EEG1]
            eeg2 = vals[irow:irow+SEGLEN, EEG2]
            eeg = np.vstack([eeg1, eeg2])
            #print(eeg.shape)  # (2, 7680)
            x.append(eeg)
            y.append(ppf)
            b.append(bis)
            c.append(caseid)

        # Valid case
        icase += 1
        print('{} samples read -> total {} samples ({}/{})'.format(len(y) - oldlen, len(y), icase, MAX_CASES))
        if icase >= MAX_CASES:
            break

    # Change the input dataset to a numpy array
    x = np.array(x)
    y = np.array(y)
    b = np.array(b)
    c = np.array(c)

    # Save cahce file
    np.savez(cachefile, x=x, y=y, b=b, c=c)


cachefile2 = '{}sec_{}cases_psd.npz'.format(SEGLEN // SRATE, MAX_CASES)

if os.path.exists(cachefile2)&a:
    dat = np.load(cachefile2)
    psd_ms, y, b, c = dat['psd'], dat['y'], dat['b'], dat['c']
    print('psd loaded')
else:
    # Remove missing values
    print('invalid samples...', end='', flush=True)
    print(x.shape)
    valid_mask = ~(np.max(np.isnan(x), axis=2) > 0) # nan이 있으면 제거
    valid_mask &= (np.max(x, axis=2) - np.min(x, axis=2) > 12)  # bis 임피던스 체크 eeg의 전체 range가 12 미만이면 제거
    '''
    x = x[valid_mask, :]
    y = y[valid_mask]
    b = b[valid_mask]
    c = c[valid_mask]
    '''
    print('{:.1f}% removed'.format(100*(1-np.mean(valid_mask))))

    
    # Filtering
    print('baseline drift...', end='', flush=True)
    x[:, EEG1, :] -= scipy.signal.savgol_filter(x[:,EEG1, :], 91, 3)  # remove baseline drift
    x[:, EEG2, :] -= scipy.signal.savgol_filter(x[:,EEG2, :], 91, 3)  # remove baseline drift
    print('removed')

    # Remove if the value of noise is bigger than 100
    print('noisy samples...', end='', flush=True)
    valid_mask = (np.nanmax(np.abs(x), axis=2) < 100) # noisy sample 
    '''
    x = x[valid_mask, :]  # To use CNN, it should be three-dimensional. Therefore, add the dimension.
    y = y[valid_mask]
    b = b[valid_mask]
    c = c[valid_mask]
    '''
    print('{:.1f}% removed'.format(100*(1-np.mean(valid_mask))))


    def consecutive(data, stepsize=0, threshold=100):
        con_list = np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
        con_count = [d for d in con_list if len(d)>threshold]
        return con_count

    sec= 0
    psd_ms = []
    for i in range(len(x)):
        eeg = x[i]
        bis = b[i]
        ppf = y[i]
        psds = []
        #freqs=[]
        # Calculate PS
        for j in range(0, SEGLEN-NFFT, SLIDE):
            if len(eeg[EEG1, j:j+NFFT]) < NFFT/2:
                continue
            if len(eeg[EEG2, j:j+NFFT]) < NFFT/2:
                continue

            nall = len(consecutive(eeg[EEG1, j:j+NFFT], stepsize=0, threshold=SRATE*10))
            nall2 = len(consecutive(eeg[EEG2, j:j+NFFT], stepsize=0, threshold=SRATE*10))
            if nall > 1 or nall2 > 1:
                continue
            
            
            psd, freq = psd_array_multitaper(eeg[EEG1, j:j+NFFT], SRATE, adaptive=True, normalization='full', verbose=0)  # https://raphaelvallat.com/bandpower.html
            psd2, freq2 = psd_array_multitaper(eeg[EEG2, j:j+NFFT], SRATE, adaptive=True, normalization='full', verbose=0)  # https://raphaelvallat.com/bandpower.html
            psd = np.vstack([psd, psd2])
            psds.append(psd)
            #freqs.append(freq)
            #biss.append(bis[j:j+NFFT])
        
        # Convert power to dB scale
        psds = 10 * np.log10(psds)
        # get median of psds
        psd_m = np.median(psds, axis=0)
        #print(psd_m.shape)
        psd_ms.append(psd_m)
        data_series = pd.DataFrame(psd_m)
        #smoothed = data_series.rolling(window=10).mean()
        #print(smoothed.shape)


        
        if sec % 60 == 0:
            # plot all psds and median
            plt.figure(figsize=(6, 6))
            plt.plot(freq, psd_m[EEG1, :], label='median_EEG1', color='red')
            plt.plot(freq, psd_m[EEG2, :], label='median_EEG2', color='green')
            #plt.plot(freq, smoothed[EEG1, :], label='median_EEG1_smooth', color='blue')
            #plt.plot(freq, smoothed[EEG2, :], label='median_EEG2_smooth', color='yellow')
            # for psd in psds:
            #     plt.plot(freq, psd, alpha=0.1, color='blue')
            plt.legend(loc="upper left")
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power spectral density (dB)')
            plt.title(f'case{c[i]}_{sec}sec_bis{bis}_ppfce{ppf}_psd')
            plt.tight_layout() 
            plt.ylim(-40, 40) 
            plt.xlim(1, 40) 
            plt.savefig(f'./plot/case{c[i]}_{sec}sec_bis{bis:.1f}_ppfce{ppf:.1f}_psd.png')
            
            plt.close()

        sec+=SLIDE / SRATE

    np.savez(cachefile2, psd=psd_ms, y=y, b=b, c=c)

# Set seed
random.seed(42)

# caseid
caseids = list(np.unique(c))
random.shuffle(caseids)

# Split dataset into training and testing data
ntest = max(1, int(len(caseids) * 0.2))
caseids_train = caseids[ntest:]
caseids_test = caseids[:ntest]

# Truncate the frequency range from 1 Hz to 40 Hz
index_1hz = int(round(1 / (SRATE / NFFT)))
index_40hz = int(round(40 / (SRATE / NFFT))) + 110 # Add 110 to make 1280 points
x = psd_ms[:, :, index_1hz:index_40hz]
print(f'Truncated from {NFFT} to {index_40hz - index_1hz} points')

train_mask = np.isin(c, caseids_train)
test_mask = np.isin(c, caseids_test)
x_train = x[train_mask]
y_train = y[train_mask]
x_test = x[test_mask]
y_test = y[test_mask]
b_test = b[test_mask]
c_test = c[test_mask]

print('====================================================')
print('total: {} cases {} samples'.format(len(caseids), len(y)))
print('train: {} cases {} samples'.format(len(np.unique(c[train_mask])), len(y_train)))
print('test {} cases {} samples'.format(len(np.unique(c_test)), len(y_test)))
print('====================================================')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Convert Numpy arrays to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32) #.unsqueeze(1)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32) #.unsqueeze(1)



# Create DataLoader objects
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize CRNN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device : {device}')
n_classes = 1  # Number of classes for classification
in_channels = x.shape[1]  # Number of EEG channels
out_channels = 64  # Number of output channels after the CNN
att_channels=16
n_len_seg = 128 # Segment length for RNN, adjust based on your needs

#model = CRNN(in_channels, out_channels, n_len_seg, n_classes, device, verbose=False).to(device)
model = ACNN(in_channels, out_channels, att_channels, n_len_seg, n_classes, device, verbose=False).to(device)

#MSE error
criterion = nn.MSELoss()
#criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
n_epochs = 200  # You can adjust this
for epoch in range(1, n_epochs + 1):
    model.train()
    train_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    # Log training information
    train_loss /= len(train_loader)
    print(f"Epoch: {epoch}, Training Loss: {train_loss:.6f}")

# Test the model and compute AUROC
model.eval()
test_loss = 0.0
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item()

        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        all_preds.append(output.cpu().numpy())
        all_labels.append(target.cpu().numpy())

test_loss /= len(test_loader)
accuracy = 100 * correct / total
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

print(f"Test Loss: {test_loss:.6f}, Test Accuracy: {accuracy:.2f}%")

# make float to int
all_preds = np.round(all_preds, decimals=0)
all_labels = np.round(all_labels, decimals=0)
print(f'Label unique : {np.unique(all_labels)}')

# Calculate AUROC for each class
n_classes = 6
auroc_list = []
for i in range(n_classes):
    try:
        auroc = roc_auc_score(all_labels == i, all_preds[:, i])
        auroc_list.append(auroc)
        print(f"Class {i} AUROC: {auroc:.4f}")
    except:
        print(f"Class {i} AUROC: nan")

mean_auroc = np.mean(auroc_list)
print(f"Mean AUROC: {mean_auroc:.4f}")



# Save the model
torch.save(model.state_dict(), 'best_model.pth')
print("Model saved as best_model.pth")
