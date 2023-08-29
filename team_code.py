#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, os, sys
import mne
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
from mne.time_frequency import psd_array_multitaper
from sklearn.preprocessing import StandardScaler

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print('Finding the Challenge data...')

    patient_ids = find_data_folders(data_folder) #************
    #patient_ids2 = find_data_folders(data_folder)[70:100]
    #patient_ids = np.concatenate([patient_ids, patient_ids2])
    num_patients = len(patient_ids)

    if num_patients==0:
        raise FileNotFoundError('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    pt_list = list()
    psd_list = list()
    outcomes = list()
    cpcs = list()

    for i in range(num_patients):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patients))
        print(f'paient id : {patient_ids[i]}')
        patient_features, eeg_features= get_features(data_folder, patient_ids[i])
        #features.append(current_features)
        pt_list.append(patient_features)
        psd_list.append(eeg_features)

        # Extract labels.
        patient_metadata = load_challenge_data(data_folder, patient_ids[i])
        current_outcome = get_outcome(patient_metadata)
        outcomes.append(current_outcome)
        current_cpc = get_cpc(patient_metadata)
        cpcs.append(current_cpc)

    #features = np.vstack(features)
    pt_list = np.vstack(pt_list)
    psd_list = np.stack(psd_list, axis=0)
    outcomes = np.vstack(outcomes)
    cpcs = np.vstack(cpcs)

    # Remove missing values
    valid_mask = ~(np.max(np.isnan(psd_list), axis=2) > 0) # nan이 있으면 제거
    valid_mask = valid_mask[:,0] & valid_mask[:,1]

    pt_list = pt_list[valid_mask]
    psd_list = psd_list[valid_mask]
    outcomes = outcomes[valid_mask]
    cpcs = cpcs[valid_mask]    

    # Impute any missing features; use the mean value by default.
    #imputer = SimpleImputer().fit(features)
    imputer = KNNImputer(n_neighbors=10).fit(pt_list)

    # Train the models.
    pt_list = imputer.transform(pt_list)

    # Train the models.
    if verbose >= 1:
        print('Training the Challenge model on the Challenge data...')
    
    # Standard scaling for patient meta data
    pt_scaler = StandardScaler()
    pt_list = pt_scaler.fit_transform(pt_list)

    # standard scaling for psd data
    mean_psds = np.mean(psd_list) #, axis=0, keepdims=True)
    std_psds = np.std(psd_list) #, axis=0, keepdims=True)
    psd_list = (psd_list - mean_psds) / std_psds

    #1 Classification model
    train_model(pt_list, psd_list, outcomes, model_folder, pretrained=True, cpc=False)

    #2 CPC model
    train_model(pt_list, psd_list, cpcs, model_folder, pretrained=True, cpc=True)

    
    # Assume eeg_data.shape = (n_samples, n_channels, n_timepoints)
    # Assume meta_data.shape = (n_samples, n_meta_features)
    # Assume labels.shape = (n_samples,)

    # Save the models.
    #save_challenge_model(model_folder, imputer, outcome_model, cpc_model, pt_scaler, mean_psds, std_psds)
    save_challenge_model(model_folder, imputer, pt_scaler, mean_psds, std_psds)

    if verbose >= 1:
        print('Done.')

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose):
    filename = os.path.join(model_folder, 'models.sav')
    return joblib.load(filename)

# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):
    imputer = models['imputer']

    # 'pt_scaler': pt_scaler, 'mean_psds': mean_psds, 'std_psds': std_psds}
    
    pt_scaler = models['pt_scaler']
    mean_psds = models['mean_psds']
    std_psds = models['std_psds']
    model_folder = models['model_folder']
    
    #outcome_model = models['outcome_model']
    #cpc_model = models['cpc_model']


    # Extract features.
    patient_features, eeg_features = get_features(data_folder, patient_id)
    #features = features.reshape(1, -1)

    # Impute missing data.
    patient_features = imputer.transform([patient_features])
    patient_features = pt_scaler.transform(patient_features)

    eeg_features = (eeg_features - mean_psds) / std_psds

    outcome_model_path = os.path.join(model_folder, 'best_model_outcome.pth')
    outcome_probability = test_model(patient_features, eeg_features, outcome_model_path, cpc=False)
    threshold = 0.5
    outcome = (outcome_probability >= threshold).astype(int)

    cpc_model_path = os.path.join(model_folder, 'best_model_cpc.pth')
    cpc = test_model(patient_features, eeg_features, cpc_model_path, cpc=True)
    ## Apply models to features.
    #outcome = outcome_model.predict(features)[0]
    #outcome_probability = outcome_model.predict_proba(features)[0, 1]
    #cpc = cpc_model.predict(features)[0]

    # Ensure that the CPC score is between (or equal to) 1 and 5.
    cpc = np.clip(cpc, 1, 5)

    return outcome, outcome_probability, cpc

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
def consecutive(data, stepsize=0, threshold=100):
    con_list = np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
    con_count = [d for d in con_list if len(d)>threshold]
    return con_count
    
################################################################################

'''
# Save your trained model.
def save_challenge_model(model_folder, imputer, outcome_model, cpc_model):
    d = {'imputer': imputer, 'outcome_model': outcome_model, 'cpc_model': cpc_model}
    filename = os.path.join(model_folder, 'models.sav')
    joblib.dump(d, filename, protocol=0)
'''
def save_challenge_model(model_folder, imputer, pt_scaler, mean_psds, std_psds):
    d = {'imputer': imputer, 'pt_scaler': pt_scaler, 'mean_psds': mean_psds, 'std_psds': std_psds, 'model_folder': model_folder}
    filename = os.path.join(model_folder, 'models.sav')
    joblib.dump(d, filename, protocol=0)


# Preprocess data.
def preprocess_data(data, sampling_frequency, utility_frequency):
    # Define the bandpass frequencies.
    #passband = [0.1, 30.0]

    # Promote the data to double precision because these libraries expect double precision.
    data = np.asarray(data, dtype=np.float64)

    #print('baseline drift...', end='', flush=True)
    data[0, :] -= scipy.signal.savgol_filter(data[0, :], 91, 3)  # remove baseline drift
    data[1, :] -= scipy.signal.savgol_filter(data[1, :], 91, 3)  # remove baseline drift
    #print('removed')

    '''
    # If the utility frequency is between bandpass frequencies, then apply a notch filter.
    if utility_frequency is not None and passband[0] <= utility_frequency <= passband[1]:
        data = mne.filter.notch_filter(data, sampling_frequency, utility_frequency, n_jobs=4, verbose='error')

    # Apply a bandpass filter.
    data = mne.filter.filter_data(data, sampling_frequency, passband[0], passband[1], n_jobs=4, verbose='error')
    '''
    # Resample the data.
    if sampling_frequency % 2 == 0:
        resampling_frequency = 128
    else:
        resampling_frequency = 125
    lcm = np.lcm(int(round(sampling_frequency)), int(round(resampling_frequency)))
    up = int(round(lcm / sampling_frequency))
    down = int(round(lcm / resampling_frequency))
    resampling_frequency = sampling_frequency * up / down
    data = scipy.signal.resample_poly(data, up, down, axis=1)

    '''
    # Scale the data to the interval [-1, 1].
    min_value = np.min(data)
    max_value = np.max(data)
    if min_value != max_value:
        data = 2.0 / (max_value - min_value) * (data - 0.5 * (min_value + max_value))
    else:
        data = 0 * data
    '''

    return data, resampling_frequency

# Extract features.
def get_features(data_folder, patient_id):
    # Load patient data.
    patient_metadata = load_challenge_data(data_folder, patient_id)
    recording_ids = find_recording_files(data_folder, patient_id)
    num_recordings = len(recording_ids)

    # Extract patient features.
    patient_features = get_patient_features(patient_metadata)

    # Extract EEG features.
    eeg_channels = ['F7', 'F3']#['F3', 'P3', 'F4', 'P4']
    group = 'EEG'

    if num_recordings > 0:
        recording_id = recording_ids[-1] ##recording_ids = ['0284_001_004', '0284_002_005', ... '0284_085_074']
        recording_location = os.path.join(data_folder, patient_id, '{}_{}'.format(recording_id, group))
        if os.path.exists(recording_location + '.hea'):
            data, channels, sampling_frequency = load_recording_data(recording_location)
            utility_frequency = get_utility_frequency(recording_location + '.hea')

            if all(channel in channels for channel in eeg_channels):
                data, channels = reduce_channels(data, channels, eeg_channels)
                data, sampling_frequency = preprocess_data(data, sampling_frequency, utility_frequency)
                #data = np.array([data[0, :] - data[1, :], data[2, :] - data[3, :]]) # Convert to bipolar montage: F3-P3 and F4-P4
                eeg_features = get_eeg_features(data, sampling_frequency) #.flatten()
            else:
                eeg_features = float('nan') * np.ones(1280) # 2 bipolar channels * 4 features / channel
        else:
            eeg_features = float('nan') * np.ones(1280) # 2 bipolar channels * 4 features / channel
    else:
        eeg_features = float('nan') * np.ones(1280) # 2 bipolar channels * 4 features / channel

    '''
    # Extract ECG features.
    ecg_channels = ['ECG', 'ECGL', 'ECGR', 'ECG1', 'ECG2']
    group = 'ECG'

    if num_recordings > 0:
        recording_id = recording_ids[0]
        recording_location = os.path.join(data_folder, patient_id, '{}_{}'.format(recording_id, group))
        if os.path.exists(recording_location + '.hea'):
            data, channels, sampling_frequency = load_recording_data(recording_location)
            utility_frequency = get_utility_frequency(recording_location + '.hea')

            data, channels = reduce_channels(data, channels, ecg_channels)
            data, sampling_frequency = preprocess_data(data, sampling_frequency, utility_frequency)
            features = get_ecg_features(data)
            ecg_features = expand_channels(features, channels, ecg_channels).flatten()
        else:
            ecg_features = float('nan') * np.ones(10) # 5 channels * 2 features / channel
    else:
        ecg_features = float('nan') * np.ones(10) # 5 channels * 2 features / channel
    '''
    # Extract features.
    #return np.hstack((patient_features, eeg_features, ecg_features))
    return patient_features, eeg_features # (8,1280)

# Extract patient features from the data.
def get_patient_features(data):
    age = get_age(data)
    sex = get_sex(data)
    rosc = get_rosc(data)
    ohca = get_ohca(data)
    shockable_rhythm = get_shockable_rhythm(data)
    ttm = get_ttm(data)

    sex_features = np.zeros(2, dtype=int)
    if sex == 'Female':
        female = 1
        male   = 0
        other  = 0
    elif sex == 'Male':
        female = 0
        male   = 1
        other  = 0
    else:
        female = 0
        male   = 0
        other  = 1

    features = np.array((age, female, male, other, rosc, ohca, shockable_rhythm, ttm))

    return features

# Extract features from the EEG data.
def get_eeg_features(data, sampling_frequency):
    num_channels, num_samples = np.shape(data)
    
    SEGLEN = int(5 * 60 * sampling_frequency)
    NFFT = int(30 * sampling_frequency)
    SLIDE = int(10 * sampling_frequency)
    EEG1 = 0
    EEG2 = 1
    '''
    if num_samples > 0:
        delta_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency,  fmin=0.5,  fmax=8.0, verbose=False)
        theta_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency,  fmin=4.0,  fmax=8.0, verbose=False)
        alpha_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency,  fmin=8.0, fmax=12.0, verbose=False)
        beta_psd,  _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency, fmin=12.0, fmax=30.0, verbose=False)

        delta_psd_mean = np.nanmean(delta_psd, axis=1)
        theta_psd_mean = np.nanmean(theta_psd, axis=1)
        alpha_psd_mean = np.nanmean(alpha_psd, axis=1)
        beta_psd_mean  = np.nanmean(beta_psd,  axis=1)
    else:
        delta_psd_mean = theta_psd_mean = alpha_psd_mean = beta_psd_mean = float('nan') * np.ones(num_channels)

    features = np.array((delta_psd_mean, theta_psd_mean, alpha_psd_mean, beta_psd_mean)).T
    '''
    psds = []
    # Calculate PS
    for j in range(0, SEGLEN-NFFT, SLIDE):
        if len(data[EEG1, j:j+NFFT]) < NFFT/2:
            continue
        if len(data[EEG2, j:j+NFFT]) < NFFT/2:
            continue

        nall = len(consecutive(data[EEG1, j:j+NFFT], stepsize=0, threshold=sampling_frequency*10))
        nall2 = len(consecutive(data[EEG2, j:j+NFFT], stepsize=0, threshold=sampling_frequency*10))
        if nall > 1 or nall2 > 1:
            continue
        
        
        psd, freq = psd_array_multitaper(data[EEG1, j:j+NFFT], sampling_frequency, adaptive=True, normalization='full', verbose=50)
        psd2, freq2 = psd_array_multitaper(data[EEG2, j:j+NFFT], sampling_frequency, adaptive=True, normalization='full', verbose=50)  
        psd = np.vstack([psd, psd2])
        psds.append(psd)
        #freqs.append(freq)
        #biss.append(bis[j:j+NFFT])
    
    psds = [p for p in psds if p.shape == (2,1921)]    
    
    if len(psds) > 1:
        psds = 10 * np.log10(psds)

        # Truncate the frequency range from 1 Hz to 40 Hz
        index_1hz = int(round(1 / (sampling_frequency / NFFT)))
        index_40hz = int(round(40 / (sampling_frequency / NFFT))) + 110 # Add 110 to make 1280 points
        
        if len(psds) > 1:
            # get median of psds
            psd_m = np.median(psds, axis=0)
            psd_m = psd_m[:, index_1hz:index_40hz]

    else:
        psd_m = float('nan') * np.ones((2, 1280))

    return psd_m
    
# Extract features from the ECG data.
def get_ecg_features(data):
    num_channels, num_samples = np.shape(data)

    if num_samples > 0:
        mean = np.mean(data, axis=1)
        std  = np.std(data, axis=1)
    elif num_samples == 1:
        mean = np.mean(data, axis=1)
        std  = float('nan') * np.ones(num_channels)
    else:
        mean = float('nan') * np.ones(num_channels)
        std = float('nan') * np.ones(num_channels)

    features = np.array((mean, std)).T

    return features

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from acnn1d import ACNN
import sklearn
from sklearn.model_selection import train_test_split
# Define custom dataset
class MultiModalDataset(Dataset):
    def __init__(self, eeg_data, meta_data, labels):
        self.eeg_data = eeg_data
        self.meta_data = meta_data
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return self.eeg_data[index], self.meta_data[index], self.labels[index]


# Define custom dataset
class TestDataset(Dataset):
    def __init__(self, eeg_data, meta_data):
        self.eeg_data = eeg_data
        self.meta_data = meta_data
    
    def __len__(self):
        return len(self.meta_data)
    
    def __getitem__(self, index):
        return self.eeg_data[index], self.meta_data[index]


# Your pre-trained EEG Model, but without the last layer
class Modified1DCNN(nn.Module):
    def __init__(self, original_model):
        super(Modified1DCNN, self).__init__()
        # Remove the last layer (Assuming sequential model)
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        print(list(original_model.children()))
    def forward(self, x):
        x = self.features(x)
        return x

# Multi-modal Model
class MultiModalModel(nn.Module):
    def __init__(self, eeg_model, meta_input_dim):
        super(MultiModalModel, self).__init__()
        self.eeg_model = eeg_model
        self.meta_model = nn.Sequential(
            nn.Linear(meta_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        #self.classifier = nn.Linear(64 + eeg_model.output_dim, 1)
        self.classifier = nn.Linear(16 + 64, 1)
    
    def forward(self, eeg_data, meta_data):
        x1 = self.eeg_model(eeg_data)
        x2 = self.meta_model(meta_data)
        #print(x1.shape, x2.shape)

        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(x)
        return x

def train_model(pt_list, psd_list, label, model_folder=None, pretrained=False, cpc=False):

    # Split dataset into training and testing data
    #ntest = max(1, int(len(ptlist) * 0.1))
    #pt_train, psd_train, y_train = pt_list[ntest:], psd_list[ntest:], label[ntest:]
    #pt_test, psd_test, y_test = pt_list[:ntest], psd_list[:ntest], label[:ntest]

    if cpc:
        stratify = None
    else:
        stratify = label
    pt_train, pt_val, psd_train, psd_val, y_train, y_val = train_test_split(pt_list, psd_list, label, test_size = 0.15, random_state=42, stratify=stratify)

    pt_train = torch.tensor(pt_train, dtype=torch.float32)
    psd_train = torch.tensor(psd_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    pt_val = torch.tensor(pt_val, dtype=torch.float32)
    psd_val = torch.tensor(psd_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    train_dataset = MultiModalDataset(psd_train, pt_train, y_train)
    val_dataset = MultiModalDataset(psd_val, pt_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    

    # Initialize CRNN model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device : {device}')

    n_classes = 1 #len(np.unique(label)) # Number of classes for classification
    in_channels = 2 #psd_list.shape[1]  # Number of EEG channels
    out_channels = 64  # Number of output channels after the CNN
    att_channels=16
    n_len_seg = 128 # Segment length for RNN, adjust based on your needs


    # Criterion and optimizer
    criterion = nn.BCEWithLogitsLoss()
    #scoring = True

    if cpc:
        criterion = nn.MSELoss()
        #scoring = False

    base_model = ACNN(in_channels, out_channels, att_channels, n_len_seg, n_classes, device, last_layer=False)
    if pretrained:
        base_model.load_state_dict(torch.load("./pretrain_model.pth"))

    ## Create the modified model
    #modified_eeg_model = Modified1DCNN(original_eeg_model)
    base_model.last_layer = False

    meta_input_dim = pt_list.shape[1]
    model = MultiModalModel(base_model, meta_input_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model = model.to(device)
    # Early stopping parameters
    patience = 10  # Number of epochs with no improvement to wait
    best_val_loss = float('inf')
    best_score = float('-inf')
    counter = 0

    # Training loop
    n_epochs = 500
    for epoch in range(1, n_epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_idx, (psd_data, pt_data, target) in enumerate(train_loader):
            psd_data, pt_data, target = psd_data.to(device), pt_data.to(device), target.to(device).float() #.unsqueeze(1)
            optimizer.zero_grad()
            output = model(psd_data, pt_data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        print(f"Epoch: {epoch}, Training Loss: {train_loss:.6f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for psd_data, pt_data, target in val_loader:
                psd_data, pt_data, target = psd_data.to(device), pt_data.to(device), target.to(device).float() #.unsqueeze(1)
                output = model(psd_data, pt_data)
                loss = criterion(output, target)
                val_loss += loss.item()
                all_labels.append(target.cpu().numpy())

                if not cpc:
                    output = torch.sigmoid(output)
                all_predictions.append(torch.sigmoid(output).cpu().numpy())
        
        val_loss /= len(val_loader)
        all_labels = np.vstack(all_labels)
        all_predictions = np.vstack(all_predictions)
        #auroc = roc_auc_score(all_labels, all_predictions)
        if not cpc:
            val_score = compute_score(all_labels, all_predictions)
            #print(all_predictions)
            print(f"Validation Loss: {val_loss:.6f}, SCORE: {val_score:.6f}")

            # Check for early stopping
            if val_score > best_score:
                best_score = val_score
                counter = 0
                torch.save(model.state_dict(), os.path.join(model_folder, 'best_model_outcome.pth'))
                print("Saved best model")
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping triggered")
                    break

        else:
            print(f"Validation Loss: {val_loss:.6f}")
            
            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                torch.save(model.state_dict(), os.path.join(model_folder, 'best_model_cpc.pth'))
                print("Saved best model")
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping triggered")
                    break

def test_model(pt_list, psd_list, modelpath=None, cpc=False):
    
    # Split dataset into training and testing data
    #ntest = max(1, int(len(ptlist) * 0.1))
    #pt_train, psd_train, y_train = pt_list[ntest:], psd_list[ntest:], label[ntest:]
    #pt_test, psd_test, y_test = pt_list[:ntest], psd_list[:ntest], label[:ntest]
    #print(psd_list.shape)

    pt_list = torch.tensor(pt_list, dtype=torch.float32)
    psd_list = torch.tensor(psd_list, dtype=torch.float32).unsqueeze(0)

    test_dataset = TestDataset(psd_list, pt_list)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # Initialize CRNN model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f'Device : {device}')

    #n_classes = len(np.unique(label)) # Number of classes for classification
    n_classes = 1 ## Never mind
    in_channels = 2 #psd_list.shape[1]  # Number of EEG channels
    out_channels = 64  # Number of output channels after the CNN
    att_channels=16
    n_len_seg = 128 # Segment length for RNN, adjust based on your needs


    base_model = ACNN(in_channels, out_channels, att_channels, n_len_seg, n_classes, device, last_layer=False)
    base_model.last_layer = False
    meta_input_dim = pt_list.shape[1]
    model = MultiModalModel(base_model, meta_input_dim)
    model.load_state_dict(torch.load(modelpath))
    model.eval()  # Switch to evaluation mode
    model = model.to(device)  # Move model to GPU if available

    # Initialize a list to store your predictions
    predictions = []
    # Evaluation loop
    with torch.no_grad():
        for batch_idx, (psd_data, pt_data) in enumerate(test_loader):
            #data = data[0].to(device)  # If there's only one element in your dataset, index into the tuple
            #print(psd_data.shape)
            #print(psd_data.unsqueeze(1).shape)
            psd_data, pt_data = psd_data.to(device), pt_data.to(device)
            output = model(psd_data, pt_data)
            if not cpc:
                output = torch.sigmoid(output)  # If your output is a logit and you want a probability
            output = output.cpu().numpy()  # Move output back to CPU and convert to NumPy array
            predictions.extend(output)

    # Convert predictions to a NumPy array
    predictions = np.array(predictions)

    return predictions
## Define parameters for random forest classifier and regressor.
    #n_estimators   = 123  # Number of trees in the forest.
    #max_leaf_nodes = 456  # Maximum number of leaf nodes in each tree.
    #random_state   = 789  # Random state; set for reproducibility.

    #outcome_model = RandomForestClassifier(
    #    n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, outcomes.ravel())
    #cpc_model = RandomForestRegressor(
    #    n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, cpcs.ravel())

 # Compute the Challenge score.
def compute_score(labels, outputs):
    # Check the data.
    assert len(labels) == len(outputs)

    # Convert the data to NumPy arrays for easier indexing.
    current_labels = np.asarray(labels.flatten(), dtype=np.float64)
    current_outputs = np.asarray(outputs.flatten(), dtype=np.float64)

    # # Identify the unique hospitals.
    # unique_hospitals = sorted(set(hospitals))
    # num_hospitals = len(unique_hospitals)

    # # Initialize a confusion matrix for each hospital.
    # tps = np.zeros(num_hospitals)
    # fps = np.zeros(num_hospitals)
    # fns = np.zeros(num_hospitals)
    # tns = np.zeros(num_hospitals)
    tps = 0
    fps = 0
    fns = 0
    tns = 0

    # Compute the confusion matrix at each output threshold separately for each hospital.
    #for i, hospital in enumerate(unique_hospitals):
    #    idx = [j for j, x in enumerate(hospitals) if x == hospital]
    #    current_labels = labels[idx]
    #    current_outputs = outputs[idx]
    num_instances = len(current_labels)

    # Collect the unique output values as the thresholds for the positive and negative classes.
    thresholds = np.unique(current_outputs)
    thresholds = np.append(thresholds, thresholds[-1]+1)
    thresholds = thresholds[::-1]
    num_thresholds = len(thresholds)

    idx = np.argsort(current_outputs)[::-1]

    # Initialize the TPs, FPs, FNs, and TNs with no positive outputs.
    tp = np.zeros(num_thresholds)
    fp = np.zeros(num_thresholds)
    fn = np.zeros(num_thresholds)
    tn = np.zeros(num_thresholds)

    tp[0] = 0
    fp[0] = 0
    fn[0] = np.sum(current_labels == 1)
    tn[0] = np.sum(current_labels == 0)

    # Update the TPs, FPs, FNs, and TNs using the values at the previous threshold.
    k = 0
    for l in range(1, num_thresholds):
        tp[l] = tp[l-1]
        fp[l] = fp[l-1]
        fn[l] = fn[l-1]
        tn[l] = tn[l-1]

        while k < num_instances and current_outputs[idx[k]] >= thresholds[l]:
            if current_labels[idx[k]] == 1:
                tp[l] += 1
                fn[l] -= 1
            else:
                fp[l] += 1
                tn[l] -= 1
            k += 1

    # Compute the FPRs.
    fpr = np.zeros(num_thresholds)
    for l in range(num_thresholds):
        if tp[l] + fn[l] > 0:
            fpr[l] = float(fp[l]) / float(tp[l] + fn[l])
        else:
            fpr[l] = float('nan')

    # Find the threshold such that FPR <= 0.05.
    max_fpr = 0.05
    if np.any(fpr <= max_fpr):
        l = max(l for l, x in enumerate(fpr) if x <= max_fpr)
        tps = tp[l]
        fps = fp[l]
        fns = fn[l]
        tns = tn[l]
    else:
        tps = tp[0]
        fps = fp[0]
        fns = fn[0]
        tns = tn[0]

    # Compute the TPR at FPR <= 0.05 for each hospital.
    tp = tps
    fp = fps
    fn = fns
    tn = tns

    print(f'tp / fp / fn / tn : {tp, fp, fn, tn}')
    if tp + fn > 0:
        max_tpr = tp / (tp + fn)
    else:
        max_tpr = float('nan')

    return max_tpr
