#!/usr/bin/env python

# Do *not* edit this script.
# These are helper functions that you can use with your code.
# Check the example code to see how to import these functions to your code.

import os, numpy as np, scipy as sp, scipy.io

### Challenge data I/O functions

# Find the folders with data files.
def find_data_folders(root_folder):
    data_folders = list()
    for x in sorted(os.listdir(root_folder)):
        data_folder = os.path.join(root_folder, x)
        if os.path.isdir(data_folder):
            data_file = os.path.join(data_folder, x + '.txt')
            if os.path.isfile(data_file):
                data_folders.append(x)
    return sorted(data_folders)

# Load the patient metadata: age, sex, etc.
def load_challenge_data(data_folder, patient_id):
    patient_metadata_file = os.path.join(data_folder, patient_id, patient_id + '.txt')
    patient_metadata = load_text_file(patient_metadata_file)
    return patient_metadata

# Find the record names.
def find_recording_files(data_folder, patient_id):
    record_names = set()
    patient_folder = os.path.join(data_folder, patient_id)
    for file_name in sorted(os.listdir(patient_folder)):
        if not file_name.startswith('.') and file_name.endswith('.hea'):
            root, ext = os.path.splitext(file_name)
            record_name = '_'.join(root.split('_')[:-1])
            record_names.add(record_name)
    return sorted(record_names)

# Load the WFDB data for the Challenge (but not all possible WFDB files).
def load_recording_data(record_name, check_values=False):
    # Allow either the record name or the header filename.
    root, ext = os.path.splitext(record_name)
    if ext=='':
        header_file = record_name + '.hea'
    else:
        header_file = record_name

    # Load the header file.
    if not os.path.isfile(header_file):
        raise FileNotFoundError('{} recording not found.'.format(record_name))

    with open(header_file, 'r') as f:
        header = [l.strip() for l in f.readlines() if l.strip()]

    # Parse the header file.
    record_name = None
    num_signals = None
    sampling_frequency = None
    num_samples = None
    signal_files = list()
    gains = list()
    offsets = list()
    channels = list()
    initial_values = list()
    checksums = list()

    for i, l in enumerate(header):
        arrs = [arr.strip() for arr in l.split(' ')]
        # Parse the record line.
        if i==0:
            record_name = arrs[0]
            num_signals = int(arrs[1])
            sampling_frequency = float(arrs[2])
            num_samples = int(arrs[3])
        # Parse the signal specification lines.
        elif not l.startswith('#') or len(l.strip()) == 0:
            signal_file = arrs[0]
            gain = float(arrs[2].split('/')[0])
            offset = int(arrs[4])
            initial_value = int(arrs[5])
            checksum = int(arrs[6])
            channel = arrs[8]
            signal_files.append(signal_file)
            gains.append(gain)
            offsets.append(offset)
            initial_values.append(initial_value)
            checksums.append(checksum)
            channels.append(channel)

    # Check that the header file only references one signal file. WFDB format allows for multiple signal files, but, for
    # simplicity, we have not done that here.
    num_signal_files = len(set(signal_files))
    if num_signal_files!=1:
        raise NotImplementedError('The header file {}'.format(header_file) \
            + ' references {} signal files; one signal file expected.'.format(num_signal_files))

    # Load the signal file.
    head, tail = os.path.split(header_file)
    signal_file = os.path.join(head, list(signal_files)[0])
    data = np.asarray(sp.io.loadmat(signal_file)['val'])

    # Check that the dimensions of the signal data in the signal file is consistent with the dimensions for the signal data given
    # in the header file.
    num_channels = len(channels)
    if np.shape(data)!=(num_channels, num_samples):
        raise ValueError('The header file {}'.format(header_file) \
            + ' is inconsistent with the dimensions of the signal file.')

    # Check that the initial value and checksums in the signal file are consistent with the initial value and checksums in the
    # header file.
    if check_values:
        for i in range(num_channels):
            if data[i, 0]!=initial_values[i]:
                raise ValueError('The initial value in header file {}'.format(header_file) \
                    + ' is inconsistent with the initial value for channel {} in the signal data'.format(channels[i]))
            if np.sum(data[i, :], dtype=np.int16)!=checksums[i]:
                raise ValueError('The checksum in header file {}'.format(header_file) \
                    + ' is inconsistent with the checksum value for channel {} in the signal data'.format(channels[i]))

    # Rescale the signal data using the gains and offsets.
    rescaled_data = np.zeros(np.shape(data), dtype=np.float32)
    for i in range(num_channels):
        rescaled_data[i, :] = (np.asarray(data[i, :], dtype=np.float64) - offsets[i]) / gains[i]

    return rescaled_data, channels, sampling_frequency

# Choose the channels.
def reduce_channels(current_data, current_channels, requested_channels):
    if current_channels == requested_channels:
        reduced_data = current_data
        reduced_channels = current_channels
    else:
        reduced_indices = [current_channels.index(channel) for channel in requested_channels if channel in current_channels]
        reduced_channels = [current_channels[i] for i in reduced_indices]
        reduced_data = current_data[reduced_indices, :]
    return reduced_data, reduced_channels

# Choose the channels.
def expand_channels(current_data, current_channels, requested_channels):
    if current_channels == requested_channels:
        expanded_data = current_data
    else:
        num_current_channels, num_samples = np.shape(current_data)
        num_requested_channels = len(requested_channels)
        expanded_data = np.zeros((num_requested_channels, num_samples))
        for i, channel in enumerate(requested_channels):
            if channel in current_channels:
                j = current_channels.index(channel)
                expanded_data[i, :] = current_data[j, :]
            else:
                expanded_data[i, :] = float('nan')
    return expanded_data

### Helper Challenge data I/O functions

# Load text file as a string.
def load_text_file(filename):
    with open(filename, 'r') as f:
        data = f.read()
    return data

# Get a variable from the patient metadata.
def get_variable(text, variable_name, variable_type):
    variable = None
    for l in text.split('\n'):
        if l.startswith(variable_name):
            variable = ':'.join(l.split(':')[1:]).strip()
            variable = cast_variable(variable, variable_type)
            return variable

# Get the patient ID variable from the patient data.
def get_patient_id(string):
    return get_variable(string, 'Patient', str)

# Get the patient ID variable from the patient data.
def get_hospital(string):
    return get_variable(string, 'Hospital', str)

# Get the age variable (in years) from the patient data.
def get_age(string):
    return get_variable(string, 'Age', int)

# Get the sex variable from the patient data.
def get_sex(string):
    return get_variable(string, 'Sex', str)

# Get the ROSC variable (in minutes) from the patient data.
def get_rosc(string):
    return get_variable(string, 'ROSC', int)

# Get the OHCA variable from the patient data.
def get_ohca(string):
    return get_variable(string, 'OHCA', bool)

# Get the shockable rhythm variable from the patient data.
def get_shockable_rhythm(string):
    return get_variable(string, 'Shockable Rhythm', bool)

# Get the TTM variable (in Celsius) from the patient data.
def get_ttm(string):
    return get_variable(string, 'TTM', int)

# Get the Outcome variable from the patient data.
def get_outcome(string):
    variable = get_variable(string, 'Outcome', str)
    if variable is None or is_nan(variable):
        raise ValueError('No outcome available. Is your code trying to load labels from the hidden data?')
    if variable == 'Good':
        variable = 0
    elif variable == 'Poor':
        variable = 1
    return variable

# Get the Outcome probability variable from the patient data.
def get_outcome_probability(string):
    variable = sanitize_scalar_value(get_variable(string, 'Outcome Probability', str))
    if variable is None or is_nan(variable):
        raise ValueError('No outcome available. Is your code trying to load labels from the hidden data?')
    return variable

# Get the CPC variable from the patient data.
def get_cpc(string):
    variable = sanitize_scalar_value(get_variable(string, 'CPC', str))
    if variable is None or is_nan(variable):
        raise ValueError('No CPC score available. Is your code trying to load labels from the hidden data?')
    return variable

# Get the utility frequency (in Hertz) from the recording data.
def get_utility_frequency(string):
    return get_variable(string, '#Utility frequency', int)

# Get the start time (in hh:mm:ss format) from the recording data.
def get_start_time(string):
    variable = get_variable(string, '#Start time', str)
    times = tuple(int(value) for value in variable.split(':'))
    return times

# Get the end time (in hh:mm:ss format) from the recording data.
def get_end_time(string):
    variable = get_variable(string, '#End time', str)
    times = tuple(int(value) for value in variable.split(':'))
    return times

# Convert seconds to days, hours, minutes, seconds.
def convert_seconds_to_hours_minutes_seconds(seconds):
    hours = int(seconds/3600 - 24*days)
    minutes = int(seconds/60 - 24*60*days - 60*hours)
    seconds = int(seconds - 24*3600*days - 3600*hours - 60*minutes)
    return hours, minutes, seconds

# Convert hours, minutes, and seconds to seconds.
def convert_hours_minutes_seconds_to_seconds(hours, minutes, seconds):
    return 3600*hours + 60*minutes + seconds

### Challenge label and output I/O functions

# Save the Challenge outputs for one file.
def save_challenge_outputs(filename, patient_id, outcome, outcome_probability, cpc):
    # Sanitize values, e.g., in case they are a singleton array.
    outcome = sanitize_boolean_value(outcome)
    outcome_probability = sanitize_scalar_value(outcome_probability)
    cpc = sanitize_scalar_value(cpc)

    # Format Challenge outputs.
    patient_string = 'Patient: {}'.format(patient_id)
    if outcome == 0:
        outcome = 'Good'
    elif outcome == 1:
        outcome = 'Poor'
    outcome_string = 'Outcome: {}'.format(outcome)
    outcome_probability_string = 'Outcome Probability: {:.3f}'.format(outcome_probability)
    cpc_string = 'CPC: {:.3f}'.format(cast_int_if_int_else_float(cpc))
    output_string = patient_string + '\n' + \
        outcome_string + '\n' + outcome_probability_string + '\n' + cpc_string + '\n'

    # Write the Challenge outputs.
    if filename is not None:
        with open(filename, 'w') as f:
            f.write(output_string)

    return output_string

### Other helper functions

# Check if a variable is a number or represents a number.
def is_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False

# Check if a variable is an integer or represents an integer.
def is_integer(x):
    if is_number(x):
        return float(x).is_integer()
    else:
        return False

# Check if a variable is a boolean or represents a boolean.
def is_boolean(x):
    if (is_number(x) and float(x)==0) or (remove_extra_characters(x) in ('False', 'false', 'FALSE', 'F', 'f')):
        return True
    elif (is_number(x) and float(x)==1) or (remove_extra_characters(x) in ('True', 'true', 'TRUE', 'T', 't')):
        return True
    else:
        return False

# Check if a variable is a finite number or represents a finite number.
def is_finite_number(x):
    if is_number(x):
        return np.isfinite(float(x))
    else:
        return False

# Check if a variable is a NaN (not a number) or represents a NaN.
def is_nan(x):
    if is_number(x):
        return np.isnan(float(x))
    else:
        return False

# Remove any quotes, brackets (for singleton arrays), and/or invisible characters.
def remove_extra_characters(x):
    return str(x).replace('"', '').replace("'", "").replace('[', '').replace(']', '').replace(' ', '').strip()

# Sanitize boolean values.
def sanitize_boolean_value(x):
    x = remove_extra_characters(x)
    if (is_number(x) and float(x)==0) or (remove_extra_characters(x) in ('False', 'false', 'FALSE', 'F', 'f')):
        return 0
    elif (is_number(x) and float(x)==1) or (remove_extra_characters(x) in ('True', 'true', 'TRUE', 'T', 't')):
        return 1
    else:
        return float('nan')

# Sanitize integer values.
def sanitize_integer_value(x):
    x = remove_extra_characters(x)
    if is_integer(x):
        return int(float(x))
    else:
        return float('nan')

# Sanitize scalar values.
def sanitize_scalar_value(x):
    x = remove_extra_characters(x)
    if is_number(x):
        return float(x)
    else:
        return float('nan')

# Cast a value to a particular type.
def cast_variable(variable, variable_type, preserve_nan=True):
    if preserve_nan and is_nan(variable):
        variable = float('nan')
    else:
        if variable_type == bool:
            variable = sanitize_boolean_value(variable)
        elif variable_type == int:
            variable = sanitize_integer_value(variable)
        elif variable_type == float:
            variable = sanitize_scalar_value(variable)
        else:
            variable = variable_type(variable)
    return variable

# Cast a value to an integer if the value is an integer, a float if the value is a non-integer float, and itself otherwise.
def cast_int_if_int_else_float(x):
    if is_integer(x):
        return int(float(x))
    elif is_number(x):
        return float(x)
    else:
        return x

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
            nn.Linear(meta_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        #self.classifier = nn.Linear(64 + eeg_model.output_dim, 1)
        self.classifier = nn.Linear(64 + 64, 1)
    
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
    n_epochs = 200
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
            print(all_predictions)
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
    current_labels = np.asarray(labels, dtype=np.float64)
    current_outputs = np.asarray(outputs, dtype=np.float64)

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