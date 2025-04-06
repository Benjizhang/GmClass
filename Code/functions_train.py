# fuctions for GmClass work: models, training, testing, etc.
#
# Zeqing Zhang
# 04/2025

import os
import torch
import tsaug
import torch.nn as nn
import numpy as np
import pandas as pd
import clip
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from scipy.fftpack import rfft,rfftfreq
# from functions import *
import matplotlib.pyplot as plt

## frequency encoder w/ 1d CNN (4 Heads + 1 FC: 512 in total)
class CNNModel4H(nn.Module):
    def __init__(self):
        super(CNNModel4H, self).__init__()        
        self.batch_norm = nn.BatchNorm1d(1)
        # head 1
        self.conv11 = nn.Conv1d(1, 128, kernel_size=5)
        self.conv12 = nn.Conv1d(128, 128, kernel_size=5)
        
        # head 2
        self.conv21 = nn.Conv1d(1, 128, kernel_size=15)
        self.conv22 = nn.Conv1d(128, 128, kernel_size=15)

        # head 3
        self.conv31 = nn.Conv1d(1, 128, kernel_size=25)
        self.conv32 = nn.Conv1d(128, 128, kernel_size=25)

        # head 4
        self.conv41 = nn.Conv1d(1, 128, kernel_size=50)
        self.conv42 = nn.Conv1d(128, 128, kernel_size=50)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)        
        self.global_avgpool = nn.AdaptiveAvgPool1d(1) 
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(4*128, 768)
        
    def forward(self, inputs):
        inputs = self.batch_norm(inputs)

        # head 1
        cnn1 = self.conv11(inputs)
        cnn1 = self.relu(cnn1)
        cnn1 = self.maxpool(cnn1)
        cnn1 = self.conv12(cnn1)
        cnn1 = self.relu(cnn1)
        gap1 = self.global_avgpool(cnn1)
        gap1 = torch.flatten(gap1, 1)
        dropout1 = self.dropout(gap1)

        # head 2
        cnn2 = self.conv21(inputs)
        cnn2 = self.relu(cnn2)
        cnn2 = self.maxpool(cnn2)
        cnn2 = self.conv22(cnn2)
        cnn2 = self.relu(cnn2)
        gap2 = self.global_avgpool(cnn2)
        gap2 = torch.flatten(gap2, 1)
        dropout2 = self.dropout(gap2)

        # head 3
        cnn3 = self.conv31(inputs)
        cnn3 = self.relu(cnn3)
        cnn3 = self.maxpool(cnn3)
        cnn3 = self.conv32(cnn3)
        cnn3 = self.relu(cnn3)
        gap3 = self.global_avgpool(cnn3)
        gap3 = torch.flatten(gap3, 1)
        dropout3 = self.dropout(gap3)

        # head 4
        cnn4 = self.conv41(inputs)
        cnn4 = self.relu(cnn4)
        cnn4 = self.maxpool(cnn4)
        cnn4 = self.conv42(cnn4)
        cnn4 = self.relu(cnn4)
        gap4 = self.global_avgpool(cnn4)
        gap4 = torch.flatten(gap4, 1)
        dropout4 = self.dropout(gap4)
        
        # merge
        x = torch.cat([dropout1, dropout2, dropout3, dropout4], dim=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x) # dim = 512, dtype = torch.float32
        
        # convert to torch.float16 & return
        # change to torch.float32/float16 if to cal. similarity with text_features from CLIP in CPU/GPU
        #           torch.float32 if not using CLIP
        return x.to(torch.float16)


class MyCLIP_freq(torch.nn.Module):
    def __init__(self, model_CLIP):
        super().__init__()
        self.model = model_CLIP
        self.encode_freq = CNNModel4H()
        self.latest_valid_acc = 0.0
        self.best_valid_acc   = 0.0
        self.best_valid_acc_Epoch = 0
        self.test1_acc        = 0.0
        self.test2_acc        = 0.0
        self.best_valid_model_dir   = ''
        self.latest_valid_model_dir = ''

    def forward(self, freq, text):
        freq_features = self.encode_freq(freq) #freq in (N,C,L) torch.float16 (for GPU)
        freq_features = freq_features / freq_features.norm(dim=1, keepdim=True) # normalized features

        text_features = self.model.encode_text(text) #text in (N,L) torch.int32
        text_features = text_features / text_features.norm(dim=1, keepdim=True) # normalized features

        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * freq_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text

    def evaluate2(self, dataloader, text_involved_class, involved_class_id, device):
        n_correct = 0
        total = 0
        with torch.no_grad():
            text_features = self.model.encode_text(text_involved_class) # in (class_num, embedding_dim), class_num is len(text_involved_class)
            # text_features = self.fc2(text_features)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            for freqs, labels in dataloader: # labels is gm_id (e.g., \in [0,23] for HKUGM24)
                freqs = freqs.to(device)
                image_features = self.encode_freq(freqs)
                image_features /= image_features.norm(dim=-1, keepdim=True) ## in (batch_num, embedding_dim) 

                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1) # in (batch_num, class_num)
                _, indices = similarity.topk(1) # in (batch_num, 1)
                indices = indices.reshape(-1,) # into a 1D array (e.g., \in [0,19] for seen_class in HKUGM24)
                total += labels.size(0)
                ## correct predicted class id acoording to 'indices' of seen_class
                predicted_class_id = involved_class_id[indices.cpu().numpy()]
                predicted_class_id = torch.tensor(predicted_class_id).to(device)
                n_correct += predicted_class_id.eq(labels.to(device)).sum().item() ## BUG Fixed !!!
        accuracy = n_correct / total
        return accuracy

    ## function to save model
    def save_model(self, saved_model_dict, switch):
            ## save the latest model
            if switch['save_latest_model'] == 1:
                torch.save(saved_model_dict, self.latest_valid_model_dir)
            
            ## save the best model
            if switch['save_best_model'] == 1:
                if self.latest_valid_acc > self.best_valid_acc:
                    self.best_valid_acc = self.latest_valid_acc
                    self.best_valid_acc_Epoch = saved_model_dict['Epoch'] # Epoch = epoch+1
                    self.test1_acc = saved_model_dict['test1_acc']
                    self.test2_acc = saved_model_dict['test2_acc']
                    torch.save(saved_model_dict, self.best_valid_model_dir)

## class for FFT function
class FFT_ts(object):
    def __init__(self):
        self.frequency_bar = 190   # Hertz
        self.SAMPLE_RATE   = 62.5  # Hertz

    def __call__(self,sample_ts):
        _, fd_row = sample_ts['tt_row'], sample_ts['fd_row']
        assert isinstance(fd_row, (np.ndarray))

        fd_arr_1 = fd_row # np.ndarray
        ## FFT        
        fxf = rfft(fd_arr_1)
        tf  = rfftfreq(fd_arr_1.size, 1/self.SAMPLE_RATE)
        abs_fxf=np.abs(fxf)
        normalization_y=abs_fxf/fd_arr_1.size
        if len(normalization_y) < self.frequency_bar:
            raise Exception('Not enough data points!')
        else:
            normalization_y_ = normalization_y[:self.frequency_bar]
            tf_              = tf[:self.frequency_bar]
            cut_off_fre = np.round(tf[self.frequency_bar-1],3)
            # print(f'cut-off frequency: {cut_off_fre} Hz')
            if cut_off_fre < 10:
                ## b/z total time of ts data > 10 seconds
                raise Exception('Cut-off frequency < 10 Hz!')
        normalization_y_ = np.array(normalization_y_).reshape(-1,1) # (200,1)
        tf_              = np.array(tf_).reshape(-1,1) # (200,1)
        
        return {'freq spectrum': tf_, 'freq magnitude':normalization_y_}


## class for add noise
class AddNoise_ts(object):
    def __init__(self, loc=0.0, scale=0.01):
        self.my_augmenter = (
            tsaug.AddNoise(loc=loc,scale=scale,seed=888) @ 0.6 # with 60% probability 
            )
    
    def __call__(self, sample):
        tt_row, fd_row = sample['tt_row'], sample['fd_row']
        assert isinstance(tt_row, (np.ndarray)) and isinstance(fd_row, (np.ndarray))
        fd_row_aug = self.my_augmenter.augment(fd_row)
        return {'tt_row': tt_row, 'fd_row': fd_row_aug} # (N,) (N,)


## class for crop
class Crop_ts(object):
    def __init__(self, crop_method='rd_rs', duration=8, start=0):
        '''
        crop_method: 'rd_rs' or 'rd_fs' or 'fd_rs' or 'fd_fs'
                        rd_rs: random duration, random start
                        rd_fs: random duration, fixed start
                        fd_rs: fixed duration,  random start
                        fd_fs: fixed duration,  fixed start
        duration: crop duration (seconds)
        start: crop start (seconds)
        '''
        self.crop_method   = crop_method
        self.crop_duration = duration
        self.crop_start    = start
    
    def __call__(self, sample):
        tt_row, fd_row = sample['tt_row'], sample['fd_row']
        assert isinstance(tt_row, (np.ndarray)) and isinstance(fd_row, (np.ndarray))

        time_arr = tt_row # (N,)
        fd_arr   = fd_row # (N,)
        if self.crop_method == 'rd_rs':
            ## random duration
            dur_tnsr = torch.empty(1).uniform_(5, 10)
            duration = np.round(dur_tnsr.item(),3)
            ## random start
            str_tnsr = torch.empty(1).uniform_(0, 10-duration)
            time_l = np.round(str_tnsr.item(),3)            
        elif self.crop_method == 'rd_fs':
            ## random duration
            dur_tnsr = torch.empty(1).uniform_(5, 10)
            duration = np.round(dur_tnsr.item(),3)
            ## fixed start
            time_l = self.crop_start
        elif self.crop_method == 'fd_rs':
            ## fixed duration
            duration = self.crop_duration
            ## random start
            str_tnsr = torch.empty(1).uniform_(0, 10-duration)
            time_l = np.round(str_tnsr.item(),3)
        elif self.crop_method == 'fd_fs':
            ## fixed duration
            duration = self.crop_duration
            ## fixed start
            time_l = self.crop_start
        else:
            raise Exception('Invalid crop method!')
        time_r = time_l+duration
        if np.round(time_r - 10.001,3) >0:
                raise Exception('Crop time out of range!')
        index_slice = np.intersect1d(np.where(time_arr>time_l), np.where(time_arr<time_r))
        fd_arr_1 = fd_arr[index_slice]     # (N_crop,)
        time_arr_1 = time_arr[index_slice] # (N_crop,)
    
        return {'tt_row': time_arr_1, 'fd_row': fd_arr_1}


## read HKUGM dataset (time-series dataset)
## Let's create a [torch.utils.data.Dataset] class for our -ts dataset, i.e., HKUGM.
class HKUGMDataset_ts(Dataset):
    """ HKUGM dataset """
    def __init__(self, dataset_name, root_dir, train = 1, transform=None):
        """
        Arguments:
            dataset_name (string): dataset name.
            root_dir (string): Directory to dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        temp = pd.read_csv(root_dir+'/'+ dataset_name +'.txt', header=None)
        ## all class names in 'dataset_name'
        self.whole_set_class_name = np.array(temp, dtype=str).flatten()
        self.root_dir = root_dir
        ## 1: train, 2: valid, 3: test1, 4: test2
        self.train = train
        self.transform = transform
        if self.train == 1:
            self.landmarks_frame = pd.read_csv(root_dir+'/'+ dataset_name +'_train.csv', header=None)
        elif self.train == 2:
            self.landmarks_frame = pd.read_csv(root_dir+'/'+ dataset_name +'_valid.csv', header=None)
        elif self.train == 3:
            self.landmarks_frame = pd.read_csv(root_dir+'/'+ dataset_name +'_test1.csv', header=None)
        elif self.train == 4:
            self.landmarks_frame = pd.read_csv(root_dir+'/'+ dataset_name +'_test2.csv', header=None)
        else:
            raise ValueError('Invalid train mode')
        self.involved_class_id   = self.get_class_id()
        self.involved_class_name = self.get_class_name()

    def __len__(self):
        return len(self.landmarks_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ## load GM name ID
        gm_name_id = self.landmarks_frame.iloc[idx, 0].astype(np.int32) # int
        gm_name_id = torch.tensor(gm_name_id, dtype=torch.int32) # torch.int32

        ## load time series
        tt_fd_row = self.landmarks_frame.iloc[idx, 1:]
        ## calculate the midpoint index        
        if len(tt_fd_row)%2 != 0:
            raise Exception('Should have even number of rows!')
        midpoint = len(tt_fd_row) // 2
        ## Split the series into two halves
        tt_row = tt_fd_row.iloc[:midpoint] # pd.series
        fd_row = tt_fd_row.iloc[midpoint:] # pd.series
        ## dinctionary (np.ndarry)
        sample_ts = {'tt_row': tt_row.values, 'fd_row': fd_row.values} # np.ndarry

        if self.transform:
            sample_ts = self.transform(sample_ts)

        ## conduct FFT
        fft_fn = FFT_ts()
        fft_result = fft_fn(sample_ts)
        freq = fft_result['freq magnitude']

        ## convert to tensor for freq
        freq = freq.T # (C, L), C=1
        freq = torch.tensor(freq, dtype=torch.float32) # torch.float32

        return freq, gm_name_id #, gm_name
    
    def get_class_id(self):
        first_column_values = self.landmarks_frame.iloc[:, 0]
        unique_int_arr = first_column_values.unique().astype(int)
        return unique_int_arr
        
    def get_class_name(self):
        return self.whole_set_class_name[self.involved_class_id]

## read HKUGM-ts-Plus (considering avd parameters, return 'avd')
## Let's create a [torch.utils.data.Dataset] class for our -ts-Plus dataset, e.g., HKUGM10-ts-Plus.
class HKUGMDataset_ts_Plus2(Dataset):
    """ HKUGM-ts-Plus dataset """
    def __init__(self, dataset_name, root_dir, train = 1, transform=None):
        """
        Arguments:
            dataset_name (string): dataset name.
            root_dir (string): Directory to dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        temp = pd.read_csv(root_dir+'/'+ dataset_name +'.txt', header=None)
        ## all class names in 'dataset_name'
        self.whole_set_class_name = np.array(temp, dtype=str).flatten()
        self.root_dir = root_dir
        ## 1: train, 2: valid, 3: test1, 4: test2
        self.train = train
        self.transform = transform
        if self.train == 1:
            self.landmarks_frame = pd.read_csv(root_dir+'/'+ dataset_name +'_train.csv', header=None)
        elif self.train == 2:
            self.landmarks_frame = pd.read_csv(root_dir+'/'+ dataset_name +'_valid.csv', header=None)
        elif self.train == 3:
            self.landmarks_frame = pd.read_csv(root_dir+'/'+ dataset_name +'_test1.csv', header=None)
        elif self.train == 4:
            self.landmarks_frame = pd.read_csv(root_dir+'/'+ dataset_name +'_test2.csv', header=None)
        else:
            raise ValueError('Invalid train mode')
        self.involved_class_id   = self.get_class_id()
        self.involved_class_name = self.get_class_name()

    def __len__(self):
        return len(self.landmarks_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ## load GM name ID
        gm_name_id = self.landmarks_frame.iloc[idx, 0].astype(np.int32) # int
        gm_name_id = torch.tensor(gm_name_id, dtype=torch.int32) # torch.int32

        ## load avd parameters
        avd = self.landmarks_frame.iloc[idx, 1:4] # pandas.core.series.Series
        avd = np.array(avd, dtype=float).reshape(3,) # (C,), C=1
        avd = torch.tensor(avd, dtype=torch.float32) # torch.float16

        ## load time series
        tt_fd_row = self.landmarks_frame.iloc[idx, 4:]
        ## calculate the midpoint index        
        if len(tt_fd_row)%2 != 0:
            raise Exception('Should have even number of rows!')
        midpoint = len(tt_fd_row) // 2
        ## Split the series into two halves
        tt_row = tt_fd_row.iloc[:midpoint] # pd.series
        fd_row = tt_fd_row.iloc[midpoint:] # pd.series
        ## dinctionary (np.ndarry)
        sample_ts = {'tt_row': tt_row.values, 'fd_row': fd_row.values} # np.ndarry

        if self.transform:
            sample_ts = self.transform(sample_ts)
        ## conduct FFT
        fft_fn = FFT_ts()
        fft_result = fft_fn(sample_ts)
        freq = fft_result['freq magnitude']

        freq = freq.T # (C, L), C=1
        freq = torch.tensor(freq, dtype=torch.float32) # torch.float32

        return avd, freq, gm_name_id #, gm_name
    
    def get_class_id(self):
        first_column_values = self.landmarks_frame.iloc[:, 0]
        unique_int_arr = first_column_values.unique().astype(int)
        return unique_int_arr
        
    def get_class_name(self):
        return self.whole_set_class_name[self.involved_class_id]

## function to load HKUGM-ts dataset
def load_HKUGM_ts(dataset_name_gm, root_dir_gm, batch_size_gm, num_workers_gm, transform_gm=AddNoise_ts(loc=0.0, scale=0.01)):
    ## NOTE train: 1, valid: 2, test1: 3, test2: 4
    gm_dataset_train = HKUGMDataset_ts(dataset_name=dataset_name_gm, train= 1, root_dir=root_dir_gm, transform=transform_gm)
    gm_dataset_valid = HKUGMDataset_ts(dataset_name=dataset_name_gm, train= 2, root_dir=root_dir_gm, transform=Crop_ts(crop_method='fd_fs'))
    gm_dataset_test1 = HKUGMDataset_ts(dataset_name=dataset_name_gm, train= 3, root_dir=root_dir_gm, transform=Crop_ts(crop_method='fd_fs'))    
    trainGMLoader = DataLoader(gm_dataset_train, batch_size=batch_size_gm, shuffle=True, num_workers=num_workers_gm)
    validGMLoader = DataLoader(gm_dataset_valid, batch_size=batch_size_gm, shuffle=True, num_workers=num_workers_gm)
    test1GMLoader = DataLoader(gm_dataset_test1, batch_size=batch_size_gm, shuffle=True, num_workers=num_workers_gm)

    ## check test2.csv exists or not
    if os.path.exists(root_dir_gm+'/'+dataset_name_gm+'_test2.csv'):
        gm_dataset_test2 = HKUGMDataset_ts(dataset_name=dataset_name_gm, train= 4, root_dir=root_dir_gm, transform=Crop_ts(crop_method='fd_fs'))
        test2GMLoader = DataLoader(gm_dataset_test2, batch_size=batch_size_gm, shuffle=True, num_workers=num_workers_gm)
    else:
        print('!! No test2 dataset !!')
        test2GMLoader = None
    return trainGMLoader, validGMLoader, test1GMLoader, test2GMLoader, gm_dataset_train


## function to load HKUGM-ts-Plus dataset (return avd parameters)
def load_HKUGM_ts_Plus2(dataset_name_gm, root_dir_gm, batch_size_gm, num_workers_gm, transform_gm=AddNoise_ts(loc=0.0, scale=0.01)):
    ## NOTE train: 1, valid: 2, test1: 3, test2: 4
    gm_dataset_train = HKUGMDataset_ts_Plus2(dataset_name=dataset_name_gm, train= 1, root_dir=root_dir_gm, transform=transform_gm)
    gm_dataset_valid = HKUGMDataset_ts_Plus2(dataset_name=dataset_name_gm, train= 2, root_dir=root_dir_gm, transform=Crop_ts(crop_method='fd_fs'))
    gm_dataset_test1 = HKUGMDataset_ts_Plus2(dataset_name=dataset_name_gm, train= 3, root_dir=root_dir_gm, transform=Crop_ts(crop_method='fd_fs'))    
    trainGMLoader = DataLoader(gm_dataset_train, batch_size=batch_size_gm, shuffle=True, num_workers=num_workers_gm)
    validGMLoader = DataLoader(gm_dataset_valid, batch_size=batch_size_gm, shuffle=True, num_workers=num_workers_gm)
    test1GMLoader = DataLoader(gm_dataset_test1, batch_size=batch_size_gm, shuffle=True, num_workers=num_workers_gm)

    ## check test2.csv exists or not
    if os.path.exists(root_dir_gm+'/'+dataset_name_gm+'_test2.csv'):
        gm_dataset_test2 = HKUGMDataset_ts_Plus2(dataset_name=dataset_name_gm, train= 4, root_dir=root_dir_gm, transform=Crop_ts(crop_method='fd_fs'))
        test2GMLoader = DataLoader(gm_dataset_test2, batch_size=batch_size_gm, shuffle=True, num_workers=num_workers_gm)
    else:
        print('!! No test2 dataset !!')
        test2GMLoader = None
    return trainGMLoader, validGMLoader, test1GMLoader, test2GMLoader, gm_dataset_train


## function to plot/save the loss/acc trend
def plot_loss_acc_trend(loss_ls, acc_ls, save_dir):
    if loss_ls != []:
        # Plotting the loss
        fig = plt.figure()
        plt.plot(loss_ls)
        plt.xlabel('epoch')
        plt.ylabel('Loss')
        plt.title('Loss Trend')
        plt.savefig(save_dir+'/loss_trend.png')
        plt.close()

    # Plotting the accuracy
    if acc_ls != []:
        fig = plt.figure()
        plt.plot(acc_ls)
        plt.xlabel('epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Trend')
        plt.savefig(save_dir+'/acc_trend.png')
        plt.close()

## function to save 4 acc lists
def save_4_acc_lists(train_acc_ls, valid_acc_ls, test1_acc_ls, test2_acc_ls, save_dir,formatted_time,dataset_name):
    # Create a new figure and set the title
    fig = plt.figure()
    fig.suptitle(f"{formatted_time} - {dataset_name}")

    # Create the first subplot and plot train_acc_ls
    ax1 = fig.add_subplot(221)
    ax1.plot(train_acc_ls)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('Epoch')
    ax1.set_title('Train Accuracy')

    # Create the second subplot and plot valid_acc_ls
    ax2 = fig.add_subplot(222)
    ax2.plot(valid_acc_ls)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Epoch')
    ax2.set_title('Validation Accuracy')

    # Create the third subplot and plot test1_acc_ls
    ax3 = fig.add_subplot(223)
    ax3.plot(test1_acc_ls)
    ax3.set_ylim(0, 1)
    ax3.set_xlabel('Epoch')
    ax3.set_title('Test1 Accuracy')

    # Create the fourth subplot and plot test2_acc_ls
    ax4 = fig.add_subplot(224)
    ax4.plot(test2_acc_ls)
    ax4.set_ylim(0, 1)
    ax4.set_xlabel('Epoch')
    ax4.set_title('Test2 Accuracy')

    # Adjust the spacing between subplots
    fig.tight_layout()
    plt.savefig(save_dir+'/acc_trend.png')

## class about gm properties
class GM_properties(object):
    def __init__(self):
        self.material_properties = {
            'sand': {
                'particle size': 'small',
                'roughness': 'rough',
                'particle shape': 'circular',
                'weight': 'heavy'
            },
            'cat litter': {
                'particle size': 'medium',
                'roughness': 'rough',
                'particle shape': 'circular',
                'weight': 'heavy'
            },
            'gravel': {
                'particle size': 'medium',
                'roughness': 'rough',
                'particle shape': 'non-circular',
                'weight': 'heavy'
            },
            'soybean': {
                'particle size': 'large',
                'roughness': 'smooth',
                'particle shape': 'circular',
                'weight': 'light'
            },
            'mung bean': {
                'particle size': 'medium',
                'roughness': 'smooth',
                'particle shape': 'circular',
                'weight': 'medium'
            },
            'red bean': {
                'particle size': 'medium',
                'roughness': 'smooth',
                'particle shape': 'circular',
                'weight': 'medium'
            },
            'broad bean': {
                'particle size': 'large',
                'roughness': 'smooth',
                'particle shape': 'non-circular',
                'weight': 'medium'
            },
            'coffee bean': {
                'particle size': 'medium',
                'roughness': 'smooth',
                'particle shape': 'circular',
                'weight': 'light'
            },
            'cassia seed': {
                'particle size': 'medium',
                'roughness': 'smooth',
                'particle shape': 'non-circular',
                'weight': 'medium'
            },
            'millet': {
                'particle size': 'small',
                'roughness': 'smooth',
                'particle shape': 'circular',
                'weight': 'light'
            },
            'silkworm excrement': {
                'particle size': 'small',
                'roughness': 'rough',
                'particle shape': 'circular',
                'weight': 'light'
            },
            'negundo chastetree fruit': {
                'particle size': 'small',
                'roughness': 'smooth',
                'particle shape': 'circular',
                'weight': 'light'
            },
            'long-grain rice': {
                'particle size': 'medium',
                'roughness': 'smooth',
                'particle shape': 'non-circular',
                'weight': 'heavy'
            },
            'pearl rice': {
                'particle size': 'medium',
                'roughness': 'smooth',
                'particle shape': 'circular',
                'weight': 'heavy'
            },
            'oatmeal rice': {
                'particle size': 'medium',
                'roughness': 'smooth',
                'particle shape': 'non-circular',
                'weight': 'heavy'
            },
            'in-shell peanut': {
                'particle size': 'large',
                'roughness': 'rough',
                'particle shape': 'non-circular',
                'weight': 'light'
            },
            'shelled peanut': {
                'particle size': 'medium',
                'roughness': 'smooth',
                'particle shape': 'circular',
                'weight': 'medium'
            },
            'crushed peanut': {
                'particle size': 'small',
                'roughness': 'rough',
                'particle shape': 'non-circular',
                'weight': 'medium'
            },
            'corn kernel': {
                'particle size': 'medium',
                'roughness': 'smooth',
                'particle shape': 'non-circular',
                'weight': 'medium'
            },
            'baysalt': {
                'particle size': 'medium',
                'roughness': 'rough',
                'particle shape': 'non-circular',
                'weight': 'heavy'
            },
            'refined salt': {
                'particle size': 'small',
                'roughness': 'rough',
                'particle shape': 'circular',
                'weight': 'heavy'
            },
            'sunflower seed': {
                'particle size': 'large',
                'roughness': 'smooth',
                'particle shape': 'non-circular',
                'weight': 'light'
            },
            'small macaroni': {
                'particle size': 'large',
                'roughness': 'smooth',
                'particle shape': 'non-circular',
                'weight': 'light'
            },
            'large macaroni': {
                'particle size': 'large',
                'roughness': 'smooth',
                'particle shape': 'non-circular',
                'weight': 'light'
            }
        }

    def get(self, material, property_name):
        if material in self.material_properties:
            properties = self.material_properties[material]
            if property_name in properties:
                return properties[property_name]
            else:
                raise ValueError(f"Property '{property_name}' of {material} not found in the database.")
        else:
            raise ValueError(f"The {material} not found in the database.")


## function to concatenate_text w/ gm prperties
def concatenate_text(prompt, class_ls, whole_set_class_name, prpt_id_ls, isClassId=False):
    concatenated_texts = []
    gm_prpt = GM_properties()
    prpt_si = 'particle size'
    prpt_sh = 'particle shape'
    prpt_ro = 'roughness'
    prpt_we = 'weight'
    prpt_name_ls = [prpt_si,prpt_sh,prpt_ro,prpt_we]
    
    for c in class_ls:
        if isClassId == True:
            # c is class id
            assert isinstance(c, np.int32)
            cc = whole_set_class_name[c]
        else:
            # c is class name
            assert isinstance(c, str)
            cc = c        
        # cc is class name
        prpt_text = prompt + f'{cc}'
        for index, prpt_id in enumerate(prpt_id_ls):
            if index == 0:
                prpt_text += ', whose '
            prpt_name = prpt_name_ls[prpt_id]
            if (index+1) == len(prpt_id_ls) and len(prpt_id_ls) != 1:
                prpt_text += 'and '
            prpt_text += f'{prpt_name} is {gm_prpt.get(cc, prpt_name)}'                 
            if (index+1) != len(prpt_id_ls):
                prpt_text += ', '
        
        tokenized_text = clip.tokenize(prpt_text)
        concatenated_texts.append(tokenized_text)
    
    text_torch_cat = torch.cat(concatenated_texts)
    return text_torch_cat, prpt_text # << last property text for test

