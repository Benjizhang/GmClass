# script to train the model from scratch
# 
# Z. Zhang
# 04/2025

from datetime import datetime 
import os
import clip
import torch
import zipfile
import random
import numpy as np
from functions_train import *

## set the seed for reproducibility
seed = 8888
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed) 
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
### to get deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Specify the dir/path
dataset_name = 'HKUGM10_ts' #!! check the correct name of the dataset
repo_dir = os.path.dirname(os.path.abspath(__file__))
zip_path = os.path.join(repo_dir, dataset_name + '.zip')
dataset_dir = os.path.splitext(zip_path)[0]
trend_fig_dir = os.path.join(repo_dir, 'trend_fig')

os.makedirs(trend_fig_dir, exist_ok=True)

if os.path.isfile(zip_path):
    print(f'The Dataset {dataset_name} (ZIP) exists.')
else:
    print(f'The Dataset {dataset_name} (ZIP) does not exist.')

if not os.path.exists(os.path.join(repo_dir, dataset_name)):
    # Create a folder to extract the contents
    os.makedirs(dataset_dir, exist_ok=True)

    # Open the ZIP file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract all files to the created folder
        zip_ref.extractall(dataset_dir)
    print(f'The Dataset {dataset_name} is Unzipped!')
else:
    print(f'The Dataset {dataset_name} is already Unzipped!')

## get current time
now = datetime.now()
formatted_time = now.strftime("%m%d%H%M%S") # 10 digits

## Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using {device} for training.')
model, preprocess = clip.load('ViT-L/14@336px', device)

## instantiate the model and move it to the GPU if available
myModel = MyCLIP_freq(model)
myModel = myModel.to(device)

## Load the TS-based dataset
batch_size   = 64
num_workers  = 0 if os.name == 'nt' else 4
print('num of workers:', num_workers)
trsf = transforms.Compose([Crop_ts(),
                        AddNoise_ts(loc=0.0, scale=0.01)])
trainGMLoader, validGMLoader, test1GMLoader, _, gm_dataset_train = load_HKUGM_ts(dataset_name, dataset_dir, batch_size, num_workers, trsf)

## Load the GM properies class
gm_prpt = GM_properties()

## freeze the CLIP network
for para in myModel.model.parameters():
    para.requires_grad = False

## setting parameters
init_lr = 1e-4 #5e-5 # 1e-3
train_epoch = 150

## optimizer selection
optimizer = torch.optim.Adam(myModel.parameters(), lr=init_lr)
optimizer_name = optimizer.__class__.__name__

## loss function selection
loss_fre = torch.nn.BCEWithLogitsLoss()
loss_txt = torch.nn.BCEWithLogitsLoss()

## setting for text
prompt = "This is "
prpt_si = 'particle size'
prpt_sh = 'particle shape'
prpt_ro = 'roughness'
prpt_we = 'weight'
prpt_id_ls = [] # empth: no properties considered
text_seen_class_, _ = concatenate_text(prompt, gm_dataset_train.involved_class_name, [], prpt_id_ls, isClassId=False)
text_all_class_, _ = concatenate_text(prompt, gm_dataset_train.whole_set_class_name, [], prpt_id_ls, isClassId=False)
text_seen_class = text_seen_class_.to(device)
text_all_class = text_all_class_.to(device)
print(_)
seen_class_id = gm_dataset_train.involved_class_id

## setting for saved model
saved_model_dir = os.path.join(repo_dir,'saved_model')
save_prefix = 'onlyonemod'
myModel.best_valid_model_dir   = saved_model_dir+f'/{save_prefix}_{dataset_name}_best_valid.pt'
myModel.latest_valid_model_dir = saved_model_dir+f'/{save_prefix}_{dataset_name}_latest_valid.pt'
switch = {'save_latest_model':0,'save_best_model':1}
if 1 in switch.values():
    os.makedirs(saved_model_dir, exist_ok=True)
    print(f'The folder to save models is created in {saved_model_dir}')

loss_ls = []
train_acc_ls, valid_acc_ls = [], []
# record the time consumption
startT = datetime.now()
for epoch in range(train_epoch):
    for inputs, labels in trainGMLoader: # labels is gm id
        texts_, _ = concatenate_text(prompt, np.array(labels), gm_dataset_train.whole_set_class_name, prpt_id_ls, isClassId=True)        
        texts = texts_.to(device)

        freqs  = inputs.to(device)
        optimizer.zero_grad()
        logits_per_freq, logits_per_text  = myModel(freqs, texts) # logits_per_freq,logits_per_text in (batch_size, batch_size)
        
        ## calculate the loss
        ground_truth = (labels[:, None] == labels).float().to(device) # BCEWithLogitsLoss() expects floating-point inputs.
        total_loss = (loss_fre(logits_per_freq, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
        total_loss.backward()
        optimizer.step()
    
    cur_lr = optimizer.param_groups[0]['lr']
    ## evaluate the model
    train_acc = myModel.evaluate2(trainGMLoader, text_seen_class, seen_class_id, device) # training dataset
    valid_acc = myModel.evaluate2(validGMLoader, text_seen_class, seen_class_id, device) # validation dataset
    
    print(f"Epoch {epoch+1}: train accuracy is {train_acc*100:.3f}% || valid accuracy is {valid_acc*100:.3f}%")
    print(f'Epoch {epoch+1}: time cost is {datetime.now() - startT}')
    print(f'Epoch {epoch+1}: cur lr is {cur_lr}')
    print('--------------------------------------------------')

    loss_ls.append(total_loss.item())
    train_acc_ls.append(train_acc)
    valid_acc_ls.append(valid_acc)
    ## Tune the learning rate
    if epoch ==69:
        second_lr = 1e-5
        optimizer = torch.optim.Adam(myModel.parameters(), lr=second_lr)
    if epoch ==119:
        third_lr = 1e-6
        optimizer = torch.optim.Adam(myModel.parameters(), lr=third_lr)

    ## save the lateset/best model
    myModel.latest_valid_acc = valid_acc
    saved_model_dict = {'dataset':dataset_name, 'prompt': prompt, 'batch_size': batch_size, 'Epoch': epoch+1, 'init_lr': init_lr,
                        'model_state_dict': myModel.state_dict(),'optim_state_dict': optimizer.state_dict()}
    myModel.save_model(saved_model_dict, switch)

    ## plot/save the loss curve
    if epoch % 10 == 9:
        plot_loss_acc_trend(loss_ls, [], trend_fig_dir)
        save_4_acc_lists(train_acc_ls, valid_acc_ls, 0, 0, trend_fig_dir, formatted_time, dataset_name)

endT = datetime.now()
time_cost = endT - startT
print(f'# {train_epoch} Epochs Done!')
print(f'time consumption: {time_cost}, dataset: {dataset_name}')
print(f'optimizer: {optimizer_name}, batch size: {batch_size}, init lr: {init_lr}, cur lr: {cur_lr}')
print(f'best valid acc: {myModel.best_valid_acc*100:.3f}% at Epoch {myModel.best_valid_acc_Epoch} at {formatted_time}.')

current_file_path = os.path.abspath(__file__)
current_file_name = os.path.basename(current_file_path)
print(f'Current File Name: {current_file_name}')

print(f'Example text in training: {_}')
