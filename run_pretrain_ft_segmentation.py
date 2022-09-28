'''
This file is for training the *last layer* of segmentation branch of HoVerNet
'''

import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from collections import OrderedDict

from models.hovernet.net_desc import create_model
from run_train import TrainManager
from models.hovernet.run_desc import pre_train_step, train_step

# 1. change model save path
model_save_name = 'ft_seg_10_np_hv.tar'
model_save_dir = r'\\babyserverdw3\PW Cloud Exp Documents\Lab work documenting\W-22-09-02 AT Establish HoverNet Training with freezing weights\saved_models\segmentation\IF'
model_save_path =os.path.join(model_save_dir, model_save_name)

# 2. add run info in "config.py" - dataset name, path to training data, nr_types, training data, model mode, etc.

# 3. set params
nr_types = None # number of nuclear types
n_epochs = 10 # number of epochs
sparse_labels = False # whether training data is sparsely labeled (ie. lymph = True, consep = False)
learning_rate = 0.0001

# 4. load pretrained model and params, choose pretrained model with segmentation & classification
pretrained_path = r"\\babyserverdw3\PW Cloud Exp Documents\Lab work documenting\W-22-09-02 AT Establish HoverNet Training with freezing weights\saved_models\full_pretrained\hovernet_original_consep_notype_tf2pytorch.tar"
pretrained_nr_types = None  # according to pretrained model
pretrained_mode = "original" # original or fast

net_state_dict = torch.load(pretrained_path)["desc"]

# initialize HoverNet model
# nr_types is dependent on pretrained dataset (ie. consep = 5)
model = create_model(mode=pretrained_mode, input_ch=3, nr_types=pretrained_nr_types, freeze=True)

# load model state_dict
model.load_state_dict(net_state_dict, strict=True)
model = torch.nn.DataParallel(model)

# print(model)

# loop through layers and freeze via requires_grad = False
for param in model.parameters():
    param.requires_grad = False

# load in data via config.py
trainer = TrainManager()
train_dataloader = trainer.get_dataloader()

model.module.decoder.np.u0 = nn.Sequential(
    OrderedDict([
        ("bn", nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
        ("relu", nn.ReLU(inplace=True)),
        ("conv", nn.Conv2d(64, 2, kernel_size=(1, 1), stride=(1, 1)))
    ]))

model.module.decoder.hv.u0 = nn.Sequential(
    OrderedDict([
        ("bn", nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
        ("bn", nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
        ("relu", nn.ReLU(inplace=True)),
        ("conv", nn.Conv2d(64, 2, kernel_size=(1, 1), stride=(1, 1)))
    ]))

# get parameters to update, check only updating the unfrozen layers
params_to_update = []
for name, param in model.named_parameters():
    if param.requires_grad:
        params_to_update.append(param)
        print("\t", name)

# set up optimizer on parameters to update
optimizer = optim.Adam(params_to_update, lr=learning_rate, betas=(0.9, 0.999))

# run_info from opt.py
run_info = trainer.model_config['phase_list'][0]['run_info']
run_info['net']['optimizer'] = optimizer
model.to("cuda")
run_info['net']['desc'] = model

epoch_save_path = model_save_path[:-4] + '_recent.tar'
# train last layer on new training data
for epoch in range(n_epochs):
    for batch_idx, batch_data in enumerate(train_dataloader):
        print("epoch: ", epoch)
        train_step(batch_data, run_info, sparse_labels=sparse_labels)
    torch.save(model.module.state_dict(), epoch_save_path)

torch.save(model.module.state_dict(), model_save_path)




