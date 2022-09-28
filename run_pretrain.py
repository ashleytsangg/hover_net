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
model_save_name = 'monusac_lymph_25_pannuke.tar'
model_save_dir = r'\\babyserverdw3\PW Cloud Exp Documents\Lab work documenting\W-22-09-02 AT Establish HoverNet Training with freezing weights\saved_models\freeze_trained\monusac applied to lymph'
model_save_path =os.path.join(model_save_dir, model_save_name)

# 2. add run info in "config.py" - dataset name, path to training data, nr_types, training data, model mode, etc.

# 3. set params
nr_types = 5 # number of nuclear types
n_epochs = 25 # number of epochs
sparse_labels = True # whether training data is sparsely labeled (ie. lymph = True, consep = False)
learning_rate = 0.0001

# 4. load pretrained model and params, choose pretrained model with segmentation & classification
pretrained_path = r"\\babyserverdw3\PW Cloud Exp Documents\Lab work documenting\W-22-09-02 AT Establish HoverNet Training with freezing weights\saved_models\classification\Monusac\monusac_class_train_20.tar"
pretrained_nr_types = 5  # according to pretrainedmodel
pretrained_mode = "fast" # original or fast

# net_state_dict = torch.load(pretrained_path)["desc"]
net_state_dict = torch.load(pretrained_path)

# initialize HoverNet model
# nr_types is dependent on pretrained dataset (ie. consep = 5)
model = create_model(mode=pretrained_mode, input_ch=3, nr_types=pretrained_nr_types, freeze=True)

# load model state_dict
model.load_state_dict(net_state_dict, strict=True)
model = torch.nn.DataParallel(model)

# loop through layers and freeze via requires_grad = False
for param in model.parameters():
    param.requires_grad = False

# load in data via config.py
trainer = TrainManager()
train_dataloader = trainer.get_dataloader()

# unfreeze last conv2d 1x1 layer of tp (nuclear classification) branch for training
# model.module.decoder.tp.u0.conv = nn.Conv2d(64, nr_types, kernel_size=(1,1), stride=(1,1))

old_model = list(model.module.decoder.tp.u0.children()) # get the last Conv2d layer
old_model.pop() # remove it since we have a different number of classes
old_model.append(nn.Conv2d(64, nr_types, kernel_size=(1, 1), stride=(1, 1))) # add the new layer with new classes
model.module.decoder.tp.u0 = nn.Sequential(*old_model) # append to original model

# replace last nn.Sequential layer
model.module.nr_types = nr_types
model.module.decoder.tp.u0 = nn.Sequential(
    OrderedDict([
        ("bn", nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
        ("relu", nn.ReLU(inplace=True)),
        ("conv", nn.Conv2d(64, nr_types, kernel_size=(1, 1), stride=(1, 1)))
    ])
)

# also replace second to last nn.Sequential layer, conv2d only
# model.module.decoder.tp.u1.conva = nn.Conv2d(256, 64, kernel_size=(5, 5), stride=(1, 1), bias=False)

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
    print("epoch: ", epoch)
    for batch_idx, batch_data in enumerate(train_dataloader):
        train_step(batch_data, run_info, sparse_labels=sparse_labels)
    torch.save(model.module.state_dict(), epoch_save_path)

torch.save(model.module.state_dict(), model_save_path)




