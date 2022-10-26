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
import time

from models.hovernet.net_desc import create_model
from run_train import TrainManager
from models.hovernet.run_desc import pre_train_step, train_step

# 1. change model save path
model_save_name = 'pannuke_10_WLM00.tar'
model_save_dir = r'\\babyserverdw3\PW Cloud Exp Documents\Lab work documenting\W-22-09-02 AT Establish HoverNet Training with freezing weights\saved_models\0921 lymph models\full branch'
if not os.path.exists(model_save_dir):
    os.mkdir(model_save_dir)
model_save_path =os.path.join(model_save_dir, model_save_name)

# 2. add run info in "config.py" - dataset name, path to training data, nr_types, training data, model mode, etc.

# 3. set params
classification = 'full' # 'full' or 'last'
n_epochs = 10 # number of epochs
loss_method = 1 # LM1 or LM2
weighted = True
learning_rate = 1e-4

nr_types = 6 # number of nuclear types
sparse_labels = True # whether training data is sparsely labeled (ie. lymph = True, consep = False)

# 4. load pretrained model and params, choose pretrained model with segmentation & classification
pretrained_path = r'\\babyserverdw3\PW Cloud Exp Documents\Lab work documenting\W-22-09-02 AT Establish HoverNet Training with freezing weights\saved_models\full_pretrained\hovernet_fast_pannuke_type_tf2pytorch.tar'
pretrained_nr_types = 6  # according to pretrainedmodel
pretrained_mode = "fast" # original or fast

net_state_dict = torch.load(pretrained_path)["desc"]
# net_state_dict = torch.load(pretrained_path)

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

if classification == 'last':
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

if classification == 'full':
    tp_branch = model.module.decoder.tp
    for param in tp_branch.parameters():
        param.requires_grad = True

    old_model = list(model.module.decoder.tp.u0.children())  # get the last Conv2d layer
    old_model.pop()  # remove it since we have a different number of classes
    old_model.append(nn.Conv2d(64, nr_types, kernel_size=(1, 1), stride=(1, 1)))  # add the new layer with new classes
    model.module.decoder.tp.u0 = nn.Sequential(*old_model)  # append to original model

    # ensure correct num classes
    model.module.nr_types = nr_types
    model.module.decoder.tp.u0 = nn.Sequential(
        OrderedDict([
            ("bn", nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
            ("relu", nn.ReLU(inplace=True)),
            ("conv", nn.Conv2d(64, nr_types, kernel_size=(1, 1), stride=(1, 1)))
        ])
    )

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
losses = []
# train last layer on new training data
tot_start = time.time()
for epoch in range(n_epochs):
    epoch_loss = 0.0
    start = time.time()
    for batch_idx, batch_data in enumerate(train_dataloader):
        loss = train_step(batch_data, run_info, sparse_labels=sparse_labels, weighted=weighted, loss_method=loss_method)
        epoch_loss += loss
    print('============ epoch {}, loss: {} ================'.format(epoch, epoch_loss))
    losses.append(epoch_loss)
    # torch.save(model.module.state_dict(), epoch_save_path)
    print("epoch elapsed time = ", time.time() - start)

torch.save(model.module.state_dict(), model_save_path)
print("total elapsed time = ", time.time() - tot_start)
print('model path: ', model_save_path)




