import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from models.hovernet.net_desc import create_model
from run_train import TrainManager
from models.hovernet.run_desc import pre_train_step, infer_step

from infer.tile_pretrain import _prepare_patching

# load pretrained model, Consep seg/classification
pretrained_path = "./models/pretrained/hovernet_original_consep_type_tf2pytorch.tar"
net_state_dict = torch.load(pretrained_path)["desc"]

# set params
nr_types = 5
n_epochs = 1

# initialize HoverNet model
model = create_model(mode="original", input_ch=3, nr_types=5, freeze=False)

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
model.module.decoder.tp.u0.conv = nn.Conv2d(64, nr_types, kernel_size=(1,1), stride=(1,1))

# get parameters to update
params_to_update = []
for name,param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        print("\t",name)

# set up optimizer on parameters to update
optimizer = optim.Adam(params_to_update, lr=0.001, betas=(0.9, 0.999))

# run_info from opt.py 
run_info = trainer.model_config['phase_list'][0]['run_info']

# train last layer on new training data
for epoch in range(n_epochs):
    print("epoch: ", epoch)
    for batch_idx, batch_data in enumerate(train_dataloader):
        pre_train_step(batch_data, model, optimizer, nr_types, run_info)
        break

PATH = './models/pretrained/hovernet_consep_consep_ft.tar'
torch.save(model.module.state_dict(), PATH)

# run inference on test data
# just try to use loaded model intead, use run_infer.py
# for batch_idx, batch_data in enumerate(test_dataloader):
#     infer_step(batch_data, model)



# prepare patches for inference


