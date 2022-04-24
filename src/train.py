import os
import gc
import sys
import glob
import yaml
import time
import torch
import numba
import wandb
import random
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
# import torch_optimizer as torchoptim

from loader import ImageDataset
from model.models import DQNET

## Seed for reproducibility.
seed = 2022 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    ## Load params
    with open(os.path.join(os.environ.get("ROOT_PATH"), 'params.yaml'), 'r') as f:
        params = yaml.safe_load(f)

    ## Load data
    print('Preparing data...')
    DATA_DIR = os.path.join(os.environ.get('DATA_PATH'), f'data')
    mappings = []
    with open(os.path.join(os.environ.get('ROOT_PATH'), os.environ.get('CAT_FILE'))) as f:
        for line in f:
            (key, i, img) = line.split()
            mappings.append(img)

    dataset_train = ImageDataset(
        image_root=DATA_DIR, 
        categories=mappings,
        split='train', 
        rgb=True,
        image_extension='JPEG'
    )
    dataset_test = ImageDataset(
        image_root=DATA_DIR, 
        categories=mappings,
        split='test', 
        rgb=True,
        image_extension='JPEG'
    )
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print('Dataset prepared.')
    
    model = DQNET(params)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    model = model.to(device)
    
    criterion = nn.MSELoss() # loss functions

    optimizer = optim.Adam(
        params=model.parameters(), 
        lr=params['lr'],
        weight_decay=params['weight_decay']
    )
    
    experiment = params['experiment']
    epochs = params['epochs']
    dataloader_train_len = len(dataloader_train)
    dataloader_test_len = len(dataloader_test)
    num_steps = dataloader_train_len // batch_size
    num_steps_vd = dataloader_test_len // batch_size
    
    ## Initialize wandb
    os.environ["WANDB_START_METHOD"] = "thread"
    ## Automate tag creation on run launch:
    wandb_tags = []
    wandb_tags.append(f"lr {params['lr']}")
    wandb.init(
        project='DQNET',
        dir=os.environ.get('LOG_PATH'),
        tags=wandb_tags,
        # resume='allow',
    )
    wandb.run.name = f'Experiment #{experiment}'
    wandb.run.save()
    print('Wandb ready.')
    wandb.watch(model)

    best_loss = np.inf
    for epoch in tqdm(range(epochs), file=sys.stdout): 

        ## TRAINING LOOP
        model = model.train()
        running_loss = 0.0
        for i, data in enumerate(dataloader_train):
            
            img_in, img_out = data[0].to(device), data[1].to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            img_out_pred = model(img_in).cpu()

            loss = criterion(img_out_pred, img_out)

            wandb.log({ 'BCE train': loss })
            
            if k % 10 == 0:
                print(f'{np.round(k / num_steps * 100,3)}% | TR Loss: {loss}')
            loss.backward()
            optimizer.step()

            img_in = img_in.cpu()
            img_out = img_out.cpu()
            img_out_pred = img_out_pred.detach().cpu()
            del img_in, img_out, img_out_pred
            gc.collect()
            torch.cuda.empty_cache()

            # print statistics
            running_loss += loss.item()
        tqdm.write('[%d, %5d] TR loss: %.3f' % (epoch + 1, i + 1, running_loss / num_steps)) 
        
        ## VALIDATION LOOP
        model = model.eval()
        running_loss = 0.0
        for i, data in enumerate(dataloader_test):

            img_in, img_out = data[0].to(device), data[1].to(device)
            
            with torch.no_grad():
                img_out_pred = model(img_in).cpu()
                loss = criterion(img_out_pred, img_out)
            
            wandb.log({ 'BCE valid': loss })
            
            if k % 10 == 0:
                print(f'{np.round(k / num_steps_vd * 100,3)}% | VD Loss: {loss}')

            img_in = img_in.cpu()
            img_out = img_out.cpu()
            img_out_pred = img_out_pred.detach().cpu()
            del img_in, img_out, img_out_pred
            gc.collect()
            torch.cuda.empty_cache()

            # print statistics
            running_loss += loss.item()
        tqdm.write('[%d, %5d] VD loss: %.3f' % (epoch + 1, i + 1, running_loss / num_steps_vd)) 
        if bool(running_loss < best_loss):
            print('Storing a new best model...')
            torch.save(model.state_dict(), os.path.join(os.environ.get('LOG_PATH'), f'experiment{experiment}/DQNET_weights_{experiment}.pt'))
            
    print('Finished Training!')