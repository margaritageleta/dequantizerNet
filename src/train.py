import os
import gc
import sys
import yaml
import torch
import wandb
import random
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
# import torch_optimizer as torchoptim

from loader import ImageDataset
from architecture import DequantizerNet as DQNET

DATA_DIR = os.path.join(os.environ.get('DATA_PATH'), f'data')

def set_reproductibility(seed= 2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

def get_params(fileName = "params.yaml"):
    with open(os.path.join(os.environ.get("ROOT_PATH"), fileName), 'r') as f:
        params = yaml.safe_load(f)
    
    batch_size = params["batch_size"]
    mappings = []
    with open(os.path.join(os.environ.get('ROOT_PATH'), os.environ.get('CAT_FILE'))) as f:
        for line in f:
            (key, i, img) = line.split()
            mappings.append(img)
        
    return params, mappings, batch_size

def get_data(categories, batch_size):
    print('Preparing data...')
    dataset_train = ImageDataset(
        image_root=DATA_DIR, 
        categories=categories,
        split='train', 
        rgb=True,
        image_extension='JPEG'
    )
    dataset_test = ImageDataset(
        image_root=DATA_DIR, 
        categories=categories,
        split='test', 
        rgb=True,
        image_extension='JPEG'
    )
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print('Dataset prepared.')

    return dataloader_train, dataloader_test 

def get_model_components(params):
    model = DQNET(params)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    model = model.to(device).double()
    
    criterion = nn.MSELoss() # loss functions

    optimizer = optim.Adam(
        params=model.parameters(), 
        lr=params['lr'],
        weight_decay=params['weight_decay']
    )

    return model, device, criterion, optimizer

def init_wandb(params, model):
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
    wandb.run.name = f'Experiment #{params["experiment"]}'
    wandb.run.save()
    print('Wandb ready.')
    wandb.watch(model)

def gradient_step(data, optimizer, model, criterion, step, num_steps, device):
    img_in, img_out = data[0].to(device), data[1].to(device)
            
    # zero the parameter gradients
    optimizer.zero_grad()

    #forward + backward + optimize
    img_out_pred = model(img_in).cpu()

    loss = criterion(img_out_pred, img_out)

    wandb.log({ 'MSE train': loss })
            
    if step % 10 == 0:
        print(f'{np.round(step / num_steps * 100,3)}% | TR Loss: {loss}')
    loss.backward()
    optimizer.step()

    img_in = img_in.cpu()
    img_out = img_out.cpu()
    img_out_pred = img_out_pred.detach().cpu()
    del img_in, img_out, img_out_pred
    gc.collect()
    torch.cuda.empty_cache()
    return loss.item()

def val_step(data, model, criterion, step, num_steps_vd, device):
    
    img_in, img_out = data[0].to(device), data[1].to(device)
    
    with torch.no_grad():
        img_out_pred = model(img_in).cpu()
        loss = criterion(img_out_pred, img_out)
    
    wandb.log({ 'BCE valid': loss })
    
    if step % 10 == 0:
        print(f'{np.round(i / num_steps_vd * 100,3)}% | VD Loss: {loss}')

    img_in = img_in.cpu()
    img_out = img_out.cpu()
    img_out_pred = img_out_pred.detach().cpu()
    del img_in, img_out, img_out_pred
    gc.collect()
    torch.cuda.empty_cache()

    return loss.item()


if __name__ == '__main__':
    set_reproductibility()
    ## Load params
    params, categories, batch_size = get_params()

    dataloader_train, dataloader_test = get_data(categories, batch_size)
    
    model, device, criterion, optimizer = get_model_components(params)

    init_wandb(params, model)
   
    epochs = params['epochs']
    
    num_steps = len(dataloader_train) // batch_size
    num_steps_vd = len(dataloader_test) // batch_size
    
    
    best_loss = np.inf
    for epoch in tqdm(range(epochs), file=sys.stdout): 
       
        model = model.train()
        running_loss = 0.0
        for i, data in enumerate(dataloader_train):
            running_loss += gradient_step(data, optimizer, model, criterion, i, num_steps, device)
        
        tqdm.write('[%d, %5d] TR loss: %.3f' % (epoch + 1, i + 1, running_loss / num_steps)) 
        
        ## VALIDATION LOOP
        model = model.eval()
        running_loss = 0.0
        for i, data in enumerate(dataloader_test):
            
            running_loss += val_step(data, model, criterion, i, num_steps_vd, device)
        tqdm.write('[%d, %5d] VD loss: %.3f' % (epoch + 1, i + 1, running_loss / num_steps_vd)) 
        if bool(running_loss < best_loss):
            print('Storing a new best model...')
            torch.save(model.state_dict(), os.path.join(os.environ.get('LOG_PATH'), f'experiment{params["experiment"]}/DQNET_weights_{params["experiment"]}.pt'))
            
    print('Finished Training!')