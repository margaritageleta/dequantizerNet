import os
import gc
import sys
import yaml
import torch
import wandb
import random
import numpy as np
import glob2 as glob
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
# import torch_optimizer as torchoptim

from loader import ImageDataset
from architecture import DequantizerNet as DQNET

DATA_DIR = os.path.join(os.environ.get('DATA_PATH'), f'data')

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

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

def get_data(categories, params):
    print('Preparing data...')
    dataset_train = ImageDataset(
        image_root=DATA_DIR, 
        categories=categories,
        split='train', 
        rgb=True,
        image_extension='JPEG',
        max_limit=params['max_limit'],
        percentage_train=params['percentage_train']
    )
    dataset_valid = ImageDataset(
        image_root=DATA_DIR, 
        categories=categories,
        split='test', 
        rgb=True,
        image_extension='JPEG',
        max_limit=params['max_limit'],
        percentage_train=params['percentage_train']
    )
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print('Dataset prepared.')

    return dataloader_train, dataloader_valid

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

def viz2wandb(img_out, img_in, img_out_pred):
    img_out = img_out.permute(1,2,0).detach().numpy().astype(np.float32)
    img_in = img_in.permute(1,2,0).detach().numpy().astype(np.float32)
    img_out_pred = img_out_pred.permute(1,2,0).detach().numpy().astype(np.float32)

    fig, ax = plt.subplots(1, 3, figsize=(12, 10))
    ax[0].imshow(img_out)
    ax[1].imshow(img_in)
    ax[2].imshow(img_out_pred)
    ax[0].set_title('Original image')
    ax[1].set_title('Quantized image')
    ax[2].set_title('Dequantized image')
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')

    plt.close('all')

    return fig

def gradient_step(data, optimizer, model, criterion, step, num_steps, device):
    img_in, img_out = data[0].to(device), data[1].to(device)
            
    # zero the parameter gradients
    optimizer.zero_grad()

    #forward + backward + optimize
    img_out_pred = model(img_in)

    loss = criterion(img_out_pred, img_out)

    wandb.log({ 'MSE train': loss })
            
    if step % 10 == 0:
        print(f'{np.round(step / num_steps * 100,3)}% | TR Loss: {loss}')
    loss.backward()
    optimizer.step()

    img_in = img_in.cpu()
    img_out = img_out.cpu()
    img_out_pred = img_out_pred.detach().cpu()

    fig = viz2wandb(img_out[0,...], img_in[0,...], img_out_pred[0,...])
    if step % 10 == 0:
         wandb.log({f"Dequantization at step {step}": fig})
    del img_in, img_out, img_out_pred
    gc.collect()
    torch.cuda.empty_cache()
    return loss.cpu().item()

def val_step(data, model, criterion, step, num_steps_vd, device):
    
    img_in, img_out = data[0].to(device), data[1].to(device)
    
    with torch.no_grad():
        img_out_pred = model(img_in)
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

    create_folder(os.path.join(os.environ.get('LOG_PATH'), f'experiment{params["experiment"]}'))

    dataloader_train, dataloader_valid = get_data(categories, params)
    
    epochs = params['epochs']
    batch_size = params['batch_size']
    
    len_dataloader_train = len(dataloader_train)
    len_dataloader_valid = len(dataloader_valid)
    num_steps = len_dataloader_train // batch_size + 1
    num_steps_vd = len_dataloader_valid // batch_size + 1
    
    if len_dataloader_train == 0 or len_dataloader_valid == 0:
        raise Exception('0 samples in the dataloader !!!')
    
    model, device, criterion, optimizer = get_model_components(params)

    init_wandb(params, model)

    best_loss = np.inf
    for epoch in range(epochs): 
       
        model = model.train()
        running_loss = 0.0
        for i, data in tqdm(enumerate(dataloader_train), file=sys.stdout):
            running_loss += gradient_step(data, optimizer, model, criterion, i, num_steps, device)
            tqdm.write(f'[{i+1}/{len_dataloader_train}] TR loss: {running_loss / (i+1)}') 
        
        ## VALIDATION LOOP
        model = model.eval()
        running_loss = 0.0
        for i, data in tqdm(enumerate(dataloader_valid), file=sys.stdout):
            running_loss += val_step(data, model, criterion, i, num_steps_vd, device)
            tqdm.write(f'[{i+1}/{len_dataloader_valid}] VD loss: {running_loss / (i+1)}') 
        if bool(running_loss < best_loss):
            print('Storing a new best model...')
            torch.save(model.state_dict(), os.path.join(os.environ.get('LOG_PATH'), f'experiment{params["experiment"]}/DQNET_weights_{params["experiment"]}.pt'))
            
    print('Finished Training!')