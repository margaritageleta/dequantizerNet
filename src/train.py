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

from loader import ImageDataset
from metrics import SSIM, PSNR
from architecture import Generator, Discriminator, ContentLoss

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
    mappings = []
    with open(os.path.join(os.environ.get('ROOT_PATH'), os.environ.get('CAT_FILE'))) as f:
        for line in f:
            (key, i, img) = line.split()
            mappings.append(img)
        
    return params, mappings

def get_data(categories, params):
    print('Preparing data...')
    dataset_train = ImageDataset(
        image_root=DATA_DIR, 
        categories=categories,
        split='train', 
        rgb=True,
        image_extension='JPEG',
        max_limit=params['max_limit'],
        percentage_train=params['percentage_train'],
        pixel_shuffle=params['concat_zero']
    )
    dataset_test = ImageDataset(
        image_root=DATA_DIR, 
        categories=categories,
        split='test', 
        rgb=True,
        image_extension='JPEG',
        max_limit=params['max_limit'],
        percentage_train=params['percentage_train'],
        pixel_shuffle=params['concat_zero']
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

    return dataloader_train, dataloader_test 

def get_model_components(params):
    ## Define adversarial models ##
    discriminator = Discriminator(params)
    generator = Generator(params)
    
    ## Send models to GPU ##
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    discriminator = discriminator.to(device).double()
    generator = generator.to(device).double()
    
    ## Print out model parameters ##
    print('Discriminator # parameters:', sum(param.numel() for param in discriminator.parameters()))
    print('Generator # parameters:', sum(param.numel() for param in generator.parameters()))
    
    ## Define adversarial loss function ##
    adv_criterion = nn.BCEWithLogitsLoss()
    content_criterion = ContentLoss(params).to(device)

    ## Define optimizers for each network ##
    d_optimizer = optim.Adam(
        params=discriminator.parameters(), 
        lr=params['lr'],
        weight_decay=params['weight_decay']
    )
    g_optimizer = optim.Adam(
        params=generator.parameters(), 
        lr=params['lr'],
        weight_decay=params['weight_decay']
    )

    return discriminator, generator, device, adv_criterion, content_criterion, d_optimizer, g_optimizer

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


def update_discriminator(
    discriminator, 
    real_img, 
    real_label, 
    fake_img, 
    fake_label,
    optimizer,
    adv_criterion
):
    # During discriminator model training, enable discriminator model backpropagation.
    for d_parameters in discriminator.parameters():
        d_parameters.requires_grad = True

    discriminator.zero_grad()
    # Calculate the classification score of the discriminator model for real samples:
    real_label_pred = discriminator(real_img)
    d_loss_real = adv_criterion(real_label_pred, real_label)
    
    # Calculate the classification score of the discriminator model for fake samples:
    fake_label_pred = discriminator(fake_img)
    d_loss_fake = adv_criterion(fake_label_pred, fake_label)
    
    d_loss = d_loss_real + d_loss_fake

    d_loss.backward(retain_graph=True)
    
    optimizer.step()
    
    wandb.log({ 'd_loss train': d_loss.item() })
    
    # Calculate the score of the discriminator on real samples and fake samples, 
    # the score of real samples is close to 1, and the score of fake samples is close to 0.
    d_real_probability = torch.sigmoid_(torch.mean(real_label_pred.detach()))
    d_fake_probability = torch.sigmoid_(torch.mean(fake_label_pred.detach()))
    wandb.log({ 'd_real_probability train': d_real_probability.item() })
    wandb.log({ 'd_fake_probability train': d_fake_probability.item() })
    
    # Before generator model training, disable discriminator model backpropagation.
    for d_parameters in discriminator.parameters():
        d_parameters.requires_grad = False
    
    return d_loss

def update_generator(
    generator, 
    discriminator,
    real_img,
    real_label,
    fake_img,
    optimizer, 
    adv_criterion,
    content_criterion,
    params
):

    generator.zero_grad()
    content_loss = params['content_weight'] * content_criterion(fake_img, real_img)
    wandb.log({ 'content_loss train': content_loss.item() })
    adversarial_loss = params['adversarial_weight'] * adv_criterion(discriminator(fake_img), real_label)
    wandb.log({ 'adversarial_loss train': adversarial_loss.item() })
    g_loss = content_loss + adversarial_loss
    g_loss.backward()

    optimizer.step()
    
    wandb.log({ 'g_loss train': g_loss.item() })
    
    return g_loss

def validation(data, generator, step, device):
    
    img_in, img_out = data[0].to(device), data[1].to(device)
    
    with torch.no_grad():
        img_out_pred = generator(img_in)
        ssim = SSIM(img_out, img_out_pred)
        psnr = PSNR(img_out, img_out_pred)
    
    wandb.log({ 'ssim valid': ssim })
    wandb.log({ 'psnr valid': psnr })
    
    if step % 10 == 0:
        #print(f'VD Loss at step {step}: {loss}')
        fig = viz2wandb(
            img_out[0,...].cpu(), 
            torch.narrow(img_in[0,...].cpu().unsqueeze(0),1, 0, 3).squeeze(0),
            img_out_pred[0,...].cpu()
          )
        wandb.log({f"Dequantization at step {step}": fig})

    img_in = img_in.cpu()
    img_out = img_out.cpu()
    img_out_pred = img_out_pred.detach().cpu()
    del img_in, img_out, img_out_pred
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    set_reproductibility()
    ## Load params
    params, categories = get_params()

    ## Create experiment folder ##
    create_folder(os.path.join(os.environ.get('LOG_PATH'), f'experiment{params["experiment"]}'))

    ## Load data ##
    dataloader_train, dataloader_test = get_data(categories, params)
    
    ## Load models, criterion and optimizer ##
    discriminator, generator, device, adv_criterion, content_criterion, d_optimizer, g_optimizer = get_model_components(params)

    ## Initialize ##
    init_wandb(params, generator) #TODO: Can we watch both??
   
    epochs = params['epochs']
    batch_size = params['batch_size']
    
    num_steps = len(dataloader_train) // batch_size
    num_steps_vd = len(dataloader_test) // batch_size
    
    ## Metrics ##
    best_loss = np.inf
    for epoch in tqdm(range(epochs), file=sys.stdout): 
       
        ## TRAINING LOOP #########################################################################
        discriminator = discriminator.train()
        generator = generator.train()
        running_loss = 0.0
        for i, data in enumerate(dataloader_train):
            img_in, img_out = data[0].to(device), data[1].to(device)
            img_out_pred = generator(img_in)
            
            # Used for the output of the discriminator binary classification, 
            # the input sample is from the dataset (real sample) and marked as 1, 
            # and the input sample from the generator (fake sample) is marked as 0.
            real_label = torch.full([img_out.size(0), 1], 1.0, device=device)
            fake_label = torch.full([img_in.size(0), 1], 0.0, device=device)
        
            #tqdm.write('[%d, %5d] TR loss: %.3f' % (epoch + 1, i + 1, running_loss / num_steps))
            
            # (1) Update D network: maximize D(x)-1-D(G(z))
            d_loss = update_discriminator(
                discriminator=discriminator,
                real_img=img_out,
                real_label=real_label,
                fake_img=img_out_pred,
                fake_label=fake_label,
                optimizer=d_optimizer, 
                adv_criterion=adv_criterion
            )
            
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            
            img_out_pred = generator(img_in)##not sure if we must repeat this step
            g_loss = update_generator(
                generator=generator,
                discriminator=discriminator,
                real_img=img_out, 
                real_label=real_label,
                fake_img=img_out_pred,
                optimizer=g_optimizer,
                adv_criterion=adv_criterion,
                content_criterion=content_criterion,
                params=params
            )
            img_in, img_out = img_in.cpu(), img_out.cpu()
            del img_in, img_out
            torch.cuda.empty_cache()
            
        ## VALIDATION LOOP #######################################################################
        generator = generator.eval()
        running_loss = 0.0
        for i, data in enumerate(dataloader_test):
            
            validation(data, generator, i, device) 
            #tqdm.write('[%d, %5d] VD loss: %.3f' % (epoch + 1, i + 1, running_loss / num_steps_vd)) 
        if bool(running_loss < best_loss):
            print('Storing a new best model...')
            ## Save generator state ##
            torch.save(
                generator.state_dict(), 
                os.path.join(os.environ.get('LOG_PATH'), f'experiment{params["experiment"]}/generator_weights_{params["experiment"]}.pt'))
            ## Save generator optimizer state ##
            torch.save(
               g_optimizer.state_dict(), 
               os.path.join(os.environ.get('LOG_PATH'), f'experiment{params["experiment"]}/generator_optimizer_{params["experiment"]}.pt'))
            ## Save discriminator state ##
            torch.save(
                discriminator.state_dict(), 
                os.path.join(os.environ.get('LOG_PATH'), f'experiment{params["experiment"]}/discriminator_weights_{params["experiment"]}.pt'))
            ## Save discriminator optimizer state ##
            torch.save(
               g_optimizer.state_dict(), 
               os.path.join(os.environ.get('LOG_PATH'), f'experiment{params["experiment"]}/discriminator_optimizer_{params["experiment"]}.pt'))
            
        ##########################################################################################
            
    print('Finished Training!')