# Add neural networks related functions, e.g forward, backward functions, training function?
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms

import matplotlib.pyplot as plt 
from datetime import datetime
import os 
from tqdm import tqdm 
import numpy as np
from torchvision import transforms
# Training example layout -- Code from Pytorch tutorial
'''
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
'''


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

def plot_training_loss(losses, labels, folder, show_image = False, save_image = True, name=None, y_lims=(0,2)):
    ''' Function that plots training loss and saves the image by default
    losses : numpy.array (n_losses, n_epochs)
        Array of epoch losses. Contains four different
        losses in following order: losses of semenatic and 
        discriminator in training and same but for evaluation. 
    labels : list
        Labels of losses
    folder : str
        Path of diroctory where the plot will be saved
    show_image : boolean
        Show figure after plotting
    save_image : boolean
        Boolean value for image saving
    name : str
        Name for the plot
    y_lims : tuple
        Limit y-axis if given
    '''
    epochs = list(range(losses.shape[1]))
    for n_loss in range(losses.shape[0]):
        plt.plot(epochs, losses[n_loss,:], label = labels[n_loss])
    
    if not name:
        name = "Model trained at " + datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    plt.title(name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    # Limit y-axis
    plt.ylim(y_lims)
    
    if show_image:
        plt.show()

    if save_image:
        if not os.path.exists(str(folder)):
            os.mkdir(str(folder))
        plt.savefig(str(os.path.join(folder, name.replace(" ", "_"))))
        
def save_losses(losses, labels, folder, name='losses.txt'):
    ''' Function that save losses into text file
    losses : numpy.array (n_losses, n_epochs)
        Array of epoch losses. Contains four different
        losses in following order: losses of semenatic and 
        discriminator in training and same but for evaluation. 
    labels : list
        Labels of losses
    folder : str
        Path of diroctory where the plot will be saved
    '''   
    with open(os.path.join(folder, name), 'w') as f:
        for n_epoch in range(losses.shape[1]):
            f.write(f"Epoch {n_epoch}: ")
            for n_loss in range(losses.shape[0]):
                f.write(f"{labels[n_loss]} is {losses[n_loss, n_epoch]:0.6f}")
                if n_loss != losses.shape[0]-1: 
                    f.write(", ")
            f.write('\n')
        
def evaluate_model(model, dataloader, device, model_type, n_epoch = 3):
    '''Function that is used to test given model
    model : class Unet
        Trained network model
    dataloader : class DataLoader
        Contains data used for evaluating the model
    device : str
        Determines where calculatios are made: CPU or Cuda.
    '''
    model.eval()
    BCE_loss = nn.BCELoss() # Set up binary cross entropy loss
    semantic_losses = []; discriminator_losses = []
    n = 0 # Used to calculate lambda
    
    for epoch in range(n_epoch):
        for images, masks in tqdm(dataloader, total=len(dataloader), desc='Validation round', unit='batch', leave=False):
            images = images.to(device)
            masks = masks.to(device)
            
            with torch.no_grad():
                if model_type == "UNET":
                    mask_pred = model(images)
                    
                    # Calculate dice loss
                    #semantic_loss = dice_loss(mask_pred.float(),
                    #                          torch.unsqueeze(masks, dim=1).float())
                    
                    # Two class version
                    semantic_loss = dice_loss(
                        F.softmax(mask_pred, dim=1).float(),
                        F.one_hot(masks.to(torch.int64), 2).permute(0, 3, 1, 2).float(),
                        multiclass=True
                        )
                    
                    
                    semantic_losses.append(semantic_loss.item())
                    
                elif model_type == "UNET_domainclassifier":
                    # Determine lambda that is used for lambda decay
                    p = float(n + 1 * len(dataloader)) / 1 / len(dataloader)
                    lambd = 2. / (1. + np.exp(-10 * p)) - 1
                    n += 1 # Is it correct?
                    
                    mask_pred, domain_label = model(images, lambd)
                    
                    # Compute BCE loss
                    label_true = torch.ones(domain_label.size()).to(device).flatten()
                    discriminator_loss = BCE_loss(domain_label.flatten(), label_true)
                    discriminator_losses.append(discriminator_loss.item())
                    
                    # Calculate dice loss
                    #semantic_loss = dice_loss(mask_pred.float(),
                    #                          torch.unsqueeze(masks, dim=1).float())
                    
                    # Two class version
                    semantic_loss = dice_loss(
                        F.softmax(mask_pred, dim=1).float(),
                        F.one_hot(masks.to(torch.int64), 2).permute(0, 3, 1, 2).float(),
                        multiclass=True
                        )
                    
                    semantic_losses.append(semantic_loss.item()) 
                
                else:
                    raise Exception("Unkown model type!")
                 
    model.train()
    
    # Return values are based on the model type
    if model_type == "UNET":
        return np.mean(semantic_losses)
    elif model_type == "UNET_domainclassifier":
        return np.mean(semantic_losses), np.mean(discriminator_losses)

def evaluate_final(model, dataloader, device, model_type):
    '''Function that is used to test given model
    model : class Unet
        Trained network model
    dataloader : class DataLoader
        Contains data used for evaluating the model
    device : str
        Determines where calculatios are made: CPU or Cuda.
    '''
    model.eval()
    BCE_loss = nn.BCELoss() # Set up binary cross entropy loss
    semantic_losses = []; discriminator_losses = []
    img_size = 512
    imgwidth = 2064
    imgheight = 1544
    
    for images, masks in tqdm(dataloader, total=len(dataloader), desc='Validation round', unit='batch', leave=False):
        images = images.to(device)
        masks = masks.to(device)
        
        with torch.no_grad():
            if model_type == "UNET":

                for i in range(4,imgheight-img_size,img_size):
                    for j in range(8,imgwidth-img_size,img_size):
                        box = (j, i, img_size, img_size)
                        crop_image = transforms.functional.crop(images, *box)
                        crop_mask = transforms.functional.crop(masks, *box)
                        #print('crop_image', crop_image.shape)
                        
                        resize = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST)
                        crop_image = resize.forward(crop_image)
                        crop_mask = torch.squeeze(resize.forward(torch.unsqueeze(crop_mask, dim=0)),dim=0)
                        
                        mask_pred = model(crop_image)
                        # Two class version
                        semantic_loss = dice_loss(
                            F.softmax(mask_pred, dim=1).float(),
                            F.one_hot(crop_mask.to(torch.int64), 2).permute(0, 3, 1, 2).float(),
                            multiclass=True
                            )
                        semantic_losses.append(semantic_loss.item())
                
            elif model_type == "UNET_domainclassifier":
                lambd = 1
                
                for i in range(4,imgheight-img_size,img_size):
                    for j in range(8,imgwidth-img_size,img_size):
                        box = (j, i, img_size, img_size)
                        crop_image = transforms.functional.crop(images, *box)
                        crop_mask = transforms.functional.crop(masks, *box)
                        
                        resize = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST)
                        crop_image = resize.forward(crop_image)
                        crop_mask = torch.squeeze(resize.forward(torch.unsqueeze(crop_mask, dim=0)),dim=0)
                        
                        mask_pred, domain_label = model(crop_image, lambd)
                        
                        # Compute BCE loss
                        label_true = torch.ones(domain_label.size()).to(device).flatten()
                        discriminator_loss = BCE_loss(domain_label.flatten(), label_true)
                        discriminator_losses.append(discriminator_loss.item())
                        # Two class version
                        semantic_loss = dice_loss(
                            F.softmax(mask_pred, dim=1).float(),
                            F.one_hot(crop_mask.to(torch.int64), 2).permute(0, 3, 1, 2).float(),
                            multiclass=True
                            )
                        
                        semantic_losses.append(semantic_loss.item()) 
            
            else:
                raise Exception("Unkown model type!")
    
    # Return values are based on the model type
    if model_type == "UNET":
        return np.mean(semantic_losses)
    elif model_type == "UNET_domainclassifier":
        return np.mean(semantic_losses), np.mean(discriminator_losses)


#def evaluate_basic_UNet(model, dataloader, device):
    '''Function that is used to evaluate vanilla UNet
    model : class Unet
        Trained network model
    dataloader : class DataLoader
        Contains data used for evaluating the model
    device : str
        Determines where calculatios are made: CPU or Cuda.
    '''
    '''
    model.eval()
    semantic_losses = []
    for images, masks in tqdm(dataloader, total=len(dataloader), desc='Validation round', unit='batch', leave=False):
        images = images.to(device)
        masks = masks.to(device)
        
        with torch.no_grad():
            mask_pred = model(images)
            semantic_loss = dice_loss(F.softmax(mask_pred, dim=1).float(),
                                      torch.unsqueeze(masks, dim=1).float())
            semantic_losses.append(semantic_loss.item())        
    model.train()
    
    return np.mean(semantic_losses)
    '''
