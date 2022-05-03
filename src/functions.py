# Add neural networks related functions, e.g forward, backward functions, training function?
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn

import matplotlib.pyplot as plt 
from datetime import datetime
import os 
from tqdm import tqdm 
import numpy as np

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

def plot_training_loss(losses, labels, folder, show_image = False, save_image = True, name=None):
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
        
def evaluate_model(model, dataloader, device):
    '''Function that is used to test given model
    model : class Unet
        Trained network model
    dataloader : class DataLoader
        Contains data used for evaluating the model
    device : str
        Determines where calculatios are made: CPU or Cuda.
    '''
    model.eval() # Set model to evaluation mode
    num_val_batches = len(dataloader) # Number of batches in dataloader
    
    # Set up binary cross entropy loss
    BCE_loss = nn.BCELoss()
    fn_sigmoid = nn.Sigmoid()
    
    n = 0
    disc_losses = []; dice_losses = []
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch[0], batch[1]
        
        # move images and labels to correct device and type
        image = image.to(device)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        
        mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float() # Is model.n_classes number of predicted classes? Yes changed to 2

        p = float(n + 1 * len(dataloader)) / 1 / len(dataloader)
        lambd = 2. / (1. + np.exp(-10 * p)) - 1
        n += 1 # Is it correct?
        
        with torch.no_grad():
            mask_pred, domain_label = model(image, lambd)
            
            # Compute the Dice score
            mask_pred = (fn_sigmoid(mask_pred) > 0.5).float()
            dice_losses.append(dice_coeff(mask_pred, mask_true, reduce_batch_first=False).item())
            
            # Compute BCE loss
            domain_label = domain_label.flatten()
            label_true = torch.ones(domain_label.size()).to(device)
            disc_losses.append(BCE_loss(domain_label, label_true).item())
            
    model.train()
    
    return np.mean(dice_losses), np.mean(disc_losses)

def evaluate_basic_UNet(model, dataloader, device):
    '''Function that is used to evaluate vanilla UNet
    model : class Unet
        Trained network model
    dataloader : class DataLoader
        Contains data used for evaluating the model
    device : str
        Determines where calculatios are made: CPU or Cuda.
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

    
