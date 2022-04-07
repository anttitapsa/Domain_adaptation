# Add neural networks related functions, e.g forward, backward functions, training function?
import torch
from torch import Tensor
import matplotlib.pyplot as plt 
from datetime import datetime
import os 

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

def plot_training_loss(loss_lst, folder, show_image = False, save_image = True, name=None):
    ''' Function that plots training loss and saves the image by default
    loss_lst : list
        List of average training loss per epoch
    folder : str
        Path of diroctory where the plot will be saved
    show_image : boolean
        Show figure after plotting
    save_image : boolean
        Boolean value for image saving
    name : str
        Name for the plot
    '''
    epochs = list(range(len(loss_lst)))
    plt.plot(epochs, loss_lst)
    
    if not name:
        name = "Model trained at " + datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    plt.title(name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    if show_image:
        plt.show()

    if save_image:
        if not os.path.exists(str(folder)):
            os.mkdir(str(folder))
        plt.savefig(str(os.path.join(folder, name.replace(" ", "_"))))
        
