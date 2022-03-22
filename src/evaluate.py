'''File for testing constructed models

Code is mainly based on https://github.com/milesial/Pytorch-UNet
'''
import torch
import torch.nn.functional as F
from tqdm import tqdm # Loading bar

# Import dice score functions
from functions import multiclass_dice_coeff, dice_coeff


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
    dice_score = 0

    # Iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
    
        for batch in dataloader:
            image, mask_true = batch[0], batch[1]
            
            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float() # Is model.n_classes number of predicted classes? Yes changed to 2

            with torch.no_grad():
                # predict the mask
                mask_pred = model(image)

                '''
                # convert to one-hot format
                if model.n_classes == 1:
                    mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                    # compute the Dice score
                    dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                else:
                '''
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), 2).permute(0, 3, 1, 2).float() # 2 after dim=1 is number of classes
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
                       
    model.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches    