# Train loop for Unet from https://github.com/milesial/Pytorch-UNet
# Lauri has commented some sections but code is untouched or left in comment

import argparse
import logging # logging duh
import sys
import os
from pathlib import Path



import torch
import torch.nn as nn
import torch.nn.functional as F
#import wandb  # logging / visualization
from torch import optim
from torch.utils.data import DataLoader, random_split

# Loading bar
from tqdm import tqdm
from datetime import date, datetime

# Change imports below these are from other modules
# Import data
from data_loader import MaskedDataset
# Import dice score (it's like F1 score)
from functions import dice_loss
# Import testing for main in this file (optional)
from evaluate import evaluate_model
# Import model
from model import Unet

# Paths need to be modified
DATA_DIR = os.path.join(os.getcwd(), "data")
TARGET_DATA_DIR = os.path.join(DATA_DIR, "target")
LIVECELL_IMG_DIR = os.path.join(DATA_DIR, "livecell", "images")
LIVECELL_MASK_DIR = os.path.join(DATA_DIR, "livecell", "masks")
dir_checkpoint = os.path.join(os.getcwd(), "model" )

# Hyperparameter defaults here
def train_net(net,
              dataset,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 0.001,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              amp: bool = False#,
              # n_images = None,
              #in_memory = False
              ):
    
    # NOTE the whole datahandling could be moved somewhere else (sections 1-3)

    # 1. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 2. Create data loaders 
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)   # num_workers is number of cores used, pin_memory enables fast data transfer to CUDA-enabled GPUs
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    
    # 3. Model saving location
    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
    model_number = 2
    save_dir = os.path.join(dir_checkpoint, f'model_{datetime.now().date()}')
    while os.path.exists(save_dir) == True:
        model_number += 1
        save_dir = os.path.join(dir_checkpoint, f'model_{model_number}_{datetime.now().date()}')
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 4. Create Log
    logging.basicConfig(filename=os.path.join(save_dir, 'model_' + str(datetime.now().date()) +'.log'), level=logging.INFO)
    logging.info(f'Training date and time:  {str(datetime.now())}')
    logging.info(f'Starting training:\nEpochs:          {epochs}\nBatch size:      {batch_size}\nLearning rate:   {learning_rate}\nValidation percent: {val_percent}\nTraining size:   {n_train}\nValidation size: {n_val}\nCheckpoints:     {save_checkpoint}\nDevice:          {device.type}\nMixed Precision: {amp}')
       
    # 5. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9) # https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)  # Tries to choose best datatype for operations (torch.float16 for convolutions and float32 for reductions)
    criterion = nn.CrossEntropyLoss()   # Cross entropy loss function
    global_step = 0 # For tqdm

    # 6. Begin training, The actual training loop
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                
                # Prepare data
                images = batch[0]
                true_masks = batch[1]
                # Check that channels match
                ''' 
                #AttributeError: 'Unet' object has no attribute 'n_channels'
                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
                '''
                # Move data to device
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                # Make prediction and calculate loss
                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)        # net is the UNET model
                    loss = criterion(masks_pred, true_masks) \
                           + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                       F.one_hot(true_masks, 2).permute(0, 3, 1, 2).float(),
                                       multiclass=True)     # Loss function is the sum of cross entropy and Dice loss
                           
                # Optimisation step
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                # Update tqdm loading bar
                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                '''
                # Evaluation round
                division_step = (n_train // (10 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
                        logging.info('Validation Dice score: {}'.format(val_score))
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score,
                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                            },
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })
                '''
            logging.info({
            'train loss': loss.item(),
            'step': global_step,
            'epoch': epoch+1
            })

            val_score = evaluate_model(net, val_loader, device)
            scheduler.step(val_score)
            logging.info(f'Validation score of epoch {epoch + 1} was {val_score}')


        if save_checkpoint:
            #Path(dir_checkpoint).makedirs(parents=True, exist_ok=True)
            #Path(os.path.join(dir_checkpoint, f'model_{datetime.now().date()}'))
            torch.save(net, str(os.path.join(save_dir, 'checkpoint_epoch{}_{}.pth'.format(epoch + 1, datetime.now().date()))))
            logging.info(f'Checkpoint {epoch + 1} saved in file checkpoint_epoch{epoch +1}_{datetime.now.date()}.pth!')

'''
# Function to help logging
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')

    return parser.parse_args()
'''

if __name__ == '__main__':
    '''
    # Prepare logging and hyperparameters from get_args
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {device}')
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')
    '''

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = Unet(numChannels=1, classes=2, dropout = 0.1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    dataset = MaskedDataset(LIVECELL_IMG_DIR, LIVECELL_MASK_DIR, length=None, in_memory=False)
    
    try:
        train_net(net=net,
                  dataset = dataset,
                  epochs= 6, # Set epochs
                  batch_size= 2, # Batch size
                  learning_rate=0.008, # Learning rate
                  device=device,
                  val_percent=0.1, # Percent of test set
                  save_checkpoint = True,
                  amp=False
                  #n_images = None,  # How many images per epoch if None goes whole dataset
                  #in_memory = True # If true, load all images into memory at setupx
                  )  
        #torch.save(net.state_dict(), 'model.pth')
    except KeyboardInterrupt:
        pass
        '''
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
        '''