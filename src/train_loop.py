# Train loop for Unet from https://github.com/milesial/Pytorch-UNet
# Lauri has commented some sections but code is untouched or left in comment
from logger import logger
import os
from pathlib import Path

# Pytorch libraries and modules
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

# Loading bar
from tqdm import tqdm
from datetime import datetime

# Import other modules 
from functions import dice_loss # Import dice score (it's like F1 score)
from evaluate import evaluate_model # Import testing for main in this file (optional)
from data_loader import MaskedDataset
from model import Unet
from logger import logger

# Paths 
DATA_DIR = os.path.join(os.getcwd(), "data")
TARGET_DATA_DIR = os.path.join(DATA_DIR, "target")
LIVECELL_IMG_DIR = os.path.join(DATA_DIR, "livecell", "images")
LIVECELL_MASK_DIR = os.path.join(DATA_DIR, "livecell", "masks")
UNITY_IMG_DIR = os.path.join(DATA_DIR, "unity_data", "images")
UNITY_MASK_DIR = os.path.join(DATA_DIR, "unity_data", "masks")
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
              amp: bool = False
              ):
    
    # NOTE the whole datahandling could be moved somewhere else (sections 1-3)
    
    # 1. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 2. Create data loaders 
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)   # num_workers is number of cores used, pin_memory enables fast data transfer to CUDA-enabled GPUs
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=True, drop_last=True, **loader_args)
    
    # 3. Model saving location
    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
    model_number = 1
    save_dir = os.path.join(dir_checkpoint, f'Unet_lr_{learning_rate}_batch_{batch_size}_{datetime.now().date()}')
    while os.path.exists(save_dir) == True:
        model_number += 1
        save_dir = os.path.join(dir_checkpoint, f'Unet_lr_{learning_rate}_batch_{batch_size}_{datetime.now().date()}_{model_number}')
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 4. Create Tensorboard Log and text file log
    comment = f' batch_size = {batch_size} lr = {learning_rate}'
    tb = SummaryWriter(comment=comment, log_dir=os.path.join(dir_checkpoint, "runs", "UNet", f'UNet_lr_{learning_rate}_batch_{batch_size}_date_{datetime.now().date()}' ))
    images, labels = next(iter(train_loader))
    images = images.to(device) 
    labels = labels.to(device)
    grid = torchvision.utils.make_grid(images)
    grid_labels = torchvision.utils.make_grid(torch.unsqueeze(labels, dim=1))
    tb.add_image("labels", grid_labels)
    tb.add_image("images", grid)
    tb.add_graph(net, images)
    # text logger
    log = logger("UNet", save_dir, ["Dice score", "Cross entropy loss", "Total loss"])

    # 5. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9) # https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)  # Tries to choose best datatype for operations (torch.float16 for convolutions and float32 for reductions)
    criterion = nn.CrossEntropyLoss()   # Cross entropy loss function
    global_step = 0 # For tqdm

    # 6. Begin training, The actual training loop
    log.start(epochs, batch_size,  learning_rate, val_percent, n_train, n_val, save_checkpoint, device, amp)    
    for epoch in range(epochs):
        
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                
                # Prepare data
                images = batch[0]
                true_masks = batch[1]
  
                # Move data to device
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                # Make prediction and calculate loss
                # cuda
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast(enabled=amp):
                        masks_pred = net(images)        # net is the UNET model
                        loss = criterion(masks_pred, true_masks) \
                            + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                        F.one_hot(true_masks, 2).permute(0, 3, 1, 2).float(),
                                        multiclass=True)     # Loss function is the sum of cross entropy and Dice loss
                # cpu
                else:
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
                
                pbar.set_postfix_str(f'loss (batch): {loss.item():.3f}')

                log.update_loss(loss.item(), "Cross entropy loss", global_step)
                tb.add_scalar("Training Loss", loss.item(), global_step)

            val_score = evaluate_model(net, val_loader, device)
            scheduler.step(val_score)

            log.update_loss(loss.item(), "Total loss", epoch)
            log.update_loss(val_score, "Dice score", epoch)
            log.update(loss.item(), epoch, val_score)

            tb.add_scalar("Epoch loss", epoch_loss, epoch)
            tb.add_scalar("Dice Score", val_score, epoch)
            if save_checkpoint:
                torch.save(net, str(os.path.join(save_dir, f'UNet_lr_{learning_rate}_batch_{batch_size}_date_{datetime.now().date()}checkpoint_epoch{epoch + 1}.pth')))
    tb.close()
    log.finish()

if __name__ == '__main__':

    # Change here to adapt to your data 
    # now dataset is combined dataset using liveCell and synthetic dataset 
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel

    # Create neural network
    net = Unet(numChannels=1, classes=2, dropout = 0.1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    
    # Create datasets
    LC_dataset = MaskedDataset(LIVECELL_IMG_DIR, LIVECELL_MASK_DIR, length=None, in_memory=False)
    Unity_dataset = MaskedDataset(UNITY_IMG_DIR, UNITY_MASK_DIR, length=None, in_memory=False)
    datasets = [LC_dataset, Unity_dataset]
    dataset = torch.utils.data.ConcatDataset(datasets)
    
    seed = 123
    test_percent = 0.001
    n_test = int(len(dataset) * test_percent)
    n_train = len(dataset) - n_test
    train_set, test_set = random_split(dataset, [n_train, n_test], generator=torch.Generator().manual_seed(seed))
    

    try:
        batch = [4, 8]
        for i in batch:
            train_net(net=net,
                      dataset = train_set,
                      epochs= 10, # Set epochs
                      batch_size= i, # Batch size
                      learning_rate=0.008, # Learning rate
                      device=device,
                      val_percent=0.01, # Percent of test set
                      save_checkpoint=True,
                      amp=False
                    )  
    except KeyboardInterrupt:
        pass
        '''
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
        '''