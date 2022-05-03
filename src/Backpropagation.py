import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datetime import date, datetime
from pathlib import Path
import random
import os
from tqdm import tqdm 
import data_loader
from model import Unet, UNET_domainclassifier
from functions import dice_loss, plot_training_loss, evaluate_model, save_losses
import numpy as np

def train_loop(net,
               datasets,
               device,
               epochs=5,
               model_name = "Backpropagation",
               learning_rate=0.001,
               amp: bool = False):

    net.to(device)
    # Model saving location
    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
    model_number = 1
    save_dir = os.path.join(dir_checkpoint, f'BackProp_{datetime.now().date()}')
    while os.path.exists(save_dir) == True:
        model_number += 1
        save_dir = os.path.join(dir_checkpoint, f'BackProp_{model_number}_{datetime.now().date()}')
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Get data loaders
    source_train_loader, target_train_loader, test_loader = datasets
    
    # Set up loss
    criterion = nn.BCELoss()

    # Set up optimisers
    optim_source = torch.optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    optim_target = torch.optim.Adam(net.parameters(), lr=learning_rate)

    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)  # Tries to choose best datatype for operations (torch.float16 for convolutions and float32 for reductions)

    print("Starting Training Loop...")

    # Actual training loop
    loss_array = np.zeros((4, epochs))
    for epoch in range(epochs):
        n = 0
        loss_training_se = []; loss_training_di = []
        for i, (data_source, data_target) in tqdm(enumerate(zip(source_train_loader, target_train_loader))):
            
            # Calculate lambda decay related stuff
            len_dataloader = min(len(source_train_loader), len(target_train_loader))
            p = float(n + epoch * len_dataloader) / epochs / len_dataloader
            lambd = 2. / (1. + np.exp(-10 * p)) - 1
            n += 1 # Is it correct?
            
            # Prepare data
            source_im = data_source[0].to(device)
            source_mask = data_source[1].to(device)
            
            target_im = data_target[0].to(device)
            mix_data = torch.cat((source_im, target_im), 0).to(device)
            true_domain_labels = torch.cat((data_source[2], data_target[1]), 0).float().to(device)
            # Teach with source encoder + decoder
            with torch.cuda.amp.autocast(enabled=amp):
                masks_pred, domain_label = net(source_im, lambd)
                semantic_loss = dice_loss(
                    F.softmax(masks_pred, dim=1).float(),
                    torch.unsqueeze(source_mask, dim=1).float(),
                    multiclass=False)     # Loss function Dice loss
            
            # Add semantic_loss
            loss_training_se.append(semantic_loss.item())
            
            # Optimisation step
            optim_source.zero_grad(set_to_none=True)
            grad_scaler.scale(semantic_loss).backward()
            grad_scaler.step(optim_source)
            grad_scaler.update()
            
            # Calculate BCE loss
            masks_pred, domain_label = net(mix_data, lambd)
            disc_loss = criterion(domain_label.flatten(), true_domain_labels)
            loss_training_di.append(disc_loss.item())
            
            # Optimisation step
            optim_target.zero_grad()
            disc_loss.backward()
            optim_target.step()    
        
        '''
        # Calculate eval losses
        loss_eval_se, loss_eval_di = evaluate_model(model=net, dataloader=test_loader, device=device)
        
        # Add epoc losses to array
        loss_array[:, epoch] = (np.mean(loss_training_se), np.mean(loss_training_di), loss_eval_se, loss_eval_di)
        '''
        
    # Save model
    torch.save(net.state_dict(), str(os.path.join(save_dir, model_name +".pth")))

    # Save training loss figure
    plot_training_loss(losses = loss_array,
                       labels = ["Training dice", "Training BCE", "Evaluation dice", "Evaluation BCE"], 
                       folder = save_dir, 
                       show_image = False, save_image = True, name=None)
    
    # Save losses into text file
    save_losses(losses = loss_array, 
                labels = ["Training dice", "Training BCE", "Evaluation dice", "Evaluation BCE"], 
                folder = save_dir, 
                name='losses.txt')

if __name__ == '__main__':
    # Hyperparameters
    epochs = 10
    batch_size = 8
    learning_rate=0.001
    
    dir_checkpoint = os.path.join(os.getcwd(), "model" )
    # Create data loaders
    LC_dataset = data_loader.MaskedDataset(data_loader.LIVECELL_IMG_DIR, data_loader.LIVECELL_MASK_DIR, length=None, in_memory=False, return_domain_identifier=True, augmented=True)
    test_dataset = data_loader.MaskedDataset(data_loader.TEST_IMG_DIR, data_loader.TEST_MASK_DIR, length=None, in_memory=False, return_domain_identifier=False)
    '''
    # Mixed data
    Unity_dataset = data_loader.MaskedDataset(data_loader.UNITY_IMG_DIR, data_loader.UNITY_MASK_DIR, length=None, in_memory=False, return_domain_identifier=True, augmented=False)
    datasets = [LC_dataset, Unity_dataset]
    dataset = torch.utils.data.ConcatDataset(datasets)
    '''
    # Only Livecell
    dataset = LC_dataset
   
    train_set = dataset
    seed = 123
    test_percent = 0.001
    n_test = int(len(dataset) * test_percent)
    n_train = len(dataset) - n_test

    # train_set, test_set = torch.utils.data.random_split(dataset, [n_train, n_test], generator=torch.Generator().manual_seed(seed))
     
    source_train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) # num_workers is number of cores used, pin_memory enables fast data transfer to CUDA-enabled GPUs
    # source_val_loader = DataLoader(test_set, shuffle=True, drop_last=True, **loader_args)
    target_dataset = data_loader.UnMaskedDataset(data_loader.TARGET_DATA_DIR, mode=1, return_domain_identifier=True)

    target_test_percent = 0 #0.01
    n_test_target = int(len(target_dataset) * target_test_percent)
    n_train_target = len(target_dataset) - n_test_target
    target_train_set, target_test_set = torch.utils.data.random_split(target_dataset, [n_train_target, n_test_target],
                                                                      generator=torch.Generator().manual_seed(seed))

    target_train_loader = DataLoader(target_train_set, shuffle=True, batch_size=batch_size, num_workers=4,
                                     pin_memory=True)

    # Test set loader
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size, num_workers=4,
                             pin_memory=True)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    datasets = (source_train_loader, target_train_loader, test_loader)
    
    train_loop(net=UNET_domainclassifier(numChannels=1, classes=1, dropout = 0.1, image_res=data_loader.IMG_SIZE, domain_classifier_level=0),
               datasets=datasets,
               device=device,
               epochs=epochs,
               learning_rate=learning_rate,
               model_name = "Backprop_10epochs")
