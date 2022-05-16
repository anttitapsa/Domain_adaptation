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
    
    # New version based on https://github.com/fungtion/DANN_py3/blob/master/main.py
    
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
    optim = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Actual training loop
    print("Starting Training Loop...")
    loss_array = np.zeros((4, epochs))
    for epoch in range(epochs):
        n = 0
        loss_training_se = []; loss_training_di = []
        for i, (data_source, data_target) in tqdm(enumerate(zip(source_train_loader, target_train_loader))):
            
            net.zero_grad()
            
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
                    F.one_hot(source_mask.to(torch.int64), 2).permute(0, 3, 1, 2).float(),
                    multiclass=True)
            
            # Add semantic_loss
            loss_training_se.append(semantic_loss.item())
            
            # Calculate BCE loss
            masks_pred, domain_label = net(mix_data, lambd)
            disc_loss = criterion(domain_label.flatten(), true_domain_labels)
            loss_training_di.append(disc_loss.item())
            
            # Optimisation step
            loss = disc_loss + semantic_loss
            loss.backward()
            optim.step()    
        
        
        # Calculate eval losses
        loss_eval_se, loss_eval_di = evaluate_model(model=net, dataloader=test_loader, device=device, model_type="UNET_domainclassifier")
        
        # Add epoc losses to array
        loss_array[:, epoch] = (np.mean(loss_training_se), np.mean(loss_training_di), loss_eval_se, loss_eval_di)
        
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
    
    
    '''
    # Two class version : WORKS CHECKED!
    
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
                    F.one_hot(source_mask.to(torch.int64), 2).permute(0, 3, 1, 2).float(),
                    multiclass=True)     # Loss function Dice loss
            
            # Add semantic_loss
            loss_training_se.append(semantic_loss.item())
            
            # Optimisation step
            optim_source.zero_grad(set_to_none=True)
            grad_scaler.scale(semantic_loss).backward()
            grad_scaler.step(optim_source)
            grad_scaler.update()
            
            # Calculate BCE loss
            optim_target.zero_grad()
            masks_pred, domain_label = net(mix_data, lambd)
            disc_loss = criterion(domain_label.flatten(), true_domain_labels)
            loss_training_di.append(disc_loss.item())
            
            # Optimisation step
            disc_loss.backward()
            optim_target.step()    
        
        
        # Calculate eval losses
        loss_eval_se, loss_eval_di = evaluate_model(model=net, dataloader=test_loader, device=device, model_type="UNET_domainclassifier")
        
        # Add epoc losses to array
        loss_array[:, epoch] = (np.mean(loss_training_se), np.mean(loss_training_di), loss_eval_se, loss_eval_di)
        
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
    '''
    '''
    # One class implementation --> poor results ?
    
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
                    masks_pred.float(),
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
            optim_target.zero_grad()
            masks_pred, domain_label = net(mix_data, lambd)
            disc_loss = criterion(domain_label.flatten(), true_domain_labels)
            loss_training_di.append(disc_loss.item())
            
            # Optimisation step
            disc_loss.backward()
            optim_target.step()    
        
        
        # Calculate eval losses
        loss_eval_se, loss_eval_di = evaluate_model(model=net, dataloader=test_loader, device=device, model_type="UNET_domainclassifier")
        
        # Add epoc losses to array
        loss_array[:, epoch] = (np.mean(loss_training_se), np.mean(loss_training_di), loss_eval_se, loss_eval_di)
        
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
    '''
if __name__ == '__main__':
    
    # Hyperparameters
    epochs = 20
    batch_size = 4
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_classes = 2
    n_layers = 1
    data_length = None
    
    # Set model path
    dir_checkpoint = os.path.join(os.getcwd(), "model" )
    
    # Orginal livecell data
    LC_dataset = data_loader.MaskedDataset(
        data_loader.LIVECELL_IMG_DIR, 
        data_loader.LIVECELL_MASK_DIR, 
        length=data_length, 
        in_memory=False, 
        return_domain_identifier=True, 
        augmented=False)
    
    # Augmented livecell data
    LC_dataset_augmented = data_loader.MaskedDataset(
        data_loader.LIVECELL_IMG_DIR, 
        data_loader.LIVECELL_MASK_DIR, 
        length=data_length, 
        in_memory=False, 
        return_domain_identifier=True, 
        augmented=True)
    '''
    # TEST livecell data. Same dataset as orginal livecell 
    # but TEST sets contain only images BV2_, HUH7_ and SkBr3_.
    LC_test_dataset = data_loader.MaskedDataset(
        data_loader.LIVECELL_TEST_IMG_DIR, 
        data_loader.LIVECELL_TEST_MASK_DIR, 
        length=data_length, 
        in_memory=False, 
        return_domain_identifier=True, 
        augmented=False)
    
    # TEST augmented livecell data
    LC_dataset_aug_test = data_loader.MaskedDataset(
        data_loader.LIVECELL_TEST_IMG_DIR, 
        data_loader.LIVECELL_TEST_MASK_DIR, 
        length=data_length, 
        in_memory=False, 
        return_domain_identifier=True, 
        augmented=True)
    '''
    # Synthetic dataset
    Unity_dataset = data_loader.MaskedDataset(
        data_loader.UNITY_IMG_DIR, 
        data_loader.UNITY_MASK_DIR, 
        length=data_length, 
        in_memory=False, 
        return_domain_identifier=True, 
        augmented=False)
    
    # Target dataset
    target_dataset = data_loader.UnMaskedDataset(
        data_loader.TARGET_DATA_DIR, 
        mode=3, 
        return_domain_identifier=True)
    
    # Evaluation dataset
    evaluation_dataset = data_loader.MaskedDataset(
        data_loader.TEST_IMG_DIR,
        data_loader.TEST_MASK_DIR, 
        length=None, 
        in_memory=False, 
        return_domain_identifier=False, 
        mode=2)
    
    # Augmented livecell + synthetic dataset
    combined_dataset = torch.utils.data.ConcatDataset([LC_dataset_augmented, Unity_dataset])
    #combined_test_dataset = torch.utils.data.ConcatDataset([LC_dataset_aug_test, Unity_dataset])
    
    # Create dataloaders
    LC_loader           = DataLoader(LC_dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True)
    LC_augmented_loader = DataLoader(LC_dataset_augmented, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True)
    #LC_test_loader      = DataLoader(LC_test_dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True)
    #LC_aug_test_loader  = DataLoader(LC_dataset_aug_test, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True)
    unity_loader        = DataLoader(Unity_dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True)
    target_loader       = DataLoader(target_dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True)
    evaluation_loader   = DataLoader(evaluation_dataset, shuffle=True, batch_size=15, num_workers=4, pin_memory=True)
    combined_loader     = DataLoader(combined_dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True)
    #combined_test_loader= DataLoader(combined_test_dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True)
 
    # Model training : combined data
    for layer in range(n_layers):
        network_model = UNET_domainclassifier(
            numChannels = 1, 
            classes = n_classes, 
            dropout = 0.1, 
            image_res = data_loader.IMG_SIZE, 
            domain_classifier_level = layer)
        
        train_loop(net=network_model,
                   datasets=(combined_loader, target_loader, evaluation_loader),
                   device=device,
                   epochs=epochs,
                   learning_rate=learning_rate,
                   model_name = f"BP_{layer}layer_Combined_{epochs}epochs_{batch_size}batch_{data_length}len")
    
    # Model training : orginal livecell data
    for layer in range(n_layers):
        network_model = UNET_domainclassifier(
            numChannels = 1, 
            classes = n_classes, 
            dropout = 0.1, 
            image_res = data_loader.IMG_SIZE, 
            domain_classifier_level = layer)
        
        train_loop(net=network_model,
                   datasets=(LC_loader, target_loader, evaluation_loader),
                   device=device,
                   epochs=epochs,
                   learning_rate=learning_rate,
                   model_name = f"BP_{layer}layer_LiveCell_{epochs}epochs_{batch_size}batch_{data_length}len")
    
    # Model training : augmented livecell data
    for layer in range(n_layers):
        network_model = UNET_domainclassifier(
            numChannels = 1, 
            classes = n_classes, 
            dropout = 0.1, 
            image_res = data_loader.IMG_SIZE, 
            domain_classifier_level = layer)
        
        train_loop(net=network_model,
                   datasets=(LC_augmented_loader, target_loader, evaluation_loader),
                   device=device,
                   epochs=epochs,
                   learning_rate=learning_rate,
                   model_name = f"BP_{layer}layer_AugLiveCell_{epochs}epochs_{batch_size}batch_{data_length}len")
    
    '''  
    # Model training : TEST livecell data
    for layer in range(n_layers):
        network_model = UNET_domainclassifier(
            numChannels = 1, 
            classes = n_classes, 
            dropout = 0.1, 
            image_res = data_loader.IMG_SIZE, 
            domain_classifier_level = layer)
        
        train_loop(net=network_model,
                   datasets=(LC_test_loader, target_loader, evaluation_loader),
                   device=device,
                   epochs=epochs,
                   learning_rate=learning_rate,
                   model_name = f"BP_{layer}layer_TESTLiveCell_{epochs}epochs_{batch_size}batch_{data_length}len")
    
    # Model training : TEST augmented livecell data
    for layer in range(n_layers):
        network_model = UNET_domainclassifier(
            numChannels = 1, 
            classes = n_classes, 
            dropout = 0.1, 
            image_res = data_loader.IMG_SIZE, 
            domain_classifier_level = layer)
        
        train_loop(net=network_model,
                   datasets=(LC_aug_test_loader, target_loader, evaluation_loader),
                   device=device,
                   epochs=epochs,
                   learning_rate=learning_rate,
                   model_name = f"BP_{layer}layer_TESTAugLiveCell_{epochs}epochs_{batch_size}batch_{data_length}len")
    
    # Model training : TEST combined dataset
    for layer in range(n_layers):
        network_model = UNET_domainclassifier(
            numChannels = 1, 
            classes = n_classes, 
            dropout = 0.1, 
            image_res = data_loader.IMG_SIZE, 
            domain_classifier_level = layer)
        
        train_loop(net=network_model,
                   datasets=(combined_test_loader, target_loader, evaluation_loader),
                   device=device,
                   epochs=epochs,
                   learning_rate=learning_rate,
                   model_name = f"BP_{layer}layer_TESTCombined_{epochs}epochs_{batch_size}batch_{data_length}len")
    '''