# Train loop for Unet from https://github.com/milesial/Pytorch-UNet
# Lauri has commented some sections but code is untouched or left in comment

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
from functions import dice_loss, plot_training_loss, evaluate_model, save_losses, evaluate_final
import numpy as np
import transformer

# Hyperparameter defaults here
def train_loop(net,
               datasets,
               device,
               epochs=5,
               model_name = "Unet",
               learning_rate=0.001,
               amp: bool = False):
    
    net.to(device)
    # Model saving location
    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
    model_number = 1
    save_dir = os.path.join(dir_checkpoint, f'Unet_{datetime.now().date()}')
    while os.path.exists(save_dir) == True:
        model_number += 1
        save_dir = os.path.join(dir_checkpoint, f'Unet_{model_number}_{datetime.now().date()}')
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Get data loaders
    source_train_loader, test_loader = datasets
    
    # 5. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    #optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9) # https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    #grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)  # Tries to choose best datatype for operations (torch.float16 for convolutions and float32 for reductions)
    #criterion = nn.CrossEntropyLoss()   # Cross entropy loss function

    # 6. Begin training, The actual training loop
    print("Starting Training Loop...")
    loss_array = np.zeros((2, epochs))
    for epoch in range(epochs):
        net.train()
        loss_training_se = []
        for i, data_source in tqdm(enumerate(source_train_loader)):
                
            # Prepare data
            images = data_source[0].to(device)
            images = transformer.add_background_noise(images)
            masks = data_source[1].to(device)

            # Make prediction and calculate loss
            optimizer.zero_grad()
            #with torch.cuda.amp.autocast(enabled=amp):
            masks_pred = net(images)        # net is the UNET model
            loss = dice_loss(
                F.softmax(masks_pred, dim=1).float(),
                F.one_hot(masks.to(torch.int64), 2).permute(0, 3, 1, 2).float(),
                multiclass=True)    # Loss function is the sum of cross entropy and Dice loss
            # Add semantic_loss
            loss_training_se.append(loss.item())
            
            # Optimisation step
            loss.backward()
            optimizer.step()

        
        # Calculate eval losses
        loss_eval_se = evaluate_model(model=net, dataloader=test_loader, device=device, model_type = "UNET")
        
        # Add epoc losses to array
        loss_array[:, epoch] = (np.mean(loss_training_se), loss_eval_se)
        
        
    # Save model
    torch.save(net.state_dict(), str(os.path.join(save_dir, model_name +".pth")))

    # Save training loss figure
    plot_training_loss(losses = loss_array,
                       labels = ["Training dice", "Evaluation dice"], 
                       folder = save_dir, 
                       show_image = False, save_image = True, name=None)
    
    # Save losses into text file
    save_losses(losses = loss_array, 
                labels = ["Training dice", "Evaluation dice"], 
                folder = save_dir, 
                name='losses.txt')

if __name__ == '__main__':
    
    
    # Hyperparameters
    epochs = 10
    batch_size = 5
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set model path
    dir_checkpoint = os.path.join(os.getcwd(), "model" )
    
    # Orginal livecell data
    LC_dataset = data_loader.MaskedDataset(
        data_loader.LIVECELL_IMG_DIR, 
        data_loader.LIVECELL_MASK_DIR, 
        length=None, 
        in_memory=False, 
        return_domain_identifier=False, 
        augmented=False)
    
    # Augmented livecell data
    LC_dataset_augmented = data_loader.MaskedDataset(
        data_loader.LIVECELL_IMG_DIR, 
        data_loader.LIVECELL_MASK_DIR, 
        length=None, 
        in_memory=False, 
        return_domain_identifier=False, 
        augmented=True)
    
    # Synthetic dataset
    Unity_dataset = data_loader.MaskedDataset(
        data_loader.UNITY_IMG_DIR, 
        data_loader.UNITY_MASK_DIR, 
        length=None, 
        in_memory=False, 
        return_domain_identifier=False, 
        augmented=False)
    
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
    
    # Create dataloaders
    LC_loader           = DataLoader(LC_dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True)
    LC_augmented_loader = DataLoader(LC_dataset_augmented, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True, drop_last=True)
    unity_loader        = DataLoader(Unity_dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True)
    evaluation_loader   = DataLoader(evaluation_dataset, shuffle=True, batch_size=1, num_workers=4, pin_memory=True)
    combined_loader     = DataLoader(combined_dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True)
    
    '''
    # Evaluate
    model = Unet(numChannels=1, classes=2, dropout = 0.1)
    model.load_state_dict(torch.load('', map_location=torch.device('cpu')))
    model.eval()
    dice_value = evaluate_final(model=model, dataloader=evaluation_loader, device='cpu', model_type = "UNET")
    print(f'Dice score: {1-dice_value}')
    '''
    
    # Model training : combined data
    network_model = Unet(numChannels=1, classes=2, dropout = 0.1)
    
    train_loop(net=network_model,
               datasets=(combined_loader, evaluation_loader),
               device=device,
               epochs=epochs,
               learning_rate=learning_rate,
               model_name = f"Unet_Combined_{epochs}epochs_{batch_size}batch_final")
    

    # Model training : orginal livecell data
    network_model = Unet(numChannels=1, classes=2, dropout = 0.1)
    
    train_loop(net=network_model,
               datasets=(LC_loader, evaluation_loader),
               device=device,
               epochs=epochs,
               learning_rate=learning_rate,
               model_name = f"Unet_LiveCell_{epochs}epochs_{batch_size}batch_final")
    
    
    # Model training : augmented livecell data
    network_model = Unet(numChannels=1, classes=2, dropout = 0.1)
    
    train_loop(net=model,
               datasets=(LC_augmented_loader, evaluation_loader),
               device=device,
               epochs=epochs,
               learning_rate=learning_rate,
               model_name = f"Unet_AugmentedLiveCell_{epochs}epochs_{batch_size}batch_final")
    
        
        
        
    '''
    # OLD train loop
    
    # Change here to adapt to your data 
    # now dataset is combined dataset using liveCell and synthetic dataset 
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    
    # Hyperparameters
    epochs = 10
    batch_size = 6
    learning_rate=0.001
    net = Unet(numChannels=1, classes=2, dropout = 0.1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    
    dir_checkpoint = os.path.join(os.getcwd(), "model" )
    # Create datasets
    LC_dataset = data_loader.MaskedDataset(data_loader.LIVECELL_IMG_DIR, data_loader.LIVECELL_MASK_DIR, length=500, in_memory=False, return_domain_identifier=True, augmented=False)
    test_dataset = data_loader.MaskedDataset(data_loader.TEST_IMG_DIR, data_loader.TEST_MASK_DIR, length=None, in_memory=False, return_domain_identifier=False, augmented=False, mode=2)
    
    # Mixed data
    Unity_dataset = data_loader.MaskedDataset(data_loader.UNITY_IMG_DIR, data_loader.UNITY_MASK_DIR, length=None, in_memory=False, return_domain_identifier=True, augmented=False)
    datasets = [LC_dataset, Unity_dataset]
    dataset = torch.utils.data.ConcatDataset(datasets)
    
    # Only Livecell
    dataset = LC_dataset
    
    # Datalaoders
    train_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True)
    # Test set loader
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=1, num_workers=4,
                             pin_memory=True)
    datasets = (train_loader, test_loader)
    
    train_loop(net=net,
              datasets=datasets,
              epochs= epochs,
              learning_rate=learning_rate,
              device=device,
              amp=False,
              model_name = "Unet_testing_noaugment")  
    '''