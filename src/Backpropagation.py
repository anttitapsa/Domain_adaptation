import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datetime import date, datetime
from pathlib import Path
import random
import os
from tqdm import tqdm 
from data_loader import MaskedDataset, UnMaskedDataset

def train_loop(net,
               datasets,
               device,
               model_name,
               epochs=5,
               batch_size=2,
               learning_rate=0.001,
               amp: bool = False):

    # Model saving location
    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
    model_number = 1
    save_dir = os.path.join(dir_checkpoint, f'BackProp_{datetime.now().date()}')
    while os.path.exists(save_dir) == True:
        model_number += 1
        save_dir = os.path.join(dir_checkpoint, f'BackProp_{model_number}_{datetime.now().date()}')
    Path(save_dir).mkdir(parents=True, exist_ok=True)


    source_train_loader, target_train_loader = datasets

    # Set up loss
    criterion = nn.BCELoss()

    # Set up optimisers
    optim_source = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    optim_target = optim.Adam(net.parameters(), lr=learning_rate)

    print("Starting Training Loop...")

    # Actual training loop
    for epoch in range(epochs):
        iters = 0
        for i, (data_source, data_target) in enumerate(zip(tqdm(source_train_loader), target_train_loader), 0):
            
            # Prepare data
            source_im = data_source[0].to(device)
            source_mask = data_source[1].to(device)
            
            target_im = data_target[0].to(device)
            mix_data = source_im.cat(target_im)
            true_domain_labels = data_source[2].cat(data_target[2])
            
            # Teach with source encoder + decoder
            with torch.cuda.amp.autocast(enabled=amp):
                masks_pred, domain_label = net(source_im)
                semantic_loss = dice_loss(F.softmax(masks_pred, dim=1).float(),
                                   F.one_hot(source_mask, 2).permute(0, 3, 1, 2).float(),
                                   multiclass=True)     # Loss function Dice loss
                
            # Optimisation step
            optim_source.zero_grad(set_to_none=True)
            grad_scaler.scale(semantic_loss).backward()
            grad_scaler.step(optim_source)
            grad_scaler.update()
            
            
            # Teach with target encoder + grl
            masks_pred, domain_label = net(mix_data)
            loss = criterion(domain_label, true_domain_labels)
            # Optimisation step
            optim_target.zero_grad()
            loss.backward()
            optim_target.step()    


if __name__ == '__main__':
    DATA_DIR = os.path.join(os.getcwd(), "data")
    TARGET_DATA_DIR = os.path.join(DATA_DIR, "target")
    LIVECELL_IMG_DIR = os.path.join(DATA_DIR, "livecell", "images")
    LIVECELL_MASK_DIR = os.path.join(DATA_DIR, "livecell", "masks")
    UNITY_IMG_DIR = os.path.join(DATA_DIR, "unity_data", "images")
    UNITY_MASK_DIR = os.path.join(DATA_DIR, "unity_data", "masks")
    dir_checkpoint = os.path.join(os.getcwd(), "model" )

    # Create data loaders
    LC_dataset = MaskedDataset(LIVECELL_IMG_DIR, LIVECELL_MASK_DIR, length=None, in_memory=False)
    Unity_dataset = MaskedDataset(UNITY_IMG_DIR, UNITY_MASK_DIR, length=None, in_memory=False)
    datasets = [LC_dataset, Unity_dataset]
    dataset = torch.utils.data.ConcatDataset(datasets)
   
    train_set = dataset
    
    seed = 123
    test_percent = 0.001
    n_test = int(len(dataset) * test_percent)
    n_train = len(dataset) - n_test

    train_set, test_set = torch.utils.data.random_split(dataset, [n_train, n_test], generator=torch.Generator().manual_seed(seed))

    batch_size = 2

     
    source_train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) # num_workers is number of cores used, pin_memory enables fast data transfer to CUDA-enabled GPUs
    # source_val_loader = DataLoader(test_set, shuffle=True, drop_last=True, **loader_args)

    Target_dataset = UnMaskedDataset(TARGET_DATA_DIR, mode=2)

    target_test_percent = 0.01
    n_test_target = int(len(Target_dataset) * target_test_percent)
    n_train_target = len(Target_dataset) - n_test_target
    target_train_set, target_test_set = torch.utils.data.random_split(Target_dataset, [n_train_target, n_test_target],
                                                                      generator=torch.Generator().manual_seed(seed))

    target_train_loader = DataLoader(target_train_set, shuffle=True, batch_size=batch_size, num_workers=4,
                                     pin_memory=True)

    # Create generators and discriminators
    # A = target
    # B = source
    # i.e. G_A2B = Generator from target to source
    G_A2B = Generator()
    G_B2A = Generator()
    D_A = Discriminator()
    D_B = Discriminator()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G_A2B.to(device)
    G_B2A.to(device)
    D_A.to(device)
    D_B.to(device)

    models = (G_A2B, G_B2A, D_A, D_B)
    datasets = (source_train_loader, target_train_loader)

    train_loop(models=models,
               datasets=datasets,
               device=device,
               model_name="test_resize",
               epochs=10,
               batch_size=2)
