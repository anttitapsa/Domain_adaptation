# based on https://towardsdatascience.com/cycle-gan-with-pytorch-ebe5db947a99
# full code in article https://colab.research.google.com/drive/1AcldVfgaalxLtBrXZqzBTvYpbuGawT5o?usp=sharing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datetime import date, datetime
from pathlib import Path
import random
import os
import sys
from tqdm import tqdm 
from data_loader import MaskedDataset, UnMaskedDataset, EmptyLiveCELLDataset
from logger import logger
import checkpoint_saver

chanels = 1
class ResBlock(nn.Module):
    def __init__(self, features):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(  nn.Conv2d(  in_channels=features,
                                                out_channels=features, 
                                                kernel_size=3, 
                                                stride=1,
                                                padding=1),
                                    nn.InstanceNorm2d(features),
                                    nn.ReLU(),
                                    nn.Conv2d(  in_channels=features,
                                                out_channels=features, 
                                                kernel_size=3, 
                                                stride=1,
                                                padding=1))
        self.norm = nn.InstanceNorm2d(features)

    def forward(self, x):
        y = self.conv.forward(x)
        y = self.norm(y+x)
        return F.relu(y)


class Generator(nn.Module):
    def __init__(self, features=64, blocks=1):
        super(Generator, self).__init__()
        self.conv1 = nn.Sequential( nn.ReflectionPad2d(3),
                                    nn.Conv2d(  in_channels=chanels,
                                                out_channels=features, 
                                                kernel_size=7, 
                                                stride=1,
                                                padding=0),
                                    nn.InstanceNorm2d(features),
                                    nn.ReLU(True),
                                    nn.Conv2d(  in_channels=features,
                                                out_channels=2*features, 
                                                kernel_size=3, 
                                                stride=2,
                                                padding=1),
                                    nn.InstanceNorm2d(2*features),
                                    nn.ReLU(True),
                                    nn.Conv2d(  in_channels=2*features,
                                                out_channels=4*features, 
                                                kernel_size=3, 
                                                stride=2,
                                                padding=1),
                                    nn.InstanceNorm2d(4*features),
                                    nn.ReLU(True))
        resblocks = []
        for i in range(int(blocks)):
            resblocks.append(ResBlock(4*features))
        
        self.ResBlocks = nn.Sequential(*resblocks)

        self.conv2 = nn.Sequential( nn.ConvTranspose2d( in_channels= 4* features, 
                                                        out_channels= 4*2* features,
                                                        kernel_size=3,
                                                        stride=1,
                                                        padding=1),
                                    nn.PixelShuffle(2),
                                    nn.InstanceNorm2d(2*features),
                                    nn.ReLU(True),
                                    nn.ConvTranspose2d( in_channels= 2* features, 
                                                        out_channels= 4* features,
                                                        kernel_size=3,
                                                        stride=1,
                                                        padding=1),
                                    nn.PixelShuffle(2),
                                    nn.InstanceNorm2d(features),
                                    nn.ReLU(True),
                                    nn.ReflectionPad2d(3),
                                    nn.Conv2d(  in_channels=features,
                                                out_channels=chanels, 
                                                kernel_size=7, 
                                                stride=1,
                                                padding=0),
                                    nn.Tanh())
    

    def forward(self, x):
        y = self.conv1.forward(x)
        y = self.ResBlocks.forward(y)
        return self.conv2.forward(y)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.disc1 = nn.Sequential(  nn.Conv2d(  in_channels=chanels,
                                                out_channels=64, 
                                                kernel_size=4, 
                                                stride=2,
                                                padding=1,
                                                bias=False),
                                    nn.LeakyReLU(negative_slope= 0.2, inplace=True),
                                    nn.Conv2d(  in_channels=64,
                                                out_channels=2*64, 
                                                kernel_size=4, 
                                                stride=2,
                                                padding=1,
                                                bias=False),
                                    nn.InstanceNorm2d(64*2))

        self.disc2 = nn.Sequential(nn.LeakyReLU(negative_slope= 0.2, inplace=True),
                                    nn.Conv2d(  in_channels=64*2,
                                                out_channels=4*64, 
                                                kernel_size=4, 
                                                stride=2,
                                                padding=1,
                                                bias=False),
                                    nn.InstanceNorm2d(64*4),
                                    nn.LeakyReLU(negative_slope= 0.2, inplace=True),
                                    nn.Conv2d(  in_channels=64*4,
                                                out_channels=8*64, 
                                                kernel_size=4, 
                                                stride=1,
                                                padding=1),
                                    nn.InstanceNorm2d(64*8),
                                    nn.LeakyReLU(negative_slope= 0.2, inplace=True),
                                    nn.Conv2d(in_channels=64*8,
                                                out_channels=1, 
                                                kernel_size=4, 
                                                stride=1,
                                                padding=1))
    def forward(self, x):
        y =  self.disc1.forward(x)
        return self.disc2.forward(y)


def LSGAN_D(real, fake):
    return (torch.mean((real - 1)**2) + torch.mean(fake**2))

def LSGAN_G(fake):
    return torch.mean((fake - 1)**2)

def train_loop(models,
               datasets,
               device,
               model_name,
               epochs=5,
               batch_size=2,
               learning_rate=0.0002,
               betas=(0.5, 0.999),
               save_checkpoint=True,
               Resume=False,
               Pause_path= ""):

    # Lists to save the losses
    '''
    img_list = []
    '''
    G_losses = []
    D_A_losses = []
    D_B_losses = []

    FDL_A2B = []
    FDL_B2A = []
    CL_A = []
    CL_B = []
    ID_B2A = []
    ID_A2B = []
    disc_A = []
    disc_B = [] 

    '''
    FDL_A2B_t = []
    FDL_B2A_t = []
    CL_A_t = []
    CL_B_t = []
    ID_B2A_t = []
    ID_A2B_t = []
    disc_A_t = []
    disc_B_t = []
    '''

    G_A2B, G_B2A, D_A, D_B = models
    source_train_loader, target_train_loader = datasets

    # Set up loss
    criterion = nn.L1Loss()

    # Set up optimisers
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=learning_rate, betas=betas)
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=learning_rate, betas=betas)
    optimizer_G_A2B = torch.optim.Adam(G_A2B.parameters(), lr=learning_rate, betas=betas)
    optimizer_G_B2A = torch.optim.Adam(G_B2A.parameters(), lr=learning_rate, betas=betas)

    optimizers = (optimizer_G_A2B,optimizer_G_B2A,optimizer_D_A,optimizer_D_B)

    #total =  len(target_train_loader) if len(source_train_loader)< len (target_train_loader) else len(source_train_loader)
    
    #Continue interrupted training

    if Resume:
        print("Loading last epoch...")
        models, optimizers, start_epoch, tb_dir, save_dir, losses, global_step = checkpoint_saver.load_paused_training(Pause_path, model =[G_A2B, G_B2A, D_A, D_B], optimizer=[optimizer_G_A2B, optimizer_G_B2A, optimizer_D_A, optimizer_D_B])
        G_A2B, G_B2A, D_A, D_B = models
        optimizer_G_A2B, optimizer_G_B2A, optimizer_D_A, optimizer_D_B = optimizers
        G_losses, D_A_losses, D_B_losses = losses

        global_step = global_step
        # Logger and tensorboard
        tb = SummaryWriter(log_dir=tb_dir)

        if save_checkpoint:
            saver = checkpoint_saver.Checkpoint_saver(epoch=start_epoch-1, model=(G_A2B, G_B2A, D_A, D_B), optimizer=(optimizer_G_A2B, optimizer_G_B2A, optimizer_D_A, optimizer_D_B), checkpoint_dir=save_dir, tb_dir=tb_dir, losses=[G_losses, D_A_losses, D_B_losses])
    
    else:
        # Model saving location
        Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
        model_number = 1
        save_dir = os.path.join(dir_checkpoint, f'CycleGan_{datetime.now().date()}')
        while os.path.exists(save_dir) == True:
            model_number += 1
            save_dir = os.path.join(dir_checkpoint, f'CycleGan_{datetime.now().date()}_{model_number}')
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        start_epoch = 0
        global_step = 0
        
        # Logger and tensorboard
        comment = f' batch size = {batch_size}, learning rate = {learning_rate}, betas = {betas}'
        log_dir = os.path.join(dir_checkpoint, "runs", "CycleGan", f'CycleGan_{datetime.now().date()}_{model_number}') + comment
        tb = SummaryWriter(log_dir=log_dir)
        sources, s_labels = next(iter(datasets[0]))
        targets = next(iter(datasets[1]))
        sources = sources.to(device)
        targets = targets.to(device)
        source_grid = torchvision.utils.make_grid(sources)
        target_grid = torchvision.utils.make_grid(targets)
        tb.add_image("source domain", source_grid)
        tb.add_image("target domain", target_grid)

        if save_checkpoint:
            saver = checkpoint_saver.Checkpoint_saver(epoch=0, model=models, optimizer=optimizers, checkpoint_dir=save_dir, tb_dir=log_dir, losses=[G_losses, D_A_losses, D_B_losses])

    
    #data = enumerate(zip(source_train_loader, target_train_loader), 0)
    #total = len(list(data))

    total = len(source_train_loader)
    
    log = logger("CycleGan", save_dir, ["D_A losses", "D_B losses", "G losses"])
    log.start(epochs=epochs, batch_size=batch_size,  learning_rate=learning_rate, n_train=total, device=device) 
    
    print("Starting Training Loop...")

    # Actual training loop
    resize = transforms.Resize((64, 64))
    try:
        for epoch in range(epochs- start_epoch):
            iters = 0
            target_iter = iter(target_train_loader)

            with tqdm(total=total, desc=f'Epoch {epoch + 1 + start_epoch}/{epochs}', unit='img') as pbar:
                for data_source in source_train_loader:
                    try:
                        data_target = next(target_iter)
                    except StopIteration:
                        target_iter = iter(target_train_loader)
                        data_target = next(target_iter)

                        
                    # Set model input
                    a_real = resize.forward(data_source[0]).to(device)
                    b_real = resize.forward(data_target).to(device)

                    # Generate images
                    b_fake = G_A2B(a_real)
                    a_rec = G_B2A(b_fake)
                    a_fake = G_B2A(b_real)
                    b_rec = G_A2B(a_fake)
                    
                    
                    # Discriminator A
                    optimizer_D_A.zero_grad()
                    if (iters > 0 or epoch > 0) and iters % 3 == 0:
                        # Calculate discriminator loss with current A + randomly chosen saved B-->A domain images
                        # Dimensions must match
                        rand_int = random.randint(batch_size, old_a_fake.shape[0]-1)
                        Disc_loss_A = LSGAN_D(D_A(a_real), D_A(old_a_fake[rand_int-batch_size:rand_int].detach()))
                        D_A_losses.append(Disc_loss_A.item())
                    else:
                        Disc_loss_A = LSGAN_D(D_A(a_real), D_A(a_fake.detach()))
                        D_A_losses.append(Disc_loss_A.item())

                    Disc_loss_A.backward()
                    optimizer_D_A.step()

                    # Discriminator B
                    optimizer_D_B.zero_grad()
                    if (iters > 0 or epoch > 0) and iters % 3 == 0:
                        # Calculate discriminator loss with current B + randomly chosen saved A-->B domain images
                        # Dimensions must match
                        rand_int = random.randint(batch_size, old_b_fake.shape[0]-1)
                        Disc_loss_B =  LSGAN_D(D_B(b_real), D_B(old_b_fake[rand_int-batch_size:rand_int].detach()))
                        D_B_losses.append(Disc_loss_B.item())
                    else:
                        Disc_loss_B =  LSGAN_D(D_B(b_real), D_B(b_fake.detach()))
                        D_B_losses.append(Disc_loss_B.item())

                    Disc_loss_B.backward()
                    optimizer_D_B.step()

                    # Generator
                    optimizer_G_A2B.zero_grad()
                    optimizer_G_B2A.zero_grad()

                    # Fool discriminator
                    Fool_disc_loss_A2B = LSGAN_G(D_B(b_fake))
                    Fool_disc_loss_B2A = LSGAN_G(D_A(a_fake))

                    # Cycle Consistency, both use the two generators
                    # Criterion is L1 loss so we multiply by batch_size
                    Cycle_loss_A = criterion(a_rec, a_real) * batch_size
                    Cycle_loss_B = criterion(b_rec, b_real) * batch_size

                    # Identity loss
                    # For some reason multiplied by 2 * batch_size
                    Id_loss_B2A = criterion(G_B2A(a_real), a_real) * batch_size * 2
                    Id_loss_A2B = criterion(G_A2B(b_real), b_real) * batch_size * 2

                    # Generator losses
                    Loss_G = Fool_disc_loss_A2B + Fool_disc_loss_B2A + Cycle_loss_A + Cycle_loss_B + Id_loss_B2A + Id_loss_A2B
                    G_losses.append(Loss_G)

                    # Backward propagation
                    Loss_G.backward()

                    # Optimisation step
                    optimizer_G_A2B.step()
                    optimizer_G_B2A.step()

                    FDL_A2B.append(Fool_disc_loss_A2B)
                    FDL_B2A.append(Fool_disc_loss_B2A)
                    CL_A.append(Cycle_loss_A)
                    CL_B.append(Cycle_loss_B)
                    ID_B2A.append(Id_loss_B2A)
                    ID_A2B.append(Id_loss_A2B)
                    disc_A.append(Disc_loss_A)
                    disc_B.append(Disc_loss_B)

                    if iters == 0 and epoch == 0:
                        old_b_fake = b_fake.clone()
                        old_a_fake = a_fake.clone()
                    elif old_b_fake.shape[0] == 5*batch_size and b_fake.shape[0] == batch_size:
                        rand_int = random.randint(batch_size, 5*batch_size-1)
                        old_b_fake[rand_int-batch_size:rand_int] = b_fake.clone()
                        old_a_fake[rand_int-batch_size:rand_int] = a_fake.clone()
                    elif old_b_fake.shape[0] < 5*batch_size:
                        old_b_fake = torch.cat((b_fake.clone(), old_b_fake))
                        old_a_fake = torch.cat((a_fake.clone(), old_a_fake))

                    # update tqdm and tensorboard and text logger  
                    global_step += 1
                    iters += 1
                    tb.add_scalar("Disc_loss_A", D_A_losses[-1], global_step)
                    tb.add_scalar("Disc_loss_B", D_B_losses[-1], global_step)
                    tb.add_scalar("Loss_G", G_losses[-1], global_step)
                    log.update_loss(D_A_losses[-1],"D_A losses",global_step)
                    log.update_loss(D_B_losses[-1],"D_B losses",global_step)
                    log.update_loss(G_losses[-1],"G losses",global_step)
                    pbar.update(1)
                #pbar.set_postfix_str(f'\nFDL_A2B: {Fool_disc_loss_A2B:.3f}\tFDL_B2A: {Fool_disc_loss_B2A:.3f}\tCL_A: {Cycle_loss_A:.3f}\tCL_B: {Cycle_loss_B:.3f}\tID_B2A: {Id_loss_B2A:.3f}\tID_A2B: {Id_loss_A2B:.3f}\tLoss_D_A: {Disc_loss_A.item():.3f}\tLoss_D_B: {Disc_loss_B.item():.3f}')
            
            losses_log = {"Disc_loss_A": D_A_losses[-1],
                        "Disc_loss_B": D_B_losses[-1],
                        "Loss_G": G_losses[-1]}
            log.update(losses_log, epoch=epoch+start_epoch)
            saver.update(epoch=epoch +start_epoch,
                        model=(G_A2B, G_B2A, D_A, D_B),
                        optimizer= (optimizer_G_A2B, optimizer_G_B2A, optimizer_D_A, optimizer_D_B),
                        step=global_step,
                        losses= [G_losses, D_A_losses, D_B_losses])

            '''
            FDL_A2B_t.append(sum(FDL_A2B)/len(FDL_A2B))
            FDL_B2A_t.append(sum(FDL_B2A)/len(FDL_B2A))
            CL_A_t.append(sum(CL_A)/len(CL_A))
            CL_B_t.append(sum(CL_B)/len(CL_B))
            ID_B2A_t.append(sum(ID_B2A)/len(ID_B2A))
            ID_A2B_t.append(sum(ID_A2B)/len(ID_A2B))
            disc_A_t.append(sum(disc_A)/len(disc_A))
            disc_B_t.append(sum(disc_B)/len(disc_B))
            '''

            FDL_A2B = []
            FDL_B2A = []
            CL_A = []
            CL_B = []
            ID_B2A = []
            ID_A2B = []
            disc_B = []
            disc_A = []

            # Save models
            torch.save(G_A2B, str(os.path.join(save_dir, model_name + "_G_A2B.pt")))
            torch.save(G_B2A, str(os.path.join(save_dir, model_name +"_G_B2A.pt")))
            torch.save(D_A, str(os.path.join(save_dir, model_name +"_D_A.pt")))
            torch.save(D_B, str(os.path.join(save_dir, model_name +"_D_B.pt")))
            
            saver.save_training()
            
        tb.close()
        log.finish()
    except KeyboardInterrupt:
        if save_checkpoint:
            print("Training interrupted")
            saver.save_training()
            print(f'Chekpoint saved in {os.path.join(save_dir, "PAUSE", "paused_training.pth")}')
        else:
            print(f'Training interrupted. Checkpoint not saved.')
        tb.close()
        log.finish()
        sys.exit()


if __name__ == '__main__':
    DATA_DIR = os.path.join(os.getcwd(), "data")
    TARGET_DATA_DIR = os.path.join(DATA_DIR, "target")
    LIVECELL_IMG_DIR = os.path.join(DATA_DIR, "livecell", "images")
    LIVECELL_MASK_DIR = os.path.join(DATA_DIR, "livecell", "masks")
    UNITY_IMG_DIR = os.path.join(DATA_DIR, "unity_data", "images")
    UNITY_MASK_DIR = os.path.join(DATA_DIR, "unity_data", "masks")
    dir_checkpoint = os.path.join(os.getcwd(), "model" )

    # Create data loaders
    LC_dataset = MaskedDataset(LIVECELL_IMG_DIR, LIVECELL_MASK_DIR, length=None, in_memory=False, IMG_SIZE=256, mode=1)
    #Unity_dataset = MaskedDataset(UNITY_IMG_DIR, UNITY_MASK_DIR, length=None, in_memory=False)
    #datasets = [LC_dataset, Unity_dataset]
    #dataset = torch.utils.data.ConcatDataset(datasets)
    LC_empty_dataset = EmptyLiveCELLDataset(3 * len(LC_dataset))
    LC_datasets = [LC_dataset, LC_empty_dataset]  # 75% empty, 25% actual LiveCELL images
    dataset = torch.utils.data.ConcatDataset(LC_datasets)
    train_set = dataset
    
    seed = 123
    test_percent = 0.001
    n_test = int(len(dataset) * test_percent)
    n_train = len(dataset) - n_test

    train_set, test_set = torch.utils.data.random_split(dataset, [n_train, n_test], generator=torch.Generator().manual_seed(seed))

    batch_size = 16

     
    source_train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True, drop_last=True) # num_workers is number of cores used, pin_memory enables fast data transfer to CUDA-enabled GPUs
    # source_val_loader = DataLoader(test_set, shuffle=True, drop_last=True, **loader_args)

    Target_dataset = UnMaskedDataset(TARGET_DATA_DIR, mode=1, IMG_SIZE=256)

    target_test_percent = 0.01
    n_test_target = int(len(Target_dataset) * target_test_percent)
    n_train_target = len(Target_dataset) - n_test_target
    target_train_set, target_test_set = torch.utils.data.random_split(Target_dataset, [n_train_target, n_test_target],
                                                                      generator=torch.Generator().manual_seed(seed))

    target_train_loader = DataLoader(target_train_set, shuffle=True, batch_size=batch_size, num_workers=4,
                                     pin_memory=True, drop_last=True)

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
               model_name="epty_images",
               epochs=10,
               batch_size=batch_size,
               save_checkpoint=True,
               Resume=False,
               Pause_path ="")
