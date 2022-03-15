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
    def __init__(self, features=64, blocks=9):
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
               betas=(0.5, 0.999)):

    # Model saving location
    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
    model_number = 1
    save_dir = os.path.join(dir_checkpoint, f'CycleGan_{datetime.now().date()}')
    while os.path.exists(save_dir) == True:
        model_number += 1
        save_dir = os.path.join(dir_checkpoint, f'CycleGan_{model_number}_{datetime.now().date()}')
    Path(save_dir).mkdir(parents=True, exist_ok=True)

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

    print("Starting Training Loop...")

    # Actual training loop
    for epoch in range(epochs):
        iters = 0
        for i, (data_source, data_target) in enumerate(zip(tqdm(source_train_loader), target_train_loader), 0):
            
            # Set model input
            a_real = data_source[0].to(device)
            b_real = data_target.to(device)

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
                rand_int = random.randint(batch_size, batch_size*batch_size-1)
                old_b_fake[rand_int-batch_size:rand_int] = b_fake.clone()
                old_a_fake[rand_int-batch_size:rand_int] = a_fake.clone()
            elif old_b_fake.shape[0] < 5*batch_size:
                old_b_fake = torch.cat((b_fake.clone(), old_b_fake))
                old_a_fake = torch.cat((a_fake.clone(), old_a_fake))

            iters += 1
        
        print('[%d/%d]\tFDL_A2B: %.4f\tFDL_B2A: %.4f\tCL_A: %.4f\tCL_B: %.4f\tID_B2A: %.4f\tID_A2B: %.4f\tLoss_D_A: %.4f\tLoss_D_A: %.4f'
                      % (epoch+1, epochs, Fool_disc_loss_A2B, Fool_disc_loss_B2A,Cycle_loss_A,Cycle_loss_B,Id_loss_B2A,
                          Id_loss_A2B, Disc_loss_A.item(), Disc_loss_B.item()))
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