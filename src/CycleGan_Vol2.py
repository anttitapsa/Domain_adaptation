import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_loader import MaskedDataset, UnMaskedDataset
# https://github.com/Hitha83/MRI-styletransfer-CycleGAN/blob/main/mri_gan.ipynb
class unet_generator(nn.Module):
    def __init__(self, in_channels, out_channels, device):
        super(unet_generator,self).__init__()
        self.device = device
        self.encoder = [nn.Sequential(nn.Conv2d(in_channels = in_channels, out_channels = 64, kernel_size = 3, padding = 1),
                                     #nn.InstanceNorm2d(num_features = out_channels),
                                     nn.LeakyReLU(inplace = True)),
                        nn.Sequential(nn.Conv2d(in_channels =64, out_channels = 128, kernel_size = 3, padding = 1),
                                     nn.InstanceNorm2d(num_features = 128),
                                     nn.LeakyReLU(inplace = True)),
                        nn.Sequential(nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1),
                                     nn.InstanceNorm2d(num_features = 256),
                                     nn.LeakyReLU(inplace = True)),
                        nn.Sequential(nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, padding = 1),
                                     nn.InstanceNorm2d(num_features = 512),
                                     nn.LeakyReLU(inplace = True))]
        
        self.decoder = [nn.Sequential(nn.ConvTranspose2d(in_channels= 512, out_channels= 256,  kernel_size= 3, padding=1),
                                     nn.InstanceNorm2d(num_features=256),
                                     nn.ReLU()),
                        nn.Sequential(nn.ConvTranspose2d(in_channels= 512, out_channels= 128, kernel_size= 3, padding=1),
                                     nn.InstanceNorm2d(num_features=128),
                                     nn.ReLU()),
                        nn.Sequential(nn.ConvTranspose2d(in_channels= 256, out_channels=64, kernel_size= 3, padding=1),
                                     nn.InstanceNorm2d(num_features=64),
                                     nn.ReLU()),
                        nn.Sequential(nn.ConvTranspose2d(in_channels= 128, out_channels=32, kernel_size= 3, padding=1),
                                     nn.InstanceNorm2d(num_features=32),
                                     nn.ReLU())]

        self.out_layer = nn.Sequential(nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=3, padding=1),
                                        nn.Tanh())
        
    def forward(self, x):
        skips = []
        for down in self.encoder:
            down.to(self.device)
            x = down.forward(x)
            skips.append(x)
              
        skips = reversed(skips[:-1])
        
        for up, skip in zip(self.decoder, skips):
            up.to(self.device)
            x = up.forward(x)
            x = torch.cat((x, skip), dim=1)
        x = self.out_layer.forward(x)
        return x
        


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.downsample = nn.Sequential(nn.Conv2d(in_channels =1, out_channels = 64, kernel_size = 4, padding = 1),
                                     nn.InstanceNorm2d(num_features = 64),
                                     nn.LeakyReLU(inplace = True),
                                        nn.Conv2d(in_channels =64, out_channels = 128, kernel_size = 4, padding = 1),
                                     nn.InstanceNorm2d(num_features = 128),
                                     nn.LeakyReLU(inplace = True),
                                    nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4, padding = 1),
                                     nn.InstanceNorm2d(num_features = 256),
                                     nn.LeakyReLU(inplace = True),
                                     nn.ZeroPad2d((1,1)),#mahdollinen ongelma
                                     nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 4, padding = 1),
                                     nn.InstanceNorm2d(num_features=512),
                                     nn.LeakyReLU(),
                                    nn.ZeroPad2d((1,1)),#mahdollinen ongelma
                                    nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1))
    
    def forward(self, x):
        return self.downsample.forward(x)


def discriminator_loss(real, generated):
    return (torch.mean((real - 1)**2) + torch.mean(generated**2))

def generator_loss(generated):
    return torch.mean((generated - 1)**2)

def calc_cycle_loss(real_image, cycled_image):
    return 10 * torch.mean(torch.abs(real_image-cycled_image))

def identity_loss(real_image, sample_image):
    return 10 * 0.5 * torch.mean(torch.abs(real_image - sample_image))

def train_loop(dataloaders,
               epochs=5,
               device="cpu",
               learning_rate = 0.0002,
               betas=(0.5, 0.999),
               log_dir=""):

    # tensorboard
    tb = SummaryWriter(log_dir=log_dir)

    generator_g = unet_generator(in_channels=1, out_channels=1, device=device)
    generator_f = unet_generator(in_channels=1, out_channels=1, device=device)
    discriminator_x = discriminator()
    discriminator_y = discriminator()

    generator_g_optimizer = torch.optim.Adam(generator_g.parameters(), lr=learning_rate, betas=betas)
    generator_f_optimizer = torch.optim.Adam(generator_f.parameters(), lr=learning_rate, betas=betas)
    discriminator_x_optimizer = torch.optim.Adam(discriminator_x.parameters(), lr=learning_rate, betas=betas)
    discriminator_y_optimizer = torch.optim.Adam(discriminator_y.parameters(), lr=learning_rate, betas=betas)

    # extract dataloaders
    train_loaders = dataloaders[0]
    test_loaders = dataloaders[1]


    for epoch in range(epochs):
        Losses = []
        with tqdm(total=len(train_loaders[1]), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for i,  (source_img, target_img) in enumerate(zip(train_loaders[0], train_loaders[1]), 0):

                source_img = source_img[0].to(device)
                target_img = target_img.to(device)
                generator_g.to(device)
                generator_f.to(device)
                discriminator_x.to(device)
                discriminator_y.to(device)


                generator_g_optimizer.zero_grad()
                generator_f_optimizer.zero_grad()
                discriminator_x_optimizer.zero_grad()
                discriminator_y_optimizer.zero_grad()
                
                '''
                # Generator G translates X -> Y
                # Generator F translates Y -> X.
                fake_y = generator_g(source_img)
                cycled_x = generator_f(fake_y)

                fake_x = generator_g(target_img)
                cycled_y = generator_f(fake_x)

                # same_x and same_y are used for identity loss.
                same_x = generator_f(source_img)
                same_y = generator_g(target_img)

                disc_source =  discriminator_x(source_img) 
                disc_target = discriminator_y(target_img) 
                
                disc_fake_source = discriminator_x(fake_x) 
                disc_fake_target = discriminator_y(fake_y) 
                '''
                #################################
                # calculate the loss
                gen_g_loss = generator_loss(discriminator_x(generator_g(target_img)))
                total_cycle_loss = calc_cycle_loss(source_img, generator_f(generator_g(source_img))) + calc_cycle_loss(target_img, generator_f(generator_g(target_img)))
                total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(target_img, generator_g(target_img))
                
                total_gen_g_loss.backward()
                generator_g_optimizer.step()
                #################################
                gen_f_loss = generator_loss(discriminator_y(generator_g(source_img)))
                total_cycle_loss = calc_cycle_loss(source_img, generator_f(generator_g(source_img))) + calc_cycle_loss(target_img, generator_f(generator_g(target_img)))
                total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(source_img, generator_f(source_img))
                
                total_gen_f_loss.backward()
                generator_f_optimizer.step()
                #################################

                disc_x_loss = discriminator_loss(discriminator_x(source_img), discriminator_x(generator_g(target_img)) ) # calculate the discriminator_loss for disc_fake_x wrt disc_real_x
                disc_x_loss.backward()
                discriminator_x_optimizer.step()
                #################################
                disc_y_loss = discriminator_loss(discriminator_y(target_img), discriminator_y(generator_g(source_img)))
                disc_y_loss.backward()
                discriminator_y_optimizer.step()
                pbar.update(1)
            
            tb.add_scalar("Total generator G training loss", total_gen_g_loss, epoch +1)
            tb.add_scalar("Total generator F training loss", total_gen_f_loss, epoch +1)
            tb.add_scalar("Total cycle training loss", total_cycle_loss, epoch +1)
    
    tb.close()
    return generator_g, generator_f, discriminator_x, discriminator_y
                
if __name__ == '__main__':
    DATA_DIR = os.path.join(os.getcwd(), "data")
    TARGET_DATA_DIR = os.path.join(DATA_DIR, "target")
    LIVECELL_IMG_DIR = os.path.join(DATA_DIR, "livecell", "images")
    LIVECELL_MASK_DIR = os.path.join(DATA_DIR, "livecell", "masks")
    UNITY_IMG_DIR = os.path.join(DATA_DIR, "unity_data", "images")
    UNITY_MASK_DIR = os.path.join(DATA_DIR, "unity_data", "masks")
    dir_checkpoint = os.path.join(os.getcwd(), "model" )

    # data loaders
    batch_size = 16
    seed = 123
    LC_test_percent = 0.001
    target_test_percent = 0.01

    LC_dataset = MaskedDataset(LIVECELL_IMG_DIR, LIVECELL_MASK_DIR, length=None, in_memory=False, IMG_SIZE=64, mode=2)
    target_dataset = UnMaskedDataset(TARGET_DATA_DIR, mode=2, IMG_SIZE=64)

    n_LC_test = int(len(LC_dataset) * LC_test_percent)
    n_LC_train = len(LC_dataset) - n_LC_test
    LC_trainset, LC_testset = torch.utils.data.random_split(LC_dataset, [n_LC_train, n_LC_test], generator=torch.Generator().manual_seed(seed))
    LC_train_loader = DataLoader(LC_trainset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True, drop_last=True)
    LC_test_loader = DataLoader(LC_testset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True, drop_last=True)

    n_target_test = int(len(target_dataset) * target_test_percent)
    n_target_train = len(target_dataset) - n_target_test
    target_trainset, target_testset = torch.utils.data.random_split(target_dataset, [n_target_train, n_target_test], generator=torch.Generator().manual_seed(seed))
    target_train_loader = DataLoader(target_trainset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True, drop_last=True)
    target_test_loader = DataLoader(target_testset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True, drop_last=True)

    dataloaders = [(LC_train_loader, target_train_loader), (LC_test_loader, target_test_loader)]
    # device GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator_g, generator_f, discriminator_x, discriminator_y = train_loop(dataloaders, log_dir = os.path.join(dir_checkpoint, "runs", "Unet_cyclegan"), device=device)

    torch.save(generator_g, os.path.join(dir_checkpoint, "test_unetcyclegan", "generator_g.pth"))
    torch.save(generator_f, os.path.join(dir_checkpoint, "test_unetcyclegan", "generator_f.pth"))
    torch.save(discriminator_x, os.path.join(dir_checkpoint, "test_unetcyclegan", "discriminator_x.pth"))
    torch.save(discriminator_y, os.path.join(dir_checkpoint, "test_unetcyclegan", "discriminator_y.pth"))
    


