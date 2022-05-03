import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""
The implementation is based on article "Unpaired Image-to-Image Translation
using Cycle-Consistent Adversarial Networks": https://arxiv.org/pdf/1703.10593.pdf
and the model is originally implemented in https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
"""
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs=3, ngf=64, norm_layer = nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()
        unet_block = UnetSkipConnectionBLock(outer_nc= ngf*8,
                                             inner_nc=ngf*8,
                                             input_nc= None,
                                             submodule= None,
                                             norm_layer=norm_layer,
                                             innermost = True)   
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBLock(outer_nc= ngf*8,
                                                inner_nc=ngf*8,
                                                input_nc= None,
                                                submodule= unet_block,
                                                norm_layer=norm_layer,
                                                use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBLock(outer_nc= ngf*4,
                                             inner_nc=ngf*8,
                                             input_nc= None,
                                             submodule= unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBLock(outer_nc= ngf*2,
                                             inner_nc=ngf*4,
                                             input_nc= None,
                                             submodule= unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBLock(outer_nc= ngf,
                                             inner_nc=ngf*2,
                                             input_nc= None,
                                             submodule= unet_block,
                                             norm_layer=norm_layer)
        self.model = UnetSkipConnectionBLock(outer_nc= output_nc,
                                             inner_nc=ngf,
                                             input_nc= input_nc,
                                             submodule= unet_block,
                                             norm_layer=norm_layer,
                                             outermost = True)

    def forward(self, input):
        return self.model(input)

class UnetSkipConnectionBLock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc= None, submodule=None, outermost= False, innermost = False, norm_layer=nn.BatchNorm2d, use_dropout= False):
        super(UnetSkipConnectionBLock, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.BatchNorm2d
        if input_nc is None:
            input_nc=outer_nc
        downconv = nn.Conv2d(in_channels=input_nc,
                            out_channels=inner_nc,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(in_channels=inner_nc*2,
                                        out_channels=outer_nc,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(in_channels=inner_nc,
                                        out_channels=outer_nc,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1,
                                        bias = use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(in_channels=inner_nc*2,
                                        out_channels=outer_nc,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1,
                                        bias = use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:

                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(NLayerDiscriminator, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(in_channels=input_nc,
                              out_channels= ndf,
                              kernel_size=kw,
                              stride=2,
                              padding=padw),
                    nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [nn.Conv2d(in_channels=ndf*nf_mult_prev,
                                    out_channels=ndf*nf_mult,
                                    kernel_size=kw,
                                    stride=2,
                                    padding=padw,
                                    bias=use_bias),
                        norm_layer(ndf*nf_mult),
                        nn.LeakyReLU(0.2,True)]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(in_channels= ndf * nf_mult_prev,
                                out_channels= ndf * nf_mult, 
                                kernel_size=kw, 
                                stride=1, 
                                padding=padw, 
                                bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)]

        sequence += [nn.Conv2d(in_channels = ndf * nf_mult,
                                out_channels = 1,
                                kernel_size=kw, 
                                stride=1, 
                                padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

class CycleGan(nn.Module):
    def __init__(self, input_nc, output_nc,lr, beta1, device, ngf=64, ndf=64):
        super(CycleGan, self).__init__()
        self.device = device

        '''
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        
        visual_names_A = ['real_A', 'fake_B', 'rec_A', 'idt_B']
        visual_names_B = ['real_B', 'fake_A', 'rec_B', 'idt_A']

        self.visual = visual_names_A + visual_names_B
        self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        '''
        self.netG_A = UnetGenerator(input_nc=input_nc, output_nc=output_nc, ngf=ngf).to(device)
        self.netG_B = UnetGenerator(input_nc=input_nc, output_nc=output_nc, ngf= ngf).to(device)

        self.netD_A = NLayerDiscriminator(input_nc=output_nc, ndf=ndf).to(device)
        self.netD_B = NLayerDiscriminator(input_nc=input_nc, ndf=ndf).to(device)

        self.criterionGAN = nn.BCEWithLogitsLoss()
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()

        self.optimizer_G = torch.optim.Adam(list(self.netG_A.parameters()) + list(self.netG_B.parameters()), lr=lr, betas=(beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(list(self.netD_A.parameters()) + list(self.netD_B.parameters()), lr=lr, betas=(beta1, 0.999))
        
        self.optimizers = [self.optimizer_D, self.optimizer_G]

    def forward(self, real_A, real_B):
        self.fake_B = self.netG_A(real_A.to(self.device))  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(real_B.to(self.device))  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):

        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, real)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, fake)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self, real_B):
        self.loss_D_A = self.backward_D_basic(self.netD_A, real_B, self.fake_B)

    def backward_D_B(self, real_A):
        self.loss_D_B = self.backward_D_basic(self.netD_B, real_A, self.fake_A)

    def backward_G(self, real_A, real_B):
        # G_A should be identity if real_B is fed: ||G_A(B) - B||
        self.idt_A = self.netG_A(real_B)
        self.loss_idt_A = self.criterionIdt(self.idt_A, real_B) * 10 * 0.5
        # G_B should be identity if real_A is fed: ||G_B(A) - A||
        self.idt_B = self.netG_B(real_A)
        self.loss_idt_B = self.criterionIdt(self.idt_B, real_A) * 10 * 0.5

         # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), real_B)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), real_A)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, real_A) * 10
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, real_B) * 10
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()
    
    def optimize_parameters(self, real_A, real_B):
        # forward
        self.forward(real_A, real_B)      # compute fake images and reconstruction images.
        # G_A and G_B
        self.netD_A.requires_grad_(False) 
        self.netD_B.requires_grad_(False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G(real_A, real_B)             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.netD_A.requires_grad_(True) 
        self.netD_B.requires_grad_(True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A(real_B)      # calculate gradients for D_A
        self.backward_D_B(real_A)      # calculate graidents for D_B
        self.optimizer_D.step() # update D_A and D_B's weights

########################################

import argparse
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import MaskedDataset, UnMaskedDataset

if __name__ == '__main__':
    DATA_DIR = os.path.join(os.getcwd(), "data")
    TARGET_DATA_DIR = os.path.join(DATA_DIR, "target")
    LIVECELL_IMG_DIR = os.path.join(DATA_DIR, "livecell", "images")
    LIVECELL_MASK_DIR = os.path.join(DATA_DIR, "livecell", "masks")
    dir_checkpoint = os.path.join(os.getcwd(), "model" )

    # parser for commandline arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, required=True, help='The number of epochs run in training')
    parser.add_argument('--lr', type=float,  help='The learning rate utilised in training. If not defined learning rate is set to 0.0001')
    parser.add_argument('--batch', type=int, help="The batch size utilised in training. If not defined, batch size is set to 16")
    args = parser.parse_args()

    if args.batch ==None:
        batch_size = 16
    else:
        batch_size = args.batch

    if args.lr ==None:
        lr = 0.0001
    else:
        lr = args.lr

    # Create data loaders
    LC_dataset = MaskedDataset(LIVECELL_IMG_DIR, LIVECELL_MASK_DIR, length=None, in_memory=False, IMG_SIZE=256, mode=1)
    dataset = LC_dataset
    train_set = dataset

    seed = 123
    test_percent = 0.001
    n_test = int(len(dataset) * test_percent)
    n_train = len(dataset) - n_test
    train_set, test_set = torch.utils.data.random_split(dataset, [n_train, n_test], generator=torch.Generator().manual_seed(seed))

    source_train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True, drop_last=True)

    Target_dataset = UnMaskedDataset(TARGET_DATA_DIR, mode=1, IMG_SIZE=256)

    target_test_percent = 0.01
    n_test_target = int(len(Target_dataset) * target_test_percent)
    n_train_target = len(Target_dataset) - n_test_target
    target_train_set, target_test_set = torch.utils.data.random_split(Target_dataset, [n_train_target, n_test_target],
                                                                      generator=torch.Generator().manual_seed(seed))
    target_train_loader = DataLoader(target_train_set, shuffle=True, batch_size=batch_size, num_workers=4,
                                     pin_memory=True, drop_last=True)
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CycleGan(input_nc=1,
                     output_nc=1,
                     lr=lr,
                     beta1=0.5,
                     device=device)
########################################
    print("Starting training loop...")

    for epoch in range(args.epochs):
        with tqdm(source_train_loader,desc=f'Epoch {epoch + 1 }/{args.epochs}', unit='img') as pbar:
            target_iter = iter(target_train_loader)

            for data_source in pbar:   
                try:
                    data_target = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_train_loader)
                    data_target = next(target_iter)
                
                a_real = data_source[0].to(device)
                b_real = data_target.to(device)

                model.optimize_parameters(a_real, b_real)