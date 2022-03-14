from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

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
                                    nn.Conv2d(  in_channels=3,
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
                                                out_channels=3, 
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
        self.disc = nn.Sequential(  nn.Conv2d(  in_channels=3,
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
                                    nn.InstanceNorm2d(64*2),
                                    nn.LeakyReLU(negative_slope= 0.2, inplace=True),
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
        return self.disc.forward(x)


def LSGAN_D(source, target):
    return (torch.mean((source - 1)**2) + torch.mean(target**2))

def LSGAN_G(target):
    return  torch.mean((target - 1)**2)


if __name__ == '__main__':

    G_A
                                                        

        

