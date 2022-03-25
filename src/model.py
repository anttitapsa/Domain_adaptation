import torch
from torch.nn import Conv2d
from torch.nn import Module
from torch.nn import MaxPool2d
from torch.nn import BatchNorm2d
from torch.nn import ReLU
import torch.nn as nn
from torch.nn import Dropout
from torch import cat
from torch.nn import ConvTranspose2d
import torch.nn.functional as F
# code based on:
# https://github.com/hlamba28/UNET-TGS 
# https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py 
class Unet(Module):
    # input_image 512x512
    # numChannels 1: grayscale and 3 RGB
    # classes: number of labels
    # dropout: During training, randomly zeroes some of the elements of the input tensor with probability 
    # dropout to prevent overtraining
    def __init__(self, numChannels = 1, classes = 2, dropout = 0.1, image_res=512):
        super(Unet, self).__init__()
        
        self.domain_classifier = domain_classifier(in_channel=int((image_res//16)**2*1024))
        
        # Encoder (traditional convolutional and max pooling layers)
        self.conv1 = Unet._conv2d_block(in_channel = numChannels, out_channel = 64)
        self.maxpool1 = MaxPool2d(kernel_size = 2, stride = 2)
        self.dropout1 = Dropout(dropout)

        self.conv2 = Unet._conv2d_block(in_channel = 64, out_channel = 128)
        self.maxpool2 = MaxPool2d(kernel_size = 2, stride = 2)
        self.dropout2 = Dropout(dropout)
       
        self.conv3 = Unet._conv2d_block(in_channel = 128, out_channel = 256)
        self.maxpool3 = MaxPool2d(kernel_size = 2, stride = 2)
        self.dropout3 = Dropout(dropout)
        
        self.conv4 = Unet._conv2d_block(in_channel= 256, out_channel = 512)
        self.maxpool4 = MaxPool2d(kernel_size = 2, stride = 2)
        self.dropout4 = Dropout(dropout)
        
        self.conv5 = Unet._conv2d_block(in_channel = 512, out_channel = 1024)
        
        # Decoder (converts a reduced image to retain pixel location infromation) 
        
        self.transpose6 = ConvTranspose2d(in_channels = 1024, out_channels = 512, kernel_size = 2, stride = 2)
        self.dropout6 = Dropout(dropout)
        self.conv6 = Unet._conv2d_block(in_channel = 1024, out_channel = 512)

        self.transpose7 = ConvTranspose2d(in_channels = 512, out_channels = 256, kernel_size = 2, stride = 2)
        self.dropout7 = Dropout(dropout)
        self.conv7 = Unet._conv2d_block(in_channel = 512, out_channel = 256)
        
        self.transpose8 = ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 2, stride = 2)
        self.dropout8 = Dropout(dropout)
        self.conv8 = Unet._conv2d_block(in_channel = 256, out_channel = 128)

        self.transpose9 = ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 2, stride = 2)
        self.dropout9 = Dropout(dropout)
        self.conv9 = Unet._conv2d_block(in_channel = 128, out_channel = 64)
        
        self.outc = Conv2d(in_channels = 64, out_channels = classes, kernel_size = 1)
        
        
    def forward(self, x):
        # function that defines how the model is going to be run, from input to output
        x1 = self.conv1(x)
        x = self.maxpool1(x1)
        x = self.dropout1(x)
        
        x2 = self.conv2(x)
        x = self.maxpool2(x2)
        x = self.dropout2(x)
        
        x3 = self.conv3(x)
        x = self.maxpool3(x3)
        x = self.dropout3(x)
        
        x4 = self.conv4(x)
        x = self.maxpool4(x4)
        x = self.dropout4(x)
        
        x5 = self.conv5(x)
        x_dom = torch.flatten(x5, start_dim=1) 
        y = self.domain_classifier(x_dom)
        
        x6 = self.transpose6(x5)
        x = cat((x6, x4), dim = 1)
        x = self.dropout6(x)
        x = self.conv6(x)
        
        x7 = self.transpose7(x)
        x = cat((x7, x3), dim = 1)
        x = self.dropout7(x)
        x = self.conv7(x)
        
        x8 = self.transpose8(x)
        x = cat((x8, x2), dim = 1)
        x = self.dropout8(x)
        x = self.conv8(x)
        
        x9 = self.transpose9(x)
        x = cat((x9, x1), dim = 1)
        x = self.dropout9(x)
        x = self.conv9(x)
        
        x = self.outc(x)
        return (torch.sigmoid(x), y)

    @staticmethod
    def _conv2d_block(in_channel, out_channel):
        return torch.nn.Sequential(
            Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size = 3, padding = 1),
            BatchNorm2d(num_features = out_channel),
            ReLU(inplace = True),
            Conv2d(in_channels = out_channel, out_channels = out_channel, kernel_size = 3, padding = 1),
            BatchNorm2d(num_features = out_channel),
            ReLU(inplace = True)
        )
   
class domain_classifier(nn.Module):
    def __init__(self, in_channel):
        super(domain_classifier, self).__init__()
        self.fc1 = nn.Linear(in_channel, 100) 
        self.fc2 = nn.Linear(100, 1)
        self.drop = nn.Dropout2d(0.25)

    def forward(self, x):
        x = grad_reverse(x)
        x = F.leaky_relu(self.drop(self.fc1(x)))
        x = self.fc2(x)
        return torch.sigmoid(x)
      
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)

