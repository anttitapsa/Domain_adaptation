from torch.nn import Conv2d
from torch.nn import Module
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch import Dropout
from torch import cat
from torch import ConvTranspose2d
import torch.nn.functional as F

class Unet(Module):
    # numChannels 1: grayscale and 3 RGB
    # classes: number of labels
    # dropout: During training, randomly zeroes some of the elements of the input tensor with probability 
    # dropout to prevent overtraining
    def __init__(self, numChannels = 1, classes = 2, dropout = 0.1):
        super(Unet, self).__init__()
        # Encoder (traditional convolutional and max pooling layers)
        self.conv1 = Conv2d_block(in_channel = numChannels, out_channel = 64)
        self.maxpool1 = MaxPool2d(2)
        self.dropout1 = Dropout(dropout)

        self.conv2 = Conv2d_block(in_channel = 64, out_channel = 128)
        self.maxpool2 = MaxPool2d(2)
        self.dropout2 = Dropout(dropout)
       
        self.conv3 = Conv2d_block(in_channel = 128, out_channel = 256)
        self.maxpool3 = MaxPool2d(2)
        self.dropout3 = Dropout(dropout)
        
        self.conv4 = Conv2d_block(in_channels= 256, out_channel = 512)
        self.maxpool4 = MaxPool2d(2)
        self.dropout4 = Dropout(dropout)
        
        self.conv5 = Conv2d_block(in_channel = 512, out_channel = 512)
        
        # Decoder (converts a reduced image to retain pixel location infromation) 
        
        self.transpose6 = ConvTranspose2d(in_channels = 1024//2, out_channels = 256//2, kernel_size = 3)
        self.dropout6 = Dropout(dropout)
        self.conv6 = Conv2d_block(in_channels = 1024, out_channel = 256)

        self.transpose7 = ConvTranspose2d(in_channels = 512//2, out_channels = 128//2, kernel_size = 3)
        self.dropout7 = Dropout(dropout)
        self.conv7 = Conv2d_block(in_channels = 512, out_channel = 128)
        
        self.transpose8 = ConvTranspose2d(in_channels = 256//2, out_channels = 64//2, kernel_size = 3)
        self.dropout8 = Dropout(dropout)
        self.conv8 = Conv2d_block(in_channels = 256, out_channel = 64)

        self.transpose9 = ConvTranspose2d(in_channels = 128//2, out_channels = 64//2, kernel_size = 3)
        self.dropout9 = Dropout(dropout)
        self.conv9 = Conv2d_block(in_channels = 128, out_channel = 64)
        
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
        
        x = self.conv5(x)
        
        x6 = self.transpose6(x)
        x = cat([x6, x4], dim = 1)
        x = self.dropout6(x)
        x = self.conv6(x)
        
        x7 = self.transpose7(x)
        x = cat([x7, x3], dim = 1)
        x = self.dropout7(x)
        x = self.conv7(x)
        
        x8 = self.transpose8(x)
        x = cat([x8, x2], dim = 1)
        x = self.dropout8(x)
        x = self.conv8(x)
        
        x9 = self.transpose9(x)
        x = self.cat9([x9, x1], dim = 1)
        x = self.dropout9(x)
        x = self.conv9(x)
        
        x = self.outc(x)
        return F.sigmoid(x)
   
   
class Conv2d_block(Module):
    # function that adds two convolutional layers
    def __init__(self, in_channel, out_channel):
        super(Conv2d_block, self).__init__()
        self.model = nn.Sequential(
            Conv2d(in_channel, out_channel, kernel_size = 3, padding = 1),
            ReLU(inplace = True),
            Conv2d(out_channel, out_channel, kernel_size = 3, padding = 1),
            ReLU(inplace = True)
        )

    def forward(self, x):
        return self.model(x)