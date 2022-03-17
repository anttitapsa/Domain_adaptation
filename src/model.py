import torch
from torch.nn import Conv2d
from torch.nn import Module
from torch.nn import MaxPool2d
from torch.nn import BatchNorm2d
from torch.nn import ReLU
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
    def __init__(self, numChannels = 1, classes = 2, dropout = 0.1):
        super(Unet, self).__init__()
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
        
        x = self.conv5(x)
        
        x6 = self.transpose6(x)
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
        return torch.sigmoid(x)

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
   
   
# Ready model for unet with GRL from https://github.com/eliottbrion/unsupervised-domain-adaptation-unet-keras/blob/master/models.py
def unet_L6(params):

    volumes = Input(batch_shape=(None, *image_size, 1))

    # Downsampling

    conv1 = Conv3D(params['n_feat_maps'], (3, 3, 3), activation='relu', padding='same')(volumes)
    conv1 = Conv3D(params['n_feat_maps'], (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(params['n_feat_maps'] * 2, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(params['n_feat_maps'] * 2, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(params['n_feat_maps'] * 4, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(params['n_feat_maps'] * 4, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(params['n_feat_maps'] * 8, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(params['n_feat_maps'] * 8, (3, 3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    conv5 = Conv3D(params['n_feat_maps'] * 16, (3, 3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv3D(params['n_feat_maps'] * 16, (3, 3, 3), activation='relu', padding='same')(conv5)
    pool5 = MaxPooling3D(pool_size=(2, 2, 2))(conv5)

    # Bottleneck

    conv6 = Conv3D(params['n_feat_maps'] * 32, (3, 3, 3), activation='relu', padding='same')(pool5)
    conv6 = Conv3D(params['n_feat_maps'] * 32, (3, 3, 3), activation='relu', padding='same')(conv6)
    model_feature = conv6
    conv6 = tf.slice(conv6, [0, 0, 0, 0, 0], [params['batch_size'] // 2, 5, 5, 4, params['n_feat_maps'] * 32])

    # Upsampling

    copy5 = tf.slice(conv5, [0, 0, 0, 0, 0], [params['batch_size'] // 2, 10, 10, 8, params['n_feat_maps'] * 16])
    up7 = concatenate(
        [Conv3DTranspose(params['n_feat_maps'] * 16, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), copy5], axis=4)
    conv7 = Conv3D(params['n_feat_maps'] * 16, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(params['n_feat_maps'] * 16, (3, 3, 3), activation='relu', padding='same')(conv7)

    copy4 = tf.slice(conv4, [0, 0, 0, 0, 0], [params['batch_size'] // 2, 20, 20, 16, params['n_feat_maps'] * 8])
    up8 = concatenate(
      [Conv3DTranspose(params['n_feat_maps'] * 8, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7), copy4], axis=4)
    conv8 = Conv3D(params['n_feat_maps'] * 8, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv3D(params['n_feat_maps'] * 8, (3, 3, 3), activation='relu', padding='same')(conv8)

    copy3 = tf.slice(conv3, [0, 0, 0, 0, 0], [params['batch_size'] // 2, 40, 40, 32, params['n_feat_maps'] * 4])
    up9 = concatenate(
      [Conv3DTranspose(params['n_feat_maps'] * 4, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv8), copy3], axis=4)
    conv9 = Conv3D(params['n_feat_maps'] * 4, (3, 3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv3D(params['n_feat_maps'] * 4, (3, 3, 3), activation='relu', padding='same')(conv9)

    copy2 = tf.slice(conv2, [0, 0, 0, 0, 0], [params['batch_size'] // 2, 80, 80, 64, params['n_feat_maps'] * 2])
    up10 = concatenate(
      [Conv3DTranspose(params['n_feat_maps'] * 2, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv9), copy2], axis=4)
    conv10 = Conv3D(params['n_feat_maps'] * 2, (3, 3, 3), activation='relu', padding='same')(up10)
    conv10 = Conv3D(params['n_feat_maps'] * 2, (3, 3, 3), activation='relu', padding='same')(conv10)

    copy1 = tf.slice(conv1, [0, 0, 0, 0, 0], [params['batch_size'] // 2, 160, 160, 128, params['n_feat_maps'] * 1])
    up11 = concatenate(
      [Conv3DTranspose(params['n_feat_maps'] * 1, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv10), copy1], axis=4)
    conv11 = Conv3D(params['n_feat_maps'] * 1, (3, 3, 3), activation='relu', padding='same')(up11)
    conv11 = Conv3D(params['n_feat_maps'] * 1, (3, 3, 3), activation='relu', padding='same')(conv11)

    segmentation_pred = Conv3D(2, (1, 1, 1), activation = 'softmax', name="segmentation_output")(conv11)

    # === Domain classifier ===

    feat = GradReverse(0.)(model_feature)

    convA = Conv3D(params['d_arch'][0], (3, 3, 3), activation='relu', padding='same')(feat)
    convA = Conv3D(params['d_arch'][0], (3, 3, 3), activation='relu', padding='same')(convA)
    fl = Flatten()(convA)
    dense = Dense(params['d_arch'][1], activation='relu')(fl)

    domain_pred = Dense(1, activation='sigmoid', name='domain_output')(dense)

    model = Model(inputs=[volumes], outputs=[segmentation_pred, domain_pred])

    losses = {
        "segmentation_output": dice_loss,
        "domain_output": "binary_crossentropy",
    }
    lossWeights = {"segmentation_output": 1.0, "domain_output": params['psi']}

    model.compile(optimizer=Adam(params['lr']), loss=losses, loss_weights=lossWeights,
                  metrics={'segmentation_output': None, 'domain_output': 'accuracy'})

    return model
