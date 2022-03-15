import numpy as np
import cv2
from skimage.util import random_noise
from PIL import Image
import torch
import torchvision.transforms as Trolli

def add_noise_to_images(noise_type, image):
    if noise_type == 'Gaussian':
        gauss = np.random.normal(0,1,image.size)
        gauss = gauss.reshape(image.shape[0],image.shape[1]).astype('uint8') 
        noise_image = cv2.add(image, gauss)
        
    elif noise_type == 'Speckle':
        gauss = np.random.normal(0,1,image.size)
        gauss = gauss.reshape(image.shape[0],image.shape[1]).astype('uint8')
        noise_image = image + image * gauss
    
    elif noise_type == 'Random':
        noise_image = random_noise(image, mode='s&p',amount=0.3)
    
    elif noise_type == 'Poisson':
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noise_image = np.random.poisson(image * vals) / float(vals)
        
    
    # random flip the image
    ''''vflipper = Trolli.RandomVerticalFlip(p=0.5)
    transformed_image = [vflipper(noise_image) for _ in range(4)]'''
    return noise_image

def resize_image_(image, size):
    resize = Trolli.Resize(size)
    return resize.forward(image)

def to_same_size(image, target_image):
    return resize_image_(image, (target_image.shape[0], target_image.shape[1]))
