import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms
# python -m pip install -U scikit-image
from skimage.util import random_noise

# function adds noise to tensors (and flips them)
def add_noise_to_images(image, amount):
    noise_image = image + torch.randn(image.size()) * 0.07 + 0.1 
    noise_image = torch.tensor(random_noise(noise_image, mode = 's&p', salt_vs_pepper = 0.0, amount=amount))
    '''
    vflipper = transforms.RandomVerticalFlip(p=0.5)
    transformed_image = [vflipper(noise_image) for _ in range(4)]
    hflipper = transforms.RandomHorizontalFlip(p=0.5)
    transformed_image = [hflipper(transformed_image[0]) for _ in range(4)][0]
    '''
    return noise_image

def resize_image_(image, size):
    resize = Trolli.Resize(size)
    return resize.forward(image)

def to_same_size(image, target_image):
    return resize_image_(image, (target_image.shape[0], target_image.shape[1]))
