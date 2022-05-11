import numpy as np
from PIL import Image
import torch
from torchvision import transforms
# python -m pip install -U scikit-image
from skimage.util import random_noise
import random

def add_noise_to_images(image, amount = 0.05):
    noise_image = image + torch.randn(image.size()) * 0.07 + 0.1 
    noise_image = torch.tensor(random_noise(noise_image, mode = 's&p', salt_vs_pepper = 0.0, amount=amount))
    return noise_image

def add_fake_magnetballs(image, mask, min_amount = 30, max_amount = 70):
    image_temp = torch.clone(image).numpy()
    mask_temp = torch.clone(mask).numpy()
    channels, row, col = image.shape
    number_of_pixels = random.randint(min_amount, max_amount)
    x_mesh,y_mesh = np.mgrid[:row,:col]
    for i in range(number_of_pixels): 
        r = random.choice([5,7,10,13])
        # Pick a random y coordinate
        y_coord=random.randint(r, row - 1 - r)
        # Pick a random x coordinate
        x_coord=random.randint(r, col - 1 - r)
        distance = np.linalg.norm(np.stack([x_mesh-x_coord,y_mesh-y_coord]),axis=0)
        mask = distance<r
        image_temp[0, mask] = 0
        mask_temp[mask] = 0
    return transforms.ToTensor()(image_temp).permute(1,2,0), transforms.ToTensor()(mask_temp).squeeze(0)

def resize_image_(image, size):
    resize = transforms.Resize(size)
    return resize.forward(image)

def to_same_size(image, target_image):
    return resize_image_(image, (target_image.shape[0], target_image.shape[1]))