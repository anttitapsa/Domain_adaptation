import numpy as np
from PIL import Image
import torch
from torchvision import transforms
# python -m pip install -U scikit-image
from skimage.util import random_noise
from data_loader import MaskedDataset, UnMaskedDataset
import random
# function adds noise to tensors (and flips them)
def add_noise_to_images(image, amount = 0.05):
    noise_image = image + torch.randn(image.size()) * 0.07 + 0.1 
    noise_image = torch.tensor(random_noise(noise_image, mode = 's&p', salt_vs_pepper = 0.0, amount=amount))
    '''
    vflipper = transforms.RandomVerticalFlip(p=0.5)
    transformed_image = [vflipper(noise_image) for _ in range(4)]
    hflipper = transforms.RandomHorizontalFlip(p=0.5)
    transformed_image = [hflipper(transformed_image[0]) for _ in range(4)][0]
    '''
    return noise_image

def add_fake_magnetballs(image, mask, min_amount = 30, max_amount = 70):
    # Getting the dimensions of the image
    image_temp = torch.clone(image).numpy()
    mask_temp = torch.clone(mask).numpy()
    channels, row, col = image.shape
    number_of_pixels = random.randint(min_amount, max_amount)
    for i in range(number_of_pixels): 
        r = random.choice([5,7,10,13])
        # Pick a random y coordinate
        y_coord=random.randint(r, row - 1 - r)
        # Pick a random x coordinate
        x_coord=random.randint(r, col - 1 - r)
        for s in range(r):
            for i in range(0,360,1):
                x1 = s * np.cos(i)
                y1 = s * np.sin(i)
                image_temp[0, y_coord+y1.astype(int),x_coord+x1.astype(int)] = 0 
                mask_temp[y_coord+y1.astype(int),x_coord+x1.astype(int)] = 0
    return transforms.ToTensor()(image_temp).permute(1,2,0), transforms.ToTensor()(mask_temp).squeeze(0)

def fake_magnetball_livecell(dataset, min_amount = 30, max_amount = 70):
    new_dataset = dataset.copy()
    i = 0
    for data in dataset:
        new_dataset[i] = add_fake_magnetballs(data[0], data[1], min_amount, max_amount)
        i += 1
    return new_dataset

def resize_image_(image, size):
    resize = transforms.Resize(size)
    return resize.forward(image)

def to_same_size(image, target_image):
    return resize_image_(image, (target_image.shape[0], target_image.shape[1]))
