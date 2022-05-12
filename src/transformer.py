import numpy as np
from PIL import Image
import torch
from torchvision import transforms
# python -m pip install -U scikit-image
from skimage.util import random_noise
import random
from tqdm import tqdm
import torch.nn as nn
import torchvision.transforms as T


# function adds noise to tensors (and flips them)
def add_noise_to_images(image, amount = 0.05):
    noise_image = image + torch.randn(image.size()) * 0.07 + 0.1 
    noise_image = torch.tensor(random_noise(noise_image, mode = 's&p', salt_vs_pepper = 0.0, amount=amount))
    return noise_image

def add_background_noise(image, device="cpu"):
    '''
    function changes the hue, brightness, contrast and saturation of the image, and it adds also random blur to images
    '''
    transforms = nn.Sequential(
        T.ColorJitter(brightness=[0,2], hue=[-0.5,0.5], contrast=[0,3], saturation=[0,3]),
        T.GaussianBlur(kernel_size=(51, 91), sigma=(0.1,6)))
    aug_img = transforms(image.to(device))
    return aug_img.to("cpu")

def add_fake_magnetballs(image, mask, min_amount = 30, max_amount = 70, max_lightness=0.15):
    # Getting the dimensions of the image
    image_temp = torch.clone(image).numpy()
    mask_temp = torch.clone(mask).numpy()
    channels, row, col = image.shape
    number_of_pixels = random.randint(min_amount, max_amount)
   
    x_mesh,y_mesh = np.mgrid[:row,:col] #uus
    for i in range(number_of_pixels): 
        r = random.choice([5,7,10,13])
        # Pick a random y coordinate
        y_coord=random.randint(r, row - 1 - r)
        # Pick a random x coordinate
        x_coord=random.randint(r, col - 1 - r)
        # Pick color randomly, not always as black
        black_col = np.random.uniform(0,max_lightness,1)[0]

        #uus
        distance = np.linalg.norm(np.stack([x_mesh-x_coord,y_mesh-y_coord]),axis=0)
        mask = distance<r
        image_temp[0, mask] = black_col
        mask_temp[0,mask] = black_col

    return transforms.ToTensor()(image_temp).permute(1,2,0), transforms.ToTensor()(mask_temp).squeeze(0).permute(1,2,0)

def rebuild(crop_list, original_size):
    rows = []
    for row in crop_list:
        new_row = torch.cat(row, -1)
        rows.append(new_row)
    return T.functional.crop(torch.cat(rows, 2), top=0, left=0, height=original_size[0], width=original_size[1])


def slice_image(image, crop_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image.to(device)

    top = 0 
    left = 0

    image_crops = []
    row = []

    while 1:
        crop = T.functional.crop(img=image, top=top, left=left, height=crop_size, width=crop_size)
        row.append(crop)
        if left + crop_size  > image.shape[3]:
            image_crops.append(row)
            top += crop_size
            left = 0
            row = []
        else:
            left+=crop_size
        if (top  > image.shape[2]) and (left +crop_size  > image.shape[3]):
            break
    return image_crops

'''

def fake_magnetball_livecell_no_mask(dataset, min_amount = 30, max_amount = 70, IMG_SIZE=512):

    new_dataset = [torch.zeros(1,IMG_SIZE, IMG_SIZE)] *len(dataset)
    
    i = 0
    for data in tqdm(dataset):
        new_dataset[i] = add_fake_magnetballs(image=data[0])
        i += 1

    return torch.stack(new_dataset)

def resize_image_(image, size):
    resize = transforms.Resize(size)
    return resize.forward(image)

def to_same_size(image, target_image):
    return resize_image_(image, (target_image.shape[0], target_image.shape[1]))

'''