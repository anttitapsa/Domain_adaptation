# File for adding data handelers

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from torchvision import transforms
from torchvision.io import read_image
import os
import numpy as np
from tqdm import tqdm


if os.path.basename(os.getcwd()) != "lst-project":
    raise Exception(f"You are in {os.getcwd()}, please move into root directory lst-project.")


DATA_DIR = os.path.join(os.getcwd(), "data")
TARGET_DATA_DIR = os.path.join(DATA_DIR, "target")
LIVECELL_IMG_DIR = os.path.join(DATA_DIR, "livecell", "images")
LIVECELL_MASK_DIR = os.path.join(DATA_DIR, "livecell", "masks")
IMG_SIZE = 512


class DataLoaderException(Exception):
    pass


class UnMaskedDataset(Dataset):
    """
    A dataset without masks.
    img_dir: the directory containing images.
    """
    def __init__(self, img_dir):
        # data loading
        self.img_dir = img_dir

    def __getitem__(self, idx):
        # return item with possible transforms
        img_path = os.path.join(self.img_dir, os.listdir(self.img_dir)[idx])
        image = read_image(img_path)
        return Resize((IMG_SIZE, IMG_SIZE), image)

    def __len__(self):
        # return len(dataset)
        return len(os.listdir(self.img_dir))


class MaskedDataset(Dataset):
    """
    img_dir: The directory containing ONLY images used foor this data set
    mask_dir: The directory containing ONLY masks of images in the same order as the images are found in img_dir
    """
    def __init__(self, img_dir, mask_path, lenght=None, in_memory=False):
        if not os.path.isdir(img_dir):
            raise DataLoaderException(f"The first argument 'img_dir' is not a directory, it is {img_dir}")
        if not os.path.isdir(mask_path):
            raise DataLoaderException(f"The second argument 'mask_path' is not a directory, it is {mask_path}")
        
        self.im_suffix = "." + os.listdir(img_dir)[0].split(".")[-1]
        self.ids = []
        self.masks = {}
        self.images = {}  # save images into memory if in_memory is set to True
        self.lenght = lenght
        self.in_memory = in_memory
        print("Reading masks...")
        for filename in tqdm(os.listdir(mask_path)):
            # Masks should be named after the original image file.
            # For example the image "image.png" should have a mask named "image_mask.npy"
            # The identifier for the mask should then be "image"
            identifier = filename.split("_mask")[0]
            self.ids.append(identifier)

            if in_memory:
                path = os.path.join(mask_path, filename)
                try:
                    self.masks[identifier] = torch.from_numpy(np.load(path))
                except FileNotFoundError:
                    raise DataLoaderException(
                        f"Couldn't read mask {filename}. Make sure your masks are saved as np arrays in {mask_path}")
                try:
                    image = Image.open(img_path)
                except FileNotFoundError:
                    raise DataLoaderException(
                        f"Couldn't read image {img_path}. Make sure your data is located in {img_dir}!")
                self.images[identifier] = image
            else:
                self.mask_path = mask_path
                self.img_dir = img_dir
            
        print("Masks successfully read!")

    def __getitem__(self, idx):
        name = self.ids[idx]

        if self.in_memory:
            image = self.images[name]
            mask = self.masks[name]
        else:
            img_path = os.path.join(self.img_dir, name + self.im_suffix)
            mask_path = os.path.join(self.mask_path, name + "_mask.npy")
            try:
                image = Image.open(img_path)
            except FileNotFoundError:
                raise DataLoaderException(
                    f"Couldn't read image {img_path}. Make sure your data is located in {self.img_dir}!")
            try:
                mask = torch.from_numpy(np.load(mask_path))
            except FileNotFoundError:
                raise DataLoaderException(
                    f"Couldn't read mask {mask_path}. Make sure your masks are located in {self.mask_path}!")
        
        image = transforms.ToTensor()(image)

        i, j, h, w = transforms.RandomCrop.get_params(
            image,
            output_size=(IMG_SIZE, IMG_SIZE))
        image = transforms.functional.crop(image, i, j, h, w)
        mask = transforms.functional.crop(mask, i, j, h, w)

        return image, mask
        
    def __len__(self):
        # return len(dataset)
        if self.lenght:
            return self.lenght
        return len(os.listdir(self.img_dir))        
