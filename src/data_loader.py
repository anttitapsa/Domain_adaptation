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
UNITY_IMG_DIR = os.path.join(DATA_DIR, "unity_data", "images")
UNITY_MASK_DIR = os.path.join(DATA_DIR, "unity_data", "masks")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_target_data", "images")
TEST_MASK_DIR = os.path.join(DATA_DIR, "test_target_data", "masks")
dir_checkpoint = os.path.join(os.getcwd(), "model" )
IMG_SIZE = 512
DOMAIN_MAP = {
    "livecell": 0,
    "target": 1,
    "unity_data": 0,
    "test_target_data": 1
}


class DataLoaderException(Exception):
    pass


class UnMaskedDataset(Dataset):
    """
    A dataset without masks.
    img_dir: the directory containing images.
    """
    def __init__(self, img_dir, mode=1, return_domain_identifier=False):
        # data loading
        # mode: 1 --> RandomCrop when using __getitem__
        #       2 --> Resized using IMAGE_SIZE when using __getitem__
        if not os.path.isdir(img_dir):
            raise DataLoaderException(f"The first argument 'img_dir' is not a directory, it is {img_dir}")
        if type(mode) != int:
            raise DataLoaderException(f"The mode can be only int and it can be 1 or 2, it is {mode}")
        elif mode < 1 or mode > 2:
            raise DataLoaderException(f"The mode can be only int and it can be 1 or 2, it is {mode}")
        self.img_dir = img_dir
        self.mode = mode
        self.return_domain_identifier = return_domain_identifier
        if return_domain_identifier:
            # Directory basename will work as the identifier for domains.
            # The directories are structured for example ".../livecell/images"
            # We want to extract "livecell" out of the path, as done below.
            # Domain map will then convert these into numeric, predictable classes
            self.domain_identifier = DOMAIN_MAP[os.path.basename(self.img_dir)]

    def __getitem__(self, idx):
        # return item with possible transforms
        img_path = os.path.join(self.img_dir, os.listdir(self.img_dir)[idx])
        # Image is changed grayscale
        try:
            image = Image.open(img_path).convert('L')
        except FileNotFoundError:
            raise DataLoaderException(
                f"Couldn't read image {img_path}. Make sure your data is located in {self.img_dir}!")
        
        image = transforms.ToTensor()(image)
        if self.mode == 1:
            i, j, h, w = transforms.RandomCrop.get_params(
                image,
                output_size=(IMG_SIZE, IMG_SIZE))
            image = transforms.functional.crop(image, i, j, h, w)
            if self.return_domain_identifier:
                return image, self.domain_identifier
            else: return image  # No mask --> None
        elif self.mode == 2:
            resize = transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=Image.NEAREST)
            if self.return_domain_identifier: return image, self.domain_identifier
            else: return image  # No mask --> None
    

    def __len__(self):
        # return len(dataset)
        return len(os.listdir(self.img_dir))


class MaskedDataset(Dataset):
    """
    img_dir: The directory containing ONLY images used foor this data set
    mask_dir: The directory containing ONLY masks of images in the same order as the images are found in img_dir
    """
    def __init__(self, img_dir, mask_path, length=None, in_memory=False, return_domain_identifier=False):
        if not os.path.isdir(img_dir):
            raise DataLoaderException(f"The first argument 'img_dir' is not a directory, it is {img_dir}")
        if not os.path.isdir(mask_path):
            raise DataLoaderException(f"The second argument 'mask_path' is not a directory, it is {mask_path}")
        
        self.im_suffix = "." + os.listdir(img_dir)[0].split(".")[-1]
        self.ids = []
        self.masks = {}
        self.images = {}
        self.length = length
        self.in_memory = in_memory  # If true, read whole dataset into memory on initialization.
        iter_count = length if length else len(os.listdir(img_dir))
        self.return_domain_identifier = return_domain_identifier
        if return_domain_identifier:
            # Directory basename will work as the identifier for domains.
            # The directories are structured for example ".../livecell/images"
            # We want to extract "livecell" out of the path, as done below.
            # Domain map will then convert these into numeric, predictable classes
            self.domain_identifier = DOMAIN_MAP[os.path.basename(os.path.split(img_dir)[0])]

        print("Reading data into memory...")
        for i in tqdm(range(iter_count)):
            mask_file = os.listdir(mask_path)[i]

            # Masks should be named after the original image file.
            # For example the image "image.png" should have a mask named "image_mask.npy"
            # The identifier for the mask should then be "image"
            identifier = mask_file.split("_mask")[0]
            self.ids.append(identifier)
            img_name = identifier + self.im_suffix

            if in_memory:
                m_path = os.path.join(mask_path, mask_file)
                try:
                    self.masks[identifier] = torch.from_numpy(np.load(m_path))
                except FileNotFoundError:
                    raise DataLoaderException(
                        f"Couldn't read mask {m_path}. Make sure your masks are saved as np arrays in {mask_path}")
                i_path = os.path.join(img_dir, img_name)
                try:
                    image = Image.open(i_path)
                except FileNotFoundError:
                    raise DataLoaderException(
                        f"Couldn't read image {i_path}. Make sure your data is located in {img_dir}!")
                self.images[identifier] = image
            else:
                self.mask_path = mask_path
                self.img_dir = img_dir
            
        print("Dataset initialized!")

    def __getitem__(self, idx):
        name = self.ids[idx]

        if self.in_memory:
            image = self.images[name]
            mask = self.masks[name]
        else:
            img_path = os.path.join(self.img_dir, name + self.im_suffix)
            mask_path = os.path.join(self.mask_path, name + "_mask.npy")
            try:
                image = Image.open(img_path).convert('L')
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
        
        if self.return_domain_identifier:
            return image, mask, self.domain_identifier
        return image, mask
        
    def __len__(self):
        if self.length:
            return self.length
        return len(os.listdir(self.img_dir))        
