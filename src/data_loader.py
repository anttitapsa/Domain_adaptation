# File for adding data handelers

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import os

if os.path.basename(os.getcwd()) != "lst-project":
    raise Exception(f"You are in {os.getcwd()}, please move into root directory lst-project.")

DATA_DIR = os.path.join(os.getcwd(), "data")
TARGET_DATA_DIR = os.path.join(DATA_DIR, "target")
LIVECELL_IMG_DIR = os.path.join(DATA_DIR, "livecell", "images")
LIVECELL_MASK_DIR = os.path.join(DATA_DIR, "livecell", "masks")


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
        return image
        
    def __len__(self):
        # return len(dataset)
        return len(os.listdir(self.img_dir))


class MaskedDataset(Dataset):
    """
    img_dir: The directory containing ONLY images used foor this data set
    mask_dir: The directory containing ONLY masks of images in the same order as the images are found in img_dir
    """
    def __init__(self, img_dir, mask_path):
        self.img_dir = img_dir
        try:
            self.masks = np.load(mask_path)
        except FileNotFoundError:
            raise DataLoaderException(f"Couldn't read mask {mask_path}. Make sure your masks are saved as np arrays in {self.mask_path}")
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, os.listdir(self.img_dir)[idx])
        try:
            image = read_image(img_path)
        except FileNotFoundError:
            raise DataLoaderException(f"Couldn't read image {img_path}. Make sure your data is located in {self.img_dir}!")
        return image, self.masks[idx]
        
    def __len__(self):
        # return len(dataset)
        return len(os.listdir(self.img_dir))
