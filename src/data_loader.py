# File for adding data handelers

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import os


class TargetDataset(Dataset):
    """ This dataset contains only target domain images that do not have labels """
    def __init__(self):
        # data loading
        self.img_dir = os.path.join(os.getcwd(), "data", "target")

    
    def __getitem__(self, idx):
        # return item with possible transforms
        img_path = os.path.join(self.img_dir, os.listdir(self.img_dir)[idx])
        image = read_image(img_path)
        label = None
        return image, label
        
    def __len__(self):
        # return len(dataset)
        return len(os.listdir(self.img_dir))


class SourceDataset(Dataset):
    """ This dataset contains source domain images and labels """
    def __init__(self):
        # data loading
        pass
    
    def __getitem__(self, index):
        # return item with possible transforms
        pass
        
    def __len__(self):
        # return len(dataset)
        pass
