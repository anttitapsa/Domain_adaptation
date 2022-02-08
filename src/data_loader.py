# File for adding data handelers

from torch.utils.data import Dataset, DataLoader

class CellDataset(Dataset):
    
    def __init__(self):
        # data loading
        pass
    
    def __getitem__(self, index):
        # return item with possible transforms
        pass
        
    def __len__(self):
        # return len(dataset)
        pass