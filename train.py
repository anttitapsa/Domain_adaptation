import sys
from tqdm import tqdm
import os
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from src.model import Unet
from src.data_loader import MaskedDataset
from torch.utils.data import DataLoader

'''
based on https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
'''
DATA_DIR = os.path.join(os.getcwd(), "data")
LIVECELL_IMG_DIR = os.path.join(DATA_DIR, "livecell", "images")
LIVECELL_MASK_DIR = os.path.join(DATA_DIR, "livecell", "masks")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

criterion = CrossEntropyLoss()
optimizer = optim.Adam(Unet().parameters(),  lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
trolli = MaskedDataset(LIVECELL_IMG_DIR, LIVECELL_MASK_DIR)

dataloader = DataLoader(trolli,batch_size=4,
                        shuffle=False, num_workers=0, )

net = Unet()
net.to(device)
net.train()
for epoch in range(10): 
    print('./Unet' + str(epoch) + '.pth')
    running_loss = 0.0
    for i, data in enumerate(tqdm(dataloader)):
        inputs, labels = data[0].to(device), data[1].to(device)

        inputs.to(device)
        optimizer.zero_grad()
       
        outputs = net.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 0:    # print every 200 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
    
    PATH = './Unet' + str(epoch) + '.pth'
    torch.save(net.state_dict(), PATH)
print('Finished Training')
