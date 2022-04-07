from data_loader import MaskedDataset, LIVECELL_IMG_DIR, LIVECELL_MASK_DIR
import data_loader
from matplotlib import pyplot as plt
import torch
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import to_pil_image
import numpy as np
from model import Unet


def imshow_side_by_side(imgs):
    for i, img in enumerate(imgs):
        plt.subplot(1,len(imgs),i+1)
        plt.imshow(to_pil_image(img).convert("L"), cmap='gray')
    plt.show()


def imshow_on_top_of_each_other(imgs):
    for i, img in enumerate(imgs):
        plt.subplot(len(imgs),1,i+1)
        plt.imshow(to_pil_image(img).convert("L"), cmap='gray')
    plt.show()


def imshow_side_by_side_model(idx, dataset, model, backprop = False):
    # Disable grad
    with torch.no_grad():
        image, mask = dataset[idx]
        if backprop:
            prediction, dom_lab = model(torch.unsqueeze(image, dim=0), 1)
            print('Domain label', dom_lab)
        else: prediction = model(torch.unsqueeze(image, dim=0))
        prediction = torch.squeeze(prediction, dim=0)
        imshow_side_by_side([image, mask, prediction])
