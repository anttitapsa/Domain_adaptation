# A function that creates a binary mask from a PNG image

from PIL import Image
import numpy as np
import os


BLUE = (0, 0, 255, 255)


def pngs_to_mask(path, mask_path):

    for img_path in os.listdir(path):
        img_number = img_path[len('segmentation_'):-4]

        img = Image.open(path + img_path)

        # Resize image to match UNet input size
        img = img.resize((512, 512))

        # Get pixel data
        sequence_of_pixels = img.getdata()
        pixels = list(sequence_of_pixels)

        # Change to binary values
        pixels = [1 if x == BLUE else 0 for x in pixels]

        # Change to numpy array and reshape
        pixels = np.array(pixels)
        pixels = np.reshape(pixels, img.size)

        # Save mask as numpy array in mask dir
        np.save(mask_path + 'img_' + img_number + '_mask.npy', pixels)

