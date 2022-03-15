# A function to blur images

from PIL import Image, ImageFilter
import os
import random


def blur_images(path, new_path, show = False):

    for img_path in os.listdir(path):
        img_number = img_path[len('rgb_'):-4]

        orig = Image.open(path + img_path)
        if show:
            orig.show()

        # Resize image to match UNet input size
        orig = orig.resize((512, 512))

        # Intensity should be between 1 and 5 - could be randomised?
        blurred = orig.filter(ImageFilter.GaussianBlur(random.randint(1, 3)))
        if show:
            blurred.show()

        blurred.save(new_path + 'img_' + img_number + '.png')
