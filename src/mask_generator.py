from pycocotools.coco import COCO
import os
import sys
import numpy as np
from tqdm import tqdm

def main():
    """
    Script that generates binary masks (numpy arrays) for LiveCell images found in images directory and in annotation json file. Script saves masks 
    in masks directory as npy files which are named as images + _mask.

    For running the script, you need to give two arguments in this order
    1. absolutepath to annotation json file.
    2. absolutepath to images directory

    Example of running
    python mask_generator.py C:\livecell_coco_train.json C:\data\livecell\images

    Script assumes that data is stored in folder structure below. If there is no masks directory allready, scrit creates it
    
    src
    |    |
    |    mask_generator.py
    |
    data
        |
        livecell
            |
            images
            |
            masks

    """
    if os.path.isfile(sys.argv[1]) == False or os.path.isdir(sys.argv[2]) == False or len(sys.argv) != 3:
        print("Give two arguments. \n First argument is path to annotation json file.\n Second argument is path to images directory")
    
    else:  
        annotation_dir = sys.argv[1]
        images_dir = sys.argv[2]  
        
        coco = COCO(annotation_dir)
        img_ids = list(coco.imgs.keys())
        count = len(os.listdir(images_dir))
        save_path = os.path.join(os.curdir, os.pardir, "data", "livecell", "masks")
        if not os.path.isdir(save_path):
            os.mkdir(save_path)


        for index in tqdm(range(count)):
            path = eval(str(coco.loadImgs(img_ids[index])[0]))
            an_ids = coco.getAnnIds(imgIds=path["id"])
            anns = coco.loadAnns(an_ids)

            mask = coco.annToMask(anns[0])
            for i in range(len(anns)):
                mask += coco.annToMask(anns[i])
            # Changes values to 0 or 1 --> chages mask to binary mask
            mask = np.where(mask> 0, 1, 0)

            np.save(os.path.join(save_path, path["file_name"].split(".")[0] +"_mask"), mask)

if __name__ == "__main__":
    main()