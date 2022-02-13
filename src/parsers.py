# Add functions that parse data, e.g. split target domain videos into images
from pycocotools.coco import COCO
import numpy as np

def generate_mask(annotation_dir, index):
    """
    Generates annotation mask (numpy darray) for image with some index (int)
    annotation_dir -->  (str) path to directory where json file, containing annotions, is located
                        For Example "livecell_coco_train.json"
    index          -->  (int) index of image
    """
    
    coco = COCO(annotation_dir)

    img_ids = list(coco.imgs.keys())
    path = eval(str(coco.loadImgs(img_ids[index])[0]))
    an_ids = coco.getAnnIds(imgIds=path["id"])
    anns = coco.loadAnns(an_ids)

    mask = coco.annToMask(anns[0])
    for i in range(len(anns)):
        mask += coco.annToMask(anns[i])
    
    # Changes values to 0 or 1 --> chages mask to binary mask
    #mask = np.where(mask> 0, 1, 0)

    return mask