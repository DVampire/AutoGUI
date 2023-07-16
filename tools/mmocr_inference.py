import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
import pathlib
import numpy as np
import cv2
import torch
from mmocr.apis.inferencers import MMOCRInferencer
from mmocr.utils import poly2bbox

ROOT = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.append(ROOT)

def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    """
    num_boxes = dt_boxes.shape[0]
    boxes = sorted(enumerate(dt_boxes), key=lambda x: (x[1][1], x[1][0]))

    sorted_indices, sorted_boxes  = [], []
    for item in boxes:
        sorted_indices.append(item[0])
        sorted_boxes.append(item[1])

    _boxes = list(sorted_boxes)
    _indices = list(sorted_indices)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][1] - _boxes[j][1]) < 10 and (_boxes[j + 1][0] < _boxes[j][0]):
                tmp = _boxes[j]
                tmp_index = _indices[j]
                _boxes[j] = _boxes[j + 1]
                _indices[j] = _indices[j + 1]
                _boxes[j + 1] = tmp
                _indices[j + 1] = tmp_index
            else:
                break
    return _indices, _boxes

class MMOCRInference():
    def __init__(self, device = None):

        det = os.path.join(ROOT, "playground/playground/mmocr_sam", "mmocr_dev/configs/textdet/dbnetpp/dbnetpp_swinv2_base_w16_in21k.py")
        det_weights = os.path.join(ROOT, "playground", "checkpoints/mmocr/db_swin_mix_pretrain.pth")
        rec = os.path.join(ROOT, "playground/playground/mmocr_sam", "mmocr_dev/configs/textrecog/abinet/abinet_20e_st-an_mj.py")
        rec_weights = os.path.join(ROOT, "playground", "checkpoints/mmocr/abinet_20e_st-an_mj_20221005_012617-ead8c139.pth")

        self.mmocr_inference = MMOCRInferencer(
            det,
            det_weights,
            rec,
            rec_weights,
            device=device)

    def infer_auto(self, image):
        result = self.mmocr_inference(image, save_vis=True, out_dir="./")['predictions'][0]

        det_polygons = result['det_polygons']
        det_bboxes = np.array([poly2bbox(poly) for poly in det_polygons])
        rec_texts = result['rec_texts']

        indices, boxes = sorted_boxes(det_bboxes)

        rec_texts = [rec_texts[index] for index in indices]

        return rec_texts

if __name__ == '__main__':
    device = torch.device("cuda")

    image = cv2.imread("twolane.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mmocr_inference = MMOCRInference(device = device)

    rec_texts = mmocr_inference.infer_auto(image)
    print(rec_texts)