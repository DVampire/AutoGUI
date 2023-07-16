import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pickle as pkl
import pyautogui
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch


ROOT = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.append(ROOT)

from sam_inference import SAMInference, show_anns, show_box, show_mask, show_points
from mmocr_inference import MMOCRInference

def main():
    device = torch.device("cuda")

    pyautogui.hotkey("ctrl", "win", "right")

    screenWidth, screenHeight = pyautogui.size()
    print(screenWidth, screenHeight)

    currentMouseX, currentMouseY = pyautogui.position()
    print(currentMouseX, currentMouseY)

    image = pyautogui.screenshot()

    image.save('screen.png')

    image = np.array(image)

    auto_params = dict(
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )

    infer = SAMInference(auto_params=auto_params, device=device)
    masks = infer.infer_auto(image)

    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.savefig("screen_seg.png")

    with open("screen.pkl","wb") as op:
        pkl.dump(masks, op)

def infer_screen1():
    device = torch.device("cuda")

    batch_input = []

    image = cv2.imread("screen.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # construct points
    input_points = []
    input_labels = []
    base_point = np.array([[788,1017]])
    threshold = 15
    for item in [np.array([i,j]) for i in [threshold] for j in [-threshold, 0, threshold]]:
        pos = base_point + item
        input_points.append(pos)
        input_labels.append([1])

    input_boxes = None
    input_points = np.array(input_points)
    input_labels = np.array(input_labels)
    item = {
        "image": image,
        "input_points": input_points,
        "input_labels": input_labels,
        "input_boxes": input_boxes,
        "input_masks": None,
    }
    batch_input.append(item)

    infer = SAMInference(device=device)

    batched_output = infer.infer_prompt(batch_input, multimask_output=False)

    fig, ax = plt.subplots(1, 1, figsize=(20, 20))

    ax.imshow(batch_input[0]["image"])
    for mask in batched_output[0]['masks']:
        show_mask(mask.cpu().numpy(), ax, random_color=True)
    for points,labels in zip(batch_input[0]["input_points"], batch_input[0]["input_labels"]):
        show_points(points, labels, ax)
    ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # main()
    infer_screen1()