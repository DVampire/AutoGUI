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

def main1():
    device = torch.device("cuda")

    pyautogui.hotkey("ctrl", "win", "right")

    screenWidth, screenHeight = pyautogui.size()
    print(screenWidth, screenHeight)

    currentMouseX, currentMouseY = pyautogui.position()
    print(currentMouseX, currentMouseY)

    image = pyautogui.screenshot()

    image.save('screen1.png')

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
    plt.savefig("screen_seg1.png")

    with open("screen1.pkl","wb") as op:
        pkl.dump(masks, op)

def main2():
    device = torch.device("cuda")

    pyautogui.hotkey("ctrl", "win", "right")

    screenWidth, screenHeight = pyautogui.size()
    print(screenWidth, screenHeight)

    currentMouseX, currentMouseY = pyautogui.position()
    print(currentMouseX, currentMouseY)

    image = pyautogui.screenshot()

    image.save('screen2.png')

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
    plt.savefig("screen_seg2.png")

    with open("screen2.pkl","wb") as op:
        pkl.dump(masks, op)

def infer_screen1():
    device = torch.device("cuda")

    batch_input = []

    image = cv2.imread("screen1.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # construct box
    base_point = (788,1017)
    width = 150
    height = 30
    boxes = np.array(
        [
            [base_point[0], max(base_point[1] - height, 0), base_point[0] + width, base_point[1]],
        ],
    )

    input_boxes = boxes
    input_points = None
    input_labels = None
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
    for box in batch_input[0]["input_boxes"]:
        show_box(box, ax)
    ax.axis('off')

    plt.tight_layout()
    plt.show()

def infer_screen2():
    device = torch.device("cuda")

    batch_input = []

    image = cv2.imread("screen2.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # construct box
    base_point = (699, 932)
    width = 500
    height = 400
    boxes = np.array(
        [
            [base_point[0], max(base_point[1] - height, 0), base_point[0] + width, base_point[1]],
        ],
    )

    input_boxes = boxes
    input_points = None
    input_labels = None
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
    for box in batch_input[0]["input_boxes"]:
        show_box(box, ax)
    ax.axis('off')

    plt.tight_layout()
    plt.show()

# 解析图标
def main3():
    device = torch.device("cuda")

    # pyautogui.hotkey("ctrl", "win", "right")
    #
    # screenWidth, screenHeight = pyautogui.size()
    # print(screenWidth, screenHeight)
    #
    # currentMouseX, currentMouseY = pyautogui.position()
    # print(currentMouseX, currentMouseY)
    #
    # image = pyautogui.screenshot()
    #
    # image.save('screen1.png')

    image = cv2.imread("screen1.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image)

    auto_params = dict(
        points_per_side=64,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )

    infer = SAMInference(auto_params=auto_params, device=device)
    masks = infer.infer_auto(image)

    # 选项栏(0, 966)
    bottom_base_point = (0, 966)
    outboxs = [
        item["bbox"] for item in masks if item["bbox"][1] > bottom_base_point[1]
    ]
    outpoints = [
        np.array([item["bbox"][0] + item["bbox"][2] // 2 , item["bbox"][1] + item["bbox"][3] // 2]) for item in masks if item["bbox"][1] > bottom_base_point[1]
    ]
    outlabels = [1 for _ in range(len(outpoints))]

    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    plt.imshow(image)

    if not os.path.exists("cut"):
        os.makedirs("cut")

    count = 0
    for point, label, box in zip(outpoints, outlabels, outboxs):
        show_points(point, label, ax)

        cropped_image = image[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join("cut", "{:04d}.png".format(count)), cropped_image)
        count += 1

    plt.axis('off')
    plt.savefig("screen_seg3.png")

if __name__ == '__main__':
    # main1()
    #infer_screen1()

    # main2()
    # infer_screen2()

    # 选项栏(0, 966)
    main3()