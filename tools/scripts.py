import pyautogui
from sam_inference import Inference, show_anns
import torch
import numpy as np
import matplotlib.pyplot as plt

def main():
    device = torch.device("cpu")

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

    infer = Inference(auto_params=auto_params, device=device)
    masks = infer.infer_auto(image)

    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.savefig("screen_seg.png")

if __name__ == '__main__':
    main()