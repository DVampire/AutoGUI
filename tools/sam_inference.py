import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from segment_anything.utils.transforms import ResizeLongestSide


ROOT = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.append(ROOT)

from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

class Inference():
    def __init__(self,
                 checkpoint_path = os.path.join(ROOT, "sam", "checkpoints", "sam_vit_h_4b8939.pth"),
                 model_type = "vit_h",
                 auto_params=dict(),
                 device = torch.device("cpu")
                 ):
        """
        auto params:
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
        """

        if auto_params is None:
            auto_params = dict()
        self.device = device

        self.model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.model.to(device=self.device)
        self.predictor = SamPredictor(self.model)
        self.mask_generator = SamAutomaticMaskGenerator(self.model, **auto_params)

        self.resize_transform = ResizeLongestSide(self.model.image_encoder.img_size)

    def prepare_image(self, image, transform):
        image = transform.apply_image(image)
        image = torch.as_tensor(image, device=self.device)
        return image.permute(2, 0, 1).contiguous()

    def infer_prompt(self, batch_input, multimask_output = False):
        batched = []

        for input in batch_input:
            item = {}
            image = input.setdefault("image", None)
            assert image is not None
            input_points = input.setdefault("input_points", None)
            input_labels = input.setdefault("input_labels", None)
            input_boxes = input.setdefault("input_boxes", None)
            input_masks = input.setdefault("input_masks", None)

            prepare_image = self.prepare_image(image, self.resize_transform)

            item["image"] = prepare_image
            if input_points is not None:
                input_points = torch.as_tensor(input_points, device=self.device)
                item["point_coords"] = self.resize_transform.apply_coords_torch(input_points, image.shape[:2])
            if input_labels is not None:
                input_labels = torch.as_tensor(input_labels, device=self.device)
                item["point_labels"] = input_labels
            if input_boxes is not None:
                input_boxes = torch.as_tensor(input_boxes, device=self.device)
                item["boxes"] = self.resize_transform.apply_boxes_torch(input_boxes, image.shape[:2])
            if input_masks is not None:
                input_masks = torch.as_tensor(input_masks, device=self.device)
                item["mask_inputs"] = input_masks

            item['original_size'] = image.shape[:2]

            batched.append(item)

        batched_output = self.model(batched, multimask_output=multimask_output)

        return batched_output

    def infer_auto(self, image):
        masks = self.mask_generator.generate(image)
        return masks

def run_infer_prompt():
    device = torch.device("cpu")

    batch_input = []

    image = cv2.imread(os.path.join(ROOT, "demo", "truck.jpg"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_boxes = np.array([[425, 600, 700, 875]])
    input_points = np.array([[[575, 750]]])
    input_labels = np.array([[0]])
    item = {
        "image":image,
        "input_points":input_points,
        "input_labels":input_labels,
        "input_boxes":input_boxes,
        "input_masks":None,
    }
    batch_input.append(item)

    image = cv2.imread(os.path.join(ROOT, "demo", "groceries.jpg"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_boxes = np.array([
            [450, 170, 520, 350],
            [350, 190, 450, 350],
            [500, 170, 580, 350],
            [580, 170, 640, 350],
        ])
    item = {
        "image":image,
        "input_boxes":input_boxes,
    }
    batch_input.append(item)

    infer = Inference(device=device)

    """
    With `multimask_output=True` (the default setting), SAM outputs 3 masks,
    where `scores` gives the model's own estimation of the quality of these masks. 
    This setting is intended for ambiguous input prompts, 
    and helps the model disambiguate different objects consistent with the prompt.
    When `False`, it will return a single mask. For ambiguous prompts such as a single point,
    it is recommended to use `multimask_output=True` even if only a single mask is desired; 
    the best single mask can be chosen by picking the one with the highest score returned in `scores`.
    This will often result in a better mask.
    """
    batched_output = infer.infer_prompt(batch_input, multimask_output=False)
    print(batched_output)

    fig, ax = plt.subplots(1, 2, figsize=(20, 20))

    ax[0].imshow(batch_input[0]["image"])
    for mask in batched_output[0]['masks']:
        show_mask(mask.cpu().numpy(), ax[0], random_color=True)
    for box in batch_input[0]["input_boxes"]:
        show_box(box, ax[0])
    ax[0].axis('off')

    ax[1].imshow(batch_input[1]["image"])
    for mask in batched_output[1]['masks']:
        show_mask(mask.cpu().numpy(), ax[1], random_color=True)
    for box in batch_input[1]["input_boxes"]:
        show_box(box, ax[1])
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()

def run_info_auto():
    device = torch.device("cpu")

    image = cv2.imread(os.path.join(ROOT, "demo", "truck.jpg"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
    plt.show()


if __name__ == '__main__':
    run_infer_prompt()
    run_info_auto()