"""
Grad-CAM visualizations
"""

import os
import cv2
import torch
import numpy as np

from PIL import Image
from torchvision.transforms import v2

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((299, 299)),
    v2.ToDtype(torch.float32, scale=True)
])

def generate_gradcam(model, image_path, output_dir="static/gradcams"):
    """ generate GradCAM viz for a loaded CNN model
    returns filename of saved image"""

    model.eval()

    # last layer
    target_layers = [model.features[-3]]

    pil_img = Image.open(image_path).convert("RGB")

    rgb_img = np.array(
        pil_img.resize((299, 299))
    ).astype(np.float32) / 255.0
    
    input_tensor = transform(pil_img)
    input_tensor = input_tensor.unsqueeze(0)

    cam = GradCAM(model=model,target_layers=target_layers)

    grayscale_cam = cam(input_tensor=input_tensor)[0]

    # overlay heatmap on image
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # save
    os.makedirs(output_dir, exist_ok=True)

    og = os.path.basename(image_path)

    filename = f"gradcam_{og}"
    output_path = os.path.join(output_dir, filename)

    cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

    return filename