"""
Grad-CAM visualizations
"""

import os
import cv2
import torch
import numpy as np

from torchvision.transforms import v2
from torchvision.io import decode_image, ImageReadMode

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def generate_gradcam(model, image_path, output_dir="static/gradcams"):
    """ generate GradCAM viz for a loaded CNN model
    returns filename of saved image"""

    model.eval()

    # last layer
    target_layers = [model.features[-3]]

    image = decode_image(image_path, mode=ImageReadMode.RGB)
    rgb_img = (v2.Resize((299, 299))(image).permute(1, 2, 0).float().numpy() / 255.0)

    transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((299, 299)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    image = decode_image(image_path, mode=ImageReadMode.RGB)
    image = transform(image)

    input_tensor = image.unsqueeze(0)

    cam = GradCAM(model=model,target_layers=target_layers)

    grayscale_cam = cam(input_tensor=input_tensor)[0]

    # overlay heatmap on image
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # save
    os.makedirs(output_dir, exist_ok=True)

    filename = f"gradcam_{os.path.basename(image_path)}"
    output_path = os.path.join(output_dir, filename)

    cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

    return filename