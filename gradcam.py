import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision.transforms import v2

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from model import CustomCNN

# from main.py
transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((299, 299)),
    v2.ToDtype(torch.float32, scale=True)
])

def generate_gradcam(model_path, image_path):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = CustomCNN()

    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    model.to(device)
    model.eval()

    # last layer
    target_layers = [model.conv2]

    # load image - change to greyscale?
    pil_img = Image.open(image_path).convert("RGB")

    rgb_img = np.array(
        pil_img.resize((299, 299))
    ) / 255.0

    input_tensor = transform(pil_img)

    input_tensor = input_tensor.unsqueeze(0).to(device)

    # gradcam
    cam = GradCAM(model=model, target_layers=target_layers)

    grayscale_cam = cam(input_tensor=input_tensor)[0]

    # overlay heatmap on image
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # save
    os.makedirs("gradcam_outputs", exist_ok=True)

    filename = os.path.basename(image_path)
    output_path = os.path.join("gradcam_outputs", f"gradcam_{filename}")

    cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

    print(f"Saved GradCAM to: {output_path}")

    # display
    plt.imshow(visualization)
    plt.axis("off")
    plt.show()

# test for this file only
if __name__ == "__main__":
    generate_gradcam(
        "data/custom.pth",
        "/Users/vickiwong/Desktop/cs4100/AI-Handwriting-Alzheimer-Diagnosis/data/annotations/TASK_02/T02_AD_001.png"
    )