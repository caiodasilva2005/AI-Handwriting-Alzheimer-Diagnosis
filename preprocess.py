from torchvision.io import decode_image, ImageReadMode
from torchvision.transforms import v2
import torch

transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((299, 299)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def preprocess_image(image_path):
    image = decode_image(image_path, mode=ImageReadMode.RGB)
    image = transform(image)
    return image.unsqueeze(0)