import os
import sys
import torch
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, "src"))

from models.cnn import CNN
from models.mlp import KinematicMLP
from models.fusion import IntermediateFusionModel

from preprocess import preprocess_image
from evaluation.gradcam import generate_gradcam


class Predictor:
    def __init__(self):
        self.device = "cpu"

        # standalone cnn
        self.cnn = CNN()
        self.cnn.load_state_dict(torch.load("models/cnn.pth", map_location=self.device))
        self.cnn.eval()

        # standalone mlp (30)
        self.mlp = KinematicMLP(input_size=30)
        self.mlp.load_state_dict(torch.load("models/mlp.pth", map_location=self.device))
        self.mlp.eval()

        # fusion model
        self.fusion = IntermediateFusionModel("models/cnn.pth", "models/mlp_fusion.pth", 18)
        self.fusion.load_state_dict(torch.load("models/fusion.pth", map_location=self.device))
        self.fusion.eval()

    def predict(self, image_path=None, csv_path=None):
        # image only - CNN
        if image_path and not csv_path:
            image = preprocess_image(image_path)
            logits = self.cnn(image)
            probs = torch.softmax(logits, dim=1)
            prob = float(probs[0, 1])
            gradcam_file = generate_gradcam(self.cnn, image_path)

            return {
                "model_used": "CNN",
                "prediction": "Alzheimer" if prob >= 0.5 else "Control",
                "confidence": prob,
                "gradcam": f"/static/gradcams/{gradcam_file}"
            }

        # kinematic only - MLP
        # fix
        if csv_path and not image_path:
            return {
                "model_used": "MLP",
                "prediction": "pass",
                "confidence": 0.0,
                "gradcam": None
            }

        # both image and kinematic - fusion
        # fix
        if image_path and csv_path:
            return {
                "model_used": "Fusion",
                "prediction": "pass",
                "confidence": 0.0,
                "gradcam": f"/static/gradcams/{gradcam_file}"
            }

        return {
            "error": "No input provided"
        }