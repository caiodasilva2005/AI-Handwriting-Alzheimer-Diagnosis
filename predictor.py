import os
import sys
import torch
import pandas as pd
import joblib
import numpy as np

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

        self.mlpselector = joblib.load("models/mlp_selector.pkl")
        self.mlpscaler = joblib.load("models/mlp_scaler.pkl")

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
        if csv_path and not image_path:
            X = self._preprocess_kinematic_csv(csv_path)

            with torch.no_grad():
                probs = self.mlp(X).squeeze()

            prob = float(probs.mean())
            
            if prob >= 0.5:
                prediction = "Alzheimer"
                confidence = prob
            else:
                prediction = "Control"
                confidence = 1 - prob

            return {
                "model_used": "MLP",
                "prediction": prediction,
                "confidence": confidence,
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

    def _preprocess_kinematic_csv(self, csv_path):
        df = pd.read_csv(csv_path)

        if "ID" in df.columns:
            df = df.drop(columns=["ID"])

        if "class" in df.columns:
            df = df.drop(columns=["class"])

        X_selected = self.mlpselector.transform(df)
        X_scaled = self.mlpscaler.transform(X_selected)

        return torch.tensor(X_scaled, dtype=torch.float32)