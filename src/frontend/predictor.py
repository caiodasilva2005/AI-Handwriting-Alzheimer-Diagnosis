import os
import sys
import torch
import pandas as pd
import joblib
import numpy as np

from torchvision.io import decode_image, ImageReadMode
from torchvision.transforms import v2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(ROOT, "src"))

from models.cnn import CNN
from models.mlp import KinematicMLP
from models.fusion import IntermediateFusionModel

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

        self.mlp_selector = joblib.load("models/mlp_selector.pkl")
        self.mlp_scaler = joblib.load("models/mlp_scaler.pkl")

        # fusion model
        self.fusion = IntermediateFusionModel("models/cnn.pth", "models/mlp_fusion.pth", 18)
        self.fusion.load_state_dict(torch.load("models/fusion.pth", map_location=self.device))
        self.fusion.eval()

        self.fusion_scaler = joblib.load("models/fusion_scaler.pkl")

    def predict(self, image_path=None, csv_path=None):
        # image only - CNN
        if image_path and not csv_path:
            image = self._preprocess_image(image_path)

            with torch.no_grad():
                logits = self.cnn(image)

            probs = torch.softmax(logits, dim=1)
            prob = float(probs[0, 1])

            gradcam_file = generate_gradcam(self.cnn, image_path)
            prediction, confidence = self._get_prediction(prob)

            return {
                "model_used": "CNN",
                "prediction": prediction,
                "confidence": confidence,
                "gradcam": f"/static/gradcams/{gradcam_file}"
            }

        # kinematic only - MLP
        if csv_path and not image_path:
            X = self._preprocess_kinematic_csv(csv_path)

            with torch.no_grad():
                probs = self.mlp(X).squeeze()

            prob = float(probs.mean())
            prediction, confidence = self._get_prediction(prob)

            return {
                "model_used": "MLP",
                "prediction": prediction,
                "confidence": confidence,
                "gradcam": None
            }

        # both image and kinematic - fusion
        if image_path and csv_path:
            image_tensor = self._preprocess_image(image_path)
            kinematic_tensor = self._preprocess_fusion_kinematic(csv_path)

            with torch.no_grad():
                prob = float(self.fusion(image_tensor, kinematic_tensor).squeeze())

            gradcam_file = generate_gradcam(self.cnn, image_path)
            prediction, confidence = self._get_prediction(prob)

            return {
                "model_used": "Fusion",
                "prediction": prediction,
                "confidence": confidence,
                "gradcam": f"/static/gradcams/{gradcam_file}"
            }

        return {
            "error": "No input provided"
        }

    def _read_csv(self, csv_path):
        import pandas.errors

        try:
            df = pd.read_csv(csv_path)
        except pandas.errors.EmptyDataError:
            raise ValueError("The CSV file is empty.")
        except pandas.errors.ParserError:
            raise ValueError("The CSV file is malformatted and could not be parsed.")
        except UnicodeDecodeError:
            raise ValueError("The CSV file could not be read. Make sure it is a plain-text CSV.")
        except Exception as e:
            raise ValueError(f"The CSV file could not be read: {e}")

        if df.empty:
            raise ValueError("The CSV file contains no data rows.")

        return df

    def _preprocess_kinematic_csv(self, csv_path):
        df = self._clean_df(self._read_csv(csv_path))

        expected = getattr(self.mlp_selector, "feature_names_in_", None)
        if expected is not None:
            missing = [c for c in expected if c not in df.columns]
            if missing:
                raise ValueError(
                    f"The CSV is missing {len(missing)} expected DARWIN feature "
                    f"column(s) (e.g. {', '.join(map(str, missing[:3]))}). The "
                    f"standalone MLP needs the full DARWIN-format CSV."
                )
            df = df[list(expected)]

        try:
            X_selected = self.mlp_selector.transform(df)
            X_scaled = self.mlp_scaler.transform(X_selected)
        except ValueError:
            raise ValueError(
                "The CSV contains non-numeric or invalid values in its feature columns."
            )

        return torch.tensor(X_scaled, dtype=torch.float32)

    def _preprocess_image(self, image_path):
        transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((299, 299)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        try:
            image = decode_image(image_path, mode=ImageReadMode.RGB)
        except Exception:
            raise ValueError(
                "The image file could not be read. It may be corrupted or not a "
                "valid image."
            )

        image = transform(image)
        return image.unsqueeze(0)

    def _preprocess_fusion_kinematic(self, csv_path):
        df = self._clean_df(self._read_csv(csv_path))

        expected = getattr(self.fusion_scaler, "n_features_in_", 18)
        if df.shape[1] != expected:
            raise ValueError(
                f"For fusion (image + CSV), the CSV must contain exactly {expected} "
                f"single-task feature columns, but {df.shape[1]} were found."
            )

        try:
            X = df.values.astype("float32")
            X_scaled = self.fusion_scaler.transform(X)
        except ValueError:
            raise ValueError(
                "The CSV contains non-numeric or invalid values in its feature columns."
            )

        return torch.tensor(X_scaled, dtype=torch.float32)

    def _get_prediction(self, prob):
        if prob >= 0.5:
            return "Alzheimer", prob
        return "Control", 1 - prob

    def _clean_df(self, df):
        return df.drop(columns=["ID", "class"], errors="ignore")