"""
Late Fusion Model (decision-level / output fusion)

Combines the probability outputs of an already-trained CNN (handwriting images)
and KinematicMLP (kinematic features) via a fixed weighted average:

    fused_prob = w * P_cnn(Alzheimer) + (1 - w) * P_mlp(Alzheimer)

Both base models are frozen; the combiner has no trainable parameters. This means
no subject-level mapping between the image dataset and the kinematic CSV is needed
to construct or run the model: at inference a single subject supplies both an image
and a kinematic feature vector, and their two probabilities are blended.

Probability convention - both streams emit P(class = 1 = Alzheimer) as shape (B, 1):
  - CNN returns (B, 2) logits  -> softmax(logits, dim=1)[:, 1:2]
  - KinematicMLP returns (B, 1) sigmoid prob (already P(Alzheimer))
"""
import torch
import torch.nn as nn

from models.cnn import CNN
from models.mlp import KinematicMLP


class LateFusionModel(nn.Module):
    def __init__(self, cnn_path, mlp_path, mlp_input_size, cnn_weight=0.5):
        super().__init__()
        if not 0.0 <= cnn_weight <= 1.0:
            raise ValueError(f"cnn_weight must be in [0, 1], got {cnn_weight}")

        cnn = CNN()
        cnn.load_state_dict(torch.load(cnn_path, map_location="cpu"))

        mlp = KinematicMLP(input_size=mlp_input_size)
        mlp.load_state_dict(torch.load(mlp_path, map_location="cpu"))
        
        self.cnn = cnn
        self.mlp = mlp
        self.cnn_weight = cnn_weight

        # Late fusion combines fixed predictions: freeze both base models.
        for param in self.cnn.parameters():
            param.requires_grad = False
        for param in self.mlp.parameters():
            param.requires_grad = False
        self.cnn.eval()
        self.mlp.eval()

    def _cnnProbAlzheimer(self, image_data):
        """P(Alzheimer) from the CNN logits, shape (B, 1)."""
        logits = self.cnn(image_data)
        return torch.softmax(logits, dim=1)[:, 1:2]

    def _mlpProbAlzheimer(self, structured_data):
        """P(Alzheimer) from the MLP, shape (B, 1) (already a sigmoid prob)."""
        return self.mlp(structured_data)

    @torch.no_grad()
    def forward(self, image_data, structured_data):
        """Weighted-average fused probability, shape (B, 1)."""
        p_cnn = self._cnnProbAlzheimer(image_data)
        p_mlp = self._mlpProbAlzheimer(structured_data)
        return self.cnn_weight * p_cnn + (1.0 - self.cnn_weight) * p_mlp

