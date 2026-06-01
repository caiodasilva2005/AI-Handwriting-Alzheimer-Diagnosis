"""
Intermediate Fusion Model (feature-level fusion)

Combines the penultimate feature vectors of an already-trained CNN (handwriting
images) and KinematicMLP (kinematic features), then learns a single linear
combiner on top of the concatenated representation:

    feat_cnn = CNN.features -> gap -> flatten        # (B, 128)
    feat_mlp = MLP.network[:-2](kinematic)           # (B, 64)
    fused    = concat([feat_cnn, feat_mlp], dim=1)   # (B, 192)
    prob     = sigmoid(Linear(192, 1)(fused))        # P(Alzheimer)

Only the final combiner is trainable. Every layer of both base models is frozen
(requires_grad = False) and kept in eval() mode so dropout is disabled and the
extracted feature vectors are deterministic. During training, gradients flow
only into the combiner's weights.

Probability convention - the model emits P(class = 1 = Alzheimer) as shape (B, 1),
matching the BCELoss convention used to train the base models.
"""
import torch
import torch.nn as nn

from models.cnn import CNN
from models.mlp import KinematicMLP

# Dimensions of the penultimate feature vectors of each base model.
CNN_FEATURE_DIM = 128   # flattened GAP output, before CNN.fc
MLP_FEATURE_DIM = 64    # output of MLP.network[:-2], before final Linear(64, 1)


class IntermediateFusionModel(nn.Module):
    def __init__(self, cnn_path, mlp_path, mlp_input_size):
        super().__init__()

        # --- Load and freeze the CNN ---
        self._cnn = CNN()
        self._cnn.load_state_dict(torch.load(cnn_path, map_location="cpu"))

        # --- Load and freeze the MLP ---
        self._mlp = KinematicMLP(input_size=mlp_input_size)
        self._mlp.load_state_dict(torch.load(mlp_path, map_location="cpu"))

        # Freeze every parameter of both base models.
        for param in self._cnn.parameters():
            param.requires_grad = False
        for param in self._mlp.parameters():
            param.requires_grad = False

        # The only trainable layer: combines the two feature vectors.
        self.combiner = nn.Linear(CNN_FEATURE_DIM + MLP_FEATURE_DIM, 1)
        self.activation = nn.Sigmoid()

        # Keep base models in eval() (no dropout, fixed stats).
        self._cnn.eval()
        self._mlp.eval()

    def train(self, mode=True):
        """Switch combiner train/eval but always keep base models frozen in eval."""
        super().train(mode)
        self._cnn.eval()
        self._mlp.eval()
        return self

    def _cnn_features(self, image):
        # CNN.forward without its final classification head (self._cnn.fc).
        x = self._cnn.features(image)
        x = self._cnn.gap(x)
        return torch.flatten(x, 1)  # (B, 128)

    def _mlp_features(self, kinematic):
        # MLP.network without its final Linear(64, 1) + Sigmoid (last two modules).
        return self._mlp.network[:-2](kinematic)  # (B, 64)

    def forward(self, image, kinematic):
        # Extract frozen features without tracking gradients through base models.
        with torch.no_grad():
            cnn_feat = self._cnn_features(image)   # (B, 128)
            mlp_feat = self._mlp_features(kinematic)  # (B, 64)

        fused = torch.cat([cnn_feat, mlp_feat], dim=1)  # (B, 192)
        logit = self.combiner(fused)                    # (B, 1)
        return self.activation(logit)                   # P(Alzheimer), (B, 1)
