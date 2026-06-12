"""
Intermediate Fusion Model (feature-level fusion)
"""
import torch
import torch.nn as nn

from models.cnn import CNN
from models.mlp import KinematicMLP

CNN_FEATURE_DIM = 128   
MLP_FEATURE_DIM = 64    

class IntermediateFusionModel(nn.Module):
    def __init__(self, cnn_path, mlp_path, mlp_input_size):
        super().__init__()

        self._cnn = CNN()
        self._cnn.load_state_dict(torch.load(cnn_path, map_location="cpu"))

        self._mlp = KinematicMLP(input_size=mlp_input_size)
        self._mlp.load_state_dict(torch.load(mlp_path, map_location="cpu"))

        # This freezes paramters of the base models so only the combiner is trainable
        for param in self._cnn.parameters():
            param.requires_grad = False
        for param in self._mlp.parameters():
            param.requires_grad = False

        # The only trainable layer is the combiner that takes concatenated features and outputs a probability with signmoid
        self.combiner = nn.Linear(CNN_FEATURE_DIM + MLP_FEATURE_DIM, 1)
        self.activation = nn.Sigmoid()

        self._cnn.eval()
        self._mlp.eval()

    def train(self, mode=True):
        super().train(mode)
        self._cnn.eval()
        self._mlp.eval()
        return self

    def _cnn_features(self, image):
        # forward CNN without its final classification layers
        x = self._cnn.features(image)
        x = self._cnn.gap(x)
        return torch.flatten(x, 1)  # (B, 128)

    def _mlp_features(self, kinematic):
        # forward MLP without its final classification layers
        return self._mlp.network[:-2](kinematic)  # (B, 64)

    def extract_features(self, image, kinematic):
        # concatenate the frozen features from both models (B, 192)
        with torch.no_grad():
            cnn_feat = self._cnn_features(image)     
            mlp_feat = self._mlp_features(kinematic)  
        return torch.cat([cnn_feat, mlp_feat], dim=1)  # (B, 192)

    def classify(self, fused):
        # runs the trainable combiner layer on the fused features to get a probability of Alzheimer (B, 1)
        return self.activation(self.combiner(fused))

    def forward(self, image, kinematic):
        # Extracts features from both models, concatenates them, and classifies
        fused = self.extract_features(image, kinematic)  # (B, 192)
        return self.classify(fused)                      # P(Alzheimer), (B, 1)
