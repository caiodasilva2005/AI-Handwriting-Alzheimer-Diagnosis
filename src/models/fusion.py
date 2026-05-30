import torch
import torch.nn as nn

class LateFusionModel(nn.Module):
    def __init__(self, cnn_feature_dim, mlp_input_dim, output_dim):
        super(LateFusionModel, self).__init__()
        
        # Stream 1: CNN Feature Extractor
        # Example using a placeholder CNN base
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Stream 2: MLP Processor
        self.mlp_stream = nn.Sequential(
            nn.Linear(mlp_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, cnn_feature_dim) # Match dimensions for concatenation
        )
        
        # Fusion and Final Prediction Layers
        # Assuming concatenation, combined dimension = cnn_dim * 2
        self.fusion_classifier = nn.Sequential(
            nn.Linear(cnn_feature_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, image_data, structured_data):
        # 1. Extract features from CNN
        cnn_features = self.cnn_backbone(image_data)
        
        # 2. Extract features from MLP
        mlp_features = self.mlp_stream(structured_data)
        
        # 3. Concatenate Features in Late Fusion
        fused_features = torch.cat((cnn_features, mlp_features), dim=1)
        
        # 4. Final Prediction
        output = self.fusion_classifier(fused_features)
        return output
