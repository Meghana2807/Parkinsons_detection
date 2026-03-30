import torch
import torch.nn as nn


class CNNTransformer(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNTransformer, self).__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Classifier
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: [B, 3, 224, 224]
        x = self.cnn(x)               # [B, 64, H, W]
        x = x.mean(dim=[2, 3])        # Global average pooling → [B, 64]
        x = x.unsqueeze(1)            # [B, 1, 64]

        x = self.transformer(x)       # Transformer
        x = x.squeeze(1)              # [B, 64]

        return self.fc(x)
