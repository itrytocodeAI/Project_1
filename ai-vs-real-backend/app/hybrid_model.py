import torch
import torch.nn as nn

# Hyperparameters (must match training settings)
PATCH_SIZE = 4
PROJECTION_DIM = 64

class HybridPatchExtractor(nn.Module):
    def __init__(self, in_channels=3, out_channels=PROJECTION_DIM):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        x = self.cnn(x)
        patches = x.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
        patches = patches.contiguous().view(x.size(0), PROJECTION_DIM, -1).transpose(1, 2)
        return patches

class TransformerBlock(nn.Module):
    def __init__(self, dim=PROJECTION_DIM, heads=4, mlp_dim=128):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class HybridViTCNNClassifier(nn.Module):
    def __init__(self, num_classes=1, num_transformer_layers=4):
        super().__init__()
        self.patch_extractor = HybridPatchExtractor()
        self.transformer = nn.Sequential(*[TransformerBlock() for _ in range(num_transformer_layers)])
        self.classifier = nn.Sequential(
            nn.LayerNorm(PROJECTION_DIM),
            nn.Linear(PROJECTION_DIM, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.patch_extractor(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)