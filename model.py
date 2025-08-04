import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, embedding_dim=10):
        super().__init__()
        # Encoder: 1x400x400 -> ... -> 3x3
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),   # 400 -> 200
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 200 -> 100
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 100 -> 50
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),# 50 -> 25
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 512, 4, stride=2, padding=1),# 25 -> 12
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(512, 512, 4, stride=2, padding=1),# 12 -> 6
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(512, 512, 4, stride=2, padding=1),# 6 -> 3
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )

        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(512 * 3 * 3, embedding_dim)
        self.fc_dec = nn.Linear(embedding_dim, 512 * 3 * 3)

        # Decoder: 3x3 -> ... -> 384x384
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1),  # 3 -> 6
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1),  # 6 -> 12
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # 12 -> 24
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 24 -> 48
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 48 -> 96
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # 96 -> 192
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),    # 192 -> 384
            nn.BatchNorm2d(16),
            nn.ReLU(True),
        )

        # 最後調整到 400×400
        self.final_conv = nn.Conv2d(16, 1, 3, padding=1)  # 384 -> 384
        self.upsample = nn.Upsample(size=(400, 400), mode='bilinear', align_corners=False)

    def encode(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        embedding = self.fc_enc(x)
        return embedding

    def decode(self, embedding):
        x = self.fc_dec(embedding)
        x = x.view(-1, 512, 3, 3)
        x = self.decoder(x)            # [B, 16, 384, 384]
        x = self.final_conv(x)         # [B, 1, 384, 384]
        x = self.upsample(x)           # [B, 1, 400, 400]
        return x

    def forward(self, x):
        embedding = self.encode(x)
        out = self.decode(embedding)
        out = torch.sigmoid(out)       # normalize 輸出
        return out, embedding