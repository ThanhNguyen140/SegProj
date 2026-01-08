import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_p)
        )
    def forward(self, x):
        return self.double_conv(x)


class UNet3D(nn.Module):
    def __init__(
        self, in_channels=1, out_channels=1, init_features=64,
    ):
        super().__init__()
        features = init_features

        # Encoder
        self.enc1 = DoubleConv(in_channels, features)
        self.pool1 =  nn.MaxPool3d(2)
        self.enc2 = DoubleConv(features, features * 2, dropout_p = 0.2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = DoubleConv(features * 2, features * 4, dropout_p = 0.2)
        self.pool3 = nn.MaxPool3d(2)
        # Bottleneck
        self.bottleneck = DoubleConv(features * 4, features * 8, dropout_p = 0.2)

        # Decoder
        #self.up4 = nn.ConvTranspose3d(
        #    features * 16, features * 8, kernel_size=2, stride=2
        #)
        #self.dec4 = DoubleConv(features * 16, features * 8)
        self.up3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.dec3 = DoubleConv(features * 8, features * 4, dropout_p = 0.2)
        self.up2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.dec2 = DoubleConv(features * 4, features * 2, dropout_p = 0.2)
        self.up1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.dec1 = DoubleConv(features * 2, features)

        # Output
        self.final_conv = nn.Conv3d(features, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool3(enc3))

        # Decoder
        #dec4 = self.up4(bottleneck)
        #dec4 = torch.cat((dec4, enc4), dim=1)
        #dec4 = self.dec4(dec4)

        dec3 = self.up3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.up2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.up1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        out = self.final_conv(dec1)
        return out
