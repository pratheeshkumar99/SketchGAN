import torch
import torch.nn as nn

# Unet based generator which takes a sketch as input and generates a colored image
# It consists of an encoder, a bottleneck, and a decoder
# The encoder downsamples the input image and the decoder upsamples it
# The bottleneck is the bridge between the encoder and decoder
# The generator is trained to generate realistic images from sketches

# U-Net Generator
class UNetGenerator(nn.Module):
    def __init__(self, dropout_value=0.5):
        super(UNetGenerator, self).__init__()

        # Define downsampling (encoder) blocks
        self.down1 = self.conv_block(3, 64, normalization=False)
        self.down2 = self.conv_block(64, 128)
        self.down3 = self.conv_block(128, 256)
        self.down4 = self.conv_block(256, 512)
        self.down5 = self.conv_block(512, 512)
        self.down6 = self.conv_block(512, 512)
        self.down7 = self.conv_block(512, 512, normalization=False)

        # Define upsampling (decoder) blocks with skip connections
        self.up1 = self.upconv_block(512, 512, dropout_value=dropout_value)
        self.up2 = self.upconv_block(1024, 512, dropout_value=dropout_value)
        self.up3 = self.upconv_block(1024, 512, dropout_value=dropout_value)
        self.up4 = self.upconv_block(1024, 256)
        self.up5 = self.upconv_block(512, 128)
        self.up6 = self.upconv_block(256, 64)

        # Final output layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def conv_block(self, in_channels, out_channels, normalization=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2)
        ]
        if normalization:
            layers.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(*layers)

    def upconv_block(self, in_channels, out_channels, dropout_value=0.0):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(out_channels)
        ]
        if dropout_value > 0.0:
            layers.append(nn.Dropout(dropout_value))  # Apply the dropout value here
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder (downsampling)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        # Decoder (upsampling) with skip connections
        u1 = self.up1(d7)
        u2 = self.up2(torch.cat([u1, d6], 1))
        u3 = self.up3(torch.cat([u2, d5], 1))
        u4 = self.up4(torch.cat([u3, d4], 1))
        u5 = self.up5(torch.cat([u4, d3], 1))
        u6 = self.up6(torch.cat([u5, d2], 1))

        # Final layer to output image
        return self.final(torch.cat([u6, d1], 1))