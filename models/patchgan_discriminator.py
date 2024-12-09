import torch
import torch.nn as nn


# The PathGAN discriminator model is trained to classify whether the pair of images (sketch and image) are real pairs or fake pairs

class PatchGANDiscriminator(nn.Module):
    def __init__(self, dropout_value=0.5):
        super(PatchGANDiscriminator, self).__init__()

        self.model = nn.Sequential(
            self.conv_block(6, 64, normalization=False, dropout_value=0.0),  # First layer doesn't usually need dropout
            self.conv_block(64, 128, dropout_value=dropout_value),
            self.conv_block(128, 256, dropout_value=dropout_value),
            self.conv_block(256, 512, dropout_value=dropout_value),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)  # Output is a single patch score
        )

    def conv_block(self, in_channels, out_channels, normalization=True, dropout_value=0.0):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2)
        ]
        if normalization:
            layers.append(nn.BatchNorm2d(out_channels))
        if dropout_value > 0.0:
            layers.append(nn.Dropout(dropout_value))  # Apply dropout based on the passed value
        return nn.Sequential(*layers)

    def forward(self, img_input, sketch_input):
        # Concatenate sketch and generated image along the channel dimension
        x = torch.cat([img_input, sketch_input], 1)
        return self.model(x)