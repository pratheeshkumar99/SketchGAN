import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    def __init__(self):
        super(PatchGANDiscriminator, self).__init__()

        self.model = nn.Sequential(
            self.conv_block(6, 64, normalization=False),  # 6 channels because the input is concatenated (image + sketch)
            self.conv_block(64, 128),
            self.conv_block(128, 256),
            self.conv_block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)  # Output is a single patch score
        )

    def conv_block(self, in_channels, out_channels, normalization=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2)
        ]
        if normalization:
            layers.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(*layers)

    def forward(self, img_input, sketch_input):
        # Concatenate sketch and generated image along the channel dimension
        x = torch.cat([img_input, sketch_input], 1)
        return self.model(x)