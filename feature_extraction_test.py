import torch.nn as nn
import torch.nn.functional as F

dropout_rate = 0.1


class VGG_FeatureExtractor_4(nn.Module):
    """
    A 4-layer VGG feature extractor, with input dimensions
    1 x 20 x 100 and output dimensions 64 x 1 x 7
    """
    def __init__(self, input_channel, output_channel=512):
        super(VGG_FeatureExtractor_4, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]  # [8, 16, 32, 64]

        self.ConvNet = nn.Sequential( # 1 x 20 x 100
            # the first conv layer
            nn.Conv2d(input_channel, self.output_channel[0], kernel_size=3, stride=1, padding=1),  # 8 x 20 x 100
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8 x 10 x 50
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),

            # the second conv layer
            nn.Conv2d(self.output_channel[0], self.output_channel[1], kernel_size=3, stride=1, padding=1),  # 16 x 10 x 50
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16 x 5 x 25
            nn.ReLU(True),
            nn.BatchNorm2d(self.output_channel[1]),

            # the third conv layer
            nn.Conv2d(self.output_channel[1], self.output_channel[2], kernel_size=3, stride=1, padding=1),  # 32 x 5 x 25
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 x 2 x 12
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),

            # the fourth conv layer
            nn.Conv2d(self.output_channel[2], self.output_channel[3], kernel_size=3, stride=1, padding=1),  # 64 x 2 x 12
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64 x 1 x 6
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, input):
        return self.ConvNet(input)