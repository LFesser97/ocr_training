import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VGG_FeatureExtractor(nn.Module):
    def __init__(self, input_channel, output_channel=512):
        super(VGG_FeatureExtractor, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]  # [8, 16, 32, 64]
        self.ConvNet = nn.Sequential( #1x20x100
            nn.Conv2d(input_channel, self.output_channel[1], kernel_size=5, stride=1, padding=0), #16x16x96
            nn.MaxPool2d(2,2),  # 16x8x48
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 5, 1, 2), #32x8x48
            nn.MaxPool2d(2,2),  # 32x4x24
            nn.ReLU(True),
            nn.BatchNorm2d(self.output_channel[2], affine=False),
            nn.Dropout(p=0.2),
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1), #64x4x24
            nn.MaxPool2d(2, 2),  # 64x2x12
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0),
            nn.ReLU(True),  # hid*1x11
            nn.BatchNorm2d(self.output_channel[3],affine=False),
            nn.Dropout(p=0.2),
        )


class VGG_FeatureExtractor(nn.Module):
    def __init__(self, input_channel, output_channel=512):
        super(VGG_FeatureExtractor, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]  # [32, 64, 128, 256]
        self.ConvNet = nn.Sequential( #1x20x100
            nn.Conv2d(input_channel, self.output_channel[1], kernel_size=5, stride=1, padding=0), #32x16x96
            nn.MaxPool2d(2,2),  # 32x8x48
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            
            # add 7 conv layers to get an output of 256 x 1 x 11
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 5, 1, 2), #64x8x48
            nn.ReLU(True),
            nn.BatchNorm2d(self.output_channel[2], affine=False),
            nn.Dropout(p=0.2),
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1), #128x8x48
            nn.ReLU(True),
            nn.BatchNorm2d(self.output_channel[3], affine=False),
            nn.Dropout(p=0.2),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1), #128x8x48
            nn.ReLU(True),
            nn.BatchNorm2d(256, affine=False),
            nn.Dropout(p=0.2),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1), #128x8x48
            nn.ReLU(True),
            nn.BatchNorm2d(256, affine=False),
            nn.Dropout(p=0.2),
            nn.Conv2d(256, 256, 3, 1, 1), #256x4x24
            nn.ReLU(True),
            nn.BatchNorm2d(256, affine=False),
            nn.Dropout(p=0.2),
            nn.Conv2d(256, 256, 3, 1, 1), #256x4x24
            nn.MaxPool2d(2, 2),  # 256x2x12
            nn.ReLU(True),
            nn.BatchNorm2d(256, affine=False),
            nn.Dropout(p=0.2),
            nn.Conv2d(256, 256, 2, 1, 0), #256x1x11
            nn.ReLU(True),
            nn.BatchNorm2d(256, affine=False),
            nn.Dropout(p=0.2),
        )