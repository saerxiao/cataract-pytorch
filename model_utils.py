import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_relu(nn.Module):
    def __init__(self, in_ch, out_ch, filter_size, stride=1, padding=0):
        super(conv_relu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, filter_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

