import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set default tensor type to float64
# torch.torch.set_default_dtype(torch.float64)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dropout = 0.01

        self.conv_layers = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=1, padding=0, dilation=1, bias=True),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ),  # input_size=28, output_size=26, receptive_field=3
            nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=0, dilation=1, bias=True),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ),  # input_size=26, output_size=24, receptive_field=5
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=0, dilation=1, bias=True),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            )  # input_size=24, output_size=22, receptive_field=7
        )

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1)  # input_size=22, output_size=11, receptive_field=8

        self.trans_layers = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=16, kernel_size=(1, 1), stride=1, padding=0, dilation=1, bias=True),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ),  # input_size=11, output_size=11, receptive_field=8
            nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=0, dilation=1, bias=True),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ),  # input_size=11, output_size=9, receptive_field=12
            nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=0, dilation=1, bias=True),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ),  # input_size=9, output_size=7, receptive_field=16
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=0, dilation=1, bias=True),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ),  # input_size=7, output_size=5, receptive_field=20
            nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1, padding=0, dilation=1, bias=True),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ),  # input_size=5, output_size=3, receptive_field=24
            nn.Conv2d(in_channels=256, out_channels=10, kernel_size=(3, 3), stride=1, padding=0, dilation=1, bias=True)  # input_size=3, output_size=1, receptive_field=28
        )

    def forward(self, x):
        x = self.conv_layers[0](x)
        x = self.conv_layers[1](x)
        x = self.conv_layers[2](x)

        x = self.pool1(x)

        x = self.trans_layers[0](x)
        x = self.trans_layers[1](x)
        x = self.trans_layers[2](x)
        x = self.trans_layers[3](x)
        x = self.trans_layers[4](x)
        x = self.trans_layers[5](x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)