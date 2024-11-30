import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, conv1_channels=16, conv2_channels=32, conv3_channels=64):
        super(Net, self).__init__()
        self.conv1_channels = conv1_channels
        self.conv2_channels = conv2_channels
        self.conv3_channels = conv3_channels
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv1_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=conv1_channels)
        self.conv2 = nn.Conv2d(in_channels=conv1_channels, out_channels=conv2_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=conv2_channels)
        self.conv3 = nn.Conv2d(in_channels=conv2_channels, out_channels=conv3_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=conv3_channels)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(7 * 7 * conv3_channels, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 7 * 7 * self.conv3_channels)
        x = self.fc1(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)