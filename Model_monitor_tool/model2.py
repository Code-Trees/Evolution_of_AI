import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=8,kernel_size=3,padding=0) #26
        self.bath1 = nn.BatchNorm2d(num_features=8)
        
        self.conv2 = nn.Conv2d(in_channels=8,out_channels=8,kernel_size=3,padding=0) #24 
        self.bath2 = nn.BatchNorm2d(num_features=8)
        
        self.conv3 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,padding=0) #22 
        self.bath3 = nn.BatchNorm2d(num_features=16)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2) #11
        
        
        self.conv4 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,padding=0) #9 
        self.bath4 = nn.BatchNorm2d(num_features=16)
        
        self.conv5 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=0) #7
        self.bath5 = nn.BatchNorm2d(num_features=32)
        
        self.conv6 = nn.Conv2d(in_channels=32,out_channels=16,kernel_size=1,padding=0) #7
        self.bath6 = nn.BatchNorm2d(num_features=16)
        
        self.conv7 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,padding=0)  #5
        self.bath7 = nn.BatchNorm2d(num_features=16)
        
        self.conv8 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=0) #3
        self.bath8 = nn.BatchNorm2d(num_features=32)
        
        self.conv9 = nn.Conv2d(in_channels=32,out_channels=10,kernel_size=3,padding=0) #1
                
    def forward(self, x):
        x = self.bath1(F.relu(self.conv1(x)))
        x = self.bath2(F.relu(self.conv2(x)))
        x = self.pool1(self.bath3(F.relu(self.conv3(x))))
        x = self.bath4(F.relu(self.conv4(x)))
        x = self.bath5(F.relu(self.conv5(x)))
        x = self.bath6(F.relu(self.conv6(x)))
        x = self.bath7(F.relu(self.conv7(x)))
        x = self.bath8(F.relu(self.conv8(x)))
        x = self.conv9(x)
        x = x.view(-1,10)
        return F.log_softmax(x)