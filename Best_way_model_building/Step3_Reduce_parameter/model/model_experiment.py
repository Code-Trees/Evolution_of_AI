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
    def __init__(self,dropout):
        self.dropout = dropout
        self.conv1 = nn.Con
        