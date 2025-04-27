import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)      # 32→28
        self.pool1 = nn.AvgPool2d(2, stride=2)            # 28→14
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)     # 14→10
        self.pool2 = nn.AvgPool2d(2, stride=2)            # 10→5
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)   # 5→1
        self.fc1   = nn.Linear(120, 84)
        self.fc2   = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = self.pool1(x)
        x = F.tanh(self.conv2(x))
        x = self.pool2(x)
        x = F.tanh(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x
