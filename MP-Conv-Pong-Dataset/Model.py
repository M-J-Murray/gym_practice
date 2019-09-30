import torch.nn as nn
import torch

class Model(nn.Module):

    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=5, stride=2),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(12, 24, kernel_size=5, stride=2),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=5, stride=2),
            nn.ReLU())
        self.fc = nn.Linear(24*2*5, num_classes)

    def forward(self, image, softmax=True):
        out = self.conv1(image)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.fc(out.view(out.size(0), -1))
        return out