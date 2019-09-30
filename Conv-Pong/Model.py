import torch.nn as nn


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.model = {
            "layer1": nn.Sequential(
                nn.Conv2d(1, 12, kernel_size=5, padding=1),
                nn.BatchNorm2d(4),
                nn.MaxPool2d(2)),
            "layer2": nn.Sequential(
                nn.Conv2d(12, 24, kernel_size=5, padding=1),
                nn.BatchNorm2d(4),
                nn.MaxPool2d(2)),
            "layer3": nn.Linear(80*80*24,2)}

    def forward(self, *input):
        out = input
        for layer in self.model.keys():
            out = self.model[layer](out)
        return out