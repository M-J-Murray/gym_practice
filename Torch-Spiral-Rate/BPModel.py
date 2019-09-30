#!/home/michael/anaconda3/envs/AIGym/bin/python3.6
import torch.nn as nn

class BPModel(nn.Module):

    def __init__(self, input_space, height, num_classes):
        super(BPModel, self).__init__()
        self.fc1 = nn.Linear(input_space, height)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(height, num_classes)

    def forward(self, out):
        out = self.relu(self.fc1(out))
        return self.fc2(out)