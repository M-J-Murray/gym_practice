#!/home/michael/anaconda3/envs/AIGym/bin/python3.6
import torch.nn as nn
import torch

class RateModel(nn.Module):

    def __init__(self, input_space, height, num_classes):
        super(RateModel, self).__init__()
        self.W1 = nn.Linear(input_space, height, bias=False)
        self.h1 = nn.Parameter(torch.zeros(1, input_space))
        self.R1 = nn.Linear(input_space, height, bias=False)
        self.R1.weight.data.fill_(0)
        self.v1 = nn.Parameter(torch.zeros(1, height))

        self.relu = nn.ReLU()
        
        self.W2 = nn.Linear(height, num_classes, bias=False)
        self.h2 = nn.Parameter(torch.zeros(1, height))
        self.R2 = nn.Linear(height, num_classes, bias=False)
        self.R2.weight.data.fill_(0)
        self.v2 = nn.Parameter(torch.zeros(1, num_classes))

    def forward(self, out):
        out = self.relu(self.v1 + self.W1(out) + self.R1(out+self.h1)**2)
        out = self.v2 + self.W2(out) + self.R2(out+self.h2)**2
        return out