import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from datetime import datetime


import sys
sys.path.append("../")
from utils import SumTree, PrioritizedReplayBuffer, ReplayBuffer


class Net(nn.Module):
    def __init__(self, input_size, output_size, ALPHA):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 32, kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=2)
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, output_size)
        
        self.optimizer = optim.Adam(self.parameters(), lr=ALPHA)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
    def forward(self, state):
        state = state.reshape(-1, 1, 8, 8)
        state = F.relu(self.conv1(state))
        state = F.relu(self.conv2(state))
        state = F.relu(self.conv3(state))
        state = state.reshape(-1, 128)
        state = F.relu(self.fc1(state))
        actions = self.fc2(state)
        return actions


