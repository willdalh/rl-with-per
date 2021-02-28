import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from datetime import datetime
import math

import sys
sys.path.append("../")

from utils import SumTree, PrioritizedReplayBuffer, ReplayBuffer


class Net(nn.Module):
    def __init__(self, input_size, output_size, ALPHA):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, output_size)

        self.optimizer = optim.Adam(self.parameters(), lr=ALPHA)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
    def forward(self, state):
        state = self.fc1(state)
        state = F.relu(state)
        state = self.fc2(state)
        state = F.relu(state)
        actions = self.fc3(state)
        return actions
