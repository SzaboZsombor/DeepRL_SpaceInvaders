import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_model_path


class DQNNetwork(nn.Module):

    def __init__(self, action_space_size):
        nn.Module.__init__(self)
        torch.random.manual_seed(0)

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 9 * 9, 512)
        self.fc2 = nn.Linear(512, action_space_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def save_model_weights(self, file_name):
        file_path = get_model_path(file_name)
        
        torch.save(self.state_dict(), file_path)
        print(f"Model weights saved to {file_path}")

    def load_model_weights(self, file_name):
        file_path = get_model_path(file_name)

        if os.path.exists(file_path):
            self.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))
            print(f"Model weights loaded successfully from {file_path}")
        else:
            print(f"Model weights file {file_path} does not exist.")