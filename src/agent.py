import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

from model import DuelingDQNNetwork
from src.replay import PrioritizedReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class Agent:

    def __init__(self, action_space_size, learning_rate, gamma, tau, batch_size, capacity):

        self.action_space_size = action_space_size
        self.device = device

        self.local_model = DuelingDQNNetwork(action_space_size).to(self.device)
        self.target_model = DuelingDQNNetwork(action_space_size).to(self.device)
        self.optimizer = optim.Adam(self.local_model.parameters(), lr=learning_rate)

        self.scaler = torch.amp.GradScaler(enabled=(self.device.type == 'cuda'))
        self.optimizer.zero_grad()

        self.memory = PrioritizedReplayBuffer(capacity)
        self.time_step = 0
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

    def step(self, state, action, reward, next_state, done):

        self.memory.push((state, action, reward, next_state, done))

        if len(self.memory) > 5000 and self.time_step % 4 == 0:
            batch = self.memory.sample(self.batch_size)
            self.learn(batch)

        self.time_step += 1


    def act(self, state, epsilon):
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        self.local_model.eval()
        q_values = self.local_model(state)
        self.local_model.train()

        if random.random() > epsilon:
            action = q_values.argmax().item()
        else:
            action = random.randint(0, self.action_space_size - 1)
        return action
    

    def learn(self, experiences):

        (states, actions, rewards, next_states, dones), indices, is_weights = experiences

        states = torch.from_numpy(np.array(states)).float().to(self.device)
        actions = torch.tensor(actions).long().unsqueeze(1).to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.array(dones)).float().to(self.device)
        is_weights = torch.from_numpy(is_weights).float().to(self.device)

        with torch.no_grad():
            best_actions_next = self.local_model(next_states).argmax(1).unsqueeze(1)
            Q_targets_next = self.target_model(next_states).gather(1, best_actions_next).squeeze(1)
            Q_target_expected = rewards + (1 - dones) * self.gamma * Q_targets_next

        with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
            Q_expected = self.local_model(states).gather(1, actions).squeeze(1)

            td_errors = (Q_target_expected - Q_expected).detach()
            loss = (is_weights * F.mse_loss(Q_expected, Q_target_expected, reduction='none')).mean()

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.memory.update_priorities(indices, td_errors.cpu().numpy())

        self.soft_update_target_network(self.local_model, self.target_model)

    def soft_update_target_network(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1 - self.tau) * target_param.data)
