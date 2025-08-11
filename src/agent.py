import torch
import torch.nn as nn
import torch.nn.functional as F
from model import DQNNetwork
import torch.optim as optim
import random
import numpy as np
from collections import deque

CAPACITY = 100000
GAMMA = 0.99
TAU = 0.001
BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class ReplayBuffer:

    def __init__(self, capacity):

        self.buffer = deque(maxlen=capacity)

    def push(self, transition):

        # The transition should be a tuple: (state, action, reward, next_state, done)
        self.buffer.append(transition)

    def sample(self, batch_size):

        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):

        return len(self.buffer)


class Agent:
    
    def __init__(self, action_space_size, learning_rate=0.0001):

        self.action_space_size = action_space_size

        self.local_model = DQNNetwork(action_space_size).to(device)
        self.target_model = DQNNetwork(action_space_size).to(device)
        self.optimizer = optim.Adam(self.local_model.parameters(), lr=learning_rate)

        self.memory = ReplayBuffer(CAPACITY)
        self.time_step = 0

    def step(self, state, action, reward, next_state, done):

        self.memory.push((state, action, reward, next_state, done))

        if len(self.memory) > BATCH_SIZE and self.time_step % 4 == 0:
            batch = self.memory.sample(BATCH_SIZE)
            self.learn(batch, GAMMA)

        if self.time_step % 1000 == 0:
            self.target_model.load_state_dict(self.local_model.state_dict())

        self.time_step += 1


    def act(self, state, epsilon):
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
        self.local_model.eval()
        q_values = self.local_model(state)
        self.local_model.train()

        if random.random() > epsilon:
            action = q_values.argmax().item()
        else:
            action = random.randint(0, self.action_space_size - 1)
        return action
    

    def learn(self, experiences, gamma):

        states, actions, rewards, next_states, dones = experiences

        states = torch.from_numpy(np.array(states)).float().to(device)
        actions = torch.tensor(actions).long().to(device)
        if states.dim() == 3:
            states = states.unsqueeze(0)  # [1, C, H, W]
        if actions.dim() == 0:
            actions = actions.unsqueeze(0)  # [1]

        Q_expected = self.local_model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        rewards = torch.from_numpy(np.array(rewards)).float().to(device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(device)
        if next_states.dim() == 3:
            next_states = next_states.unsqueeze(0)

        dones = torch.from_numpy(np.array(dones)).float().to(device)

        Q_target_expected = rewards + (1 - dones) * gamma * self.target_model(next_states).max(1)[0].detach()
        Q_expected = self.local_model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = F.mse_loss(Q_expected, Q_target_expected)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update_target_network(self.local_model, self.target_model, tau=TAU)

    def soft_update_target_network(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)
