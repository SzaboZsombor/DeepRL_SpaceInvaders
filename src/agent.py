import torch
import torch.nn.functional as F
from model import DuelingDQNNetwork
import torch.optim as optim
import random
import numpy as np
from collections import deque


CAPACITY = 100000
GAMMA = 0.99
TAU = 0.001
BATCH_SIZE = 64
LEARNING_RATE = 0.00025

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

    def __init__(self, action_space_size, learning_rate=LEARNING_RATE, gamma=GAMMA, tau=TAU, batch_size=BATCH_SIZE):

        self.action_space_size = action_space_size

        self.local_model = DuelingDQNNetwork(action_space_size).to(device)
        self.target_model = DuelingDQNNetwork(action_space_size).to(device)
        self.optimizer = optim.Adam(self.local_model.parameters(), lr=learning_rate)

        self.scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
        self.optimizer.zero_grad()

        self.memory = ReplayBuffer(CAPACITY)
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
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
        self.local_model.eval()
        q_values = self.local_model(state)
        self.local_model.train()

        if random.random() > epsilon:
            action = q_values.argmax().item()
        else:
            action = random.randint(0, self.action_space_size - 1)
        return action
    

    def learn(self, experiences):

        states, actions, rewards, next_states, dones = experiences

        states = torch.from_numpy(np.array(states)).float().to(device)
        actions = torch.tensor(actions).long().to(device)
        if states.dim() == 3:
            states = states.unsqueeze(0)  # [1, C, H, W]
        if actions.dim() == 0:
            actions = actions.unsqueeze(0)  # [1]

        rewards = torch.from_numpy(np.array(rewards)).float().to(device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(device)
        if next_states.dim() == 3:
            next_states = next_states.unsqueeze(0)

        dones = torch.from_numpy(np.array(dones)).float().to(device)

        with torch.no_grad():
            best_actions_next = self.local_model(next_states).argmax(1).unsqueeze(1)
            Q_targets_next = self.target_model(next_states).gather(1, best_actions_next).squeeze(1)
            Q_target_expected = rewards + (1 - dones) * self.gamma * Q_targets_next

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            Q_expected = self.local_model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            loss = F.mse_loss(Q_expected, Q_target_expected)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.soft_update_target_network(self.local_model, self.target_model)

    def soft_update_target_network(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1 - self.tau) * target_param.data)
