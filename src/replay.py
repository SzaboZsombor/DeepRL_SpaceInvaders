import numpy as np
import random


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.size = 0


    def add(self, priority, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
        if self.size < self.capacity:
            self.size += 1


    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change


    def get_leaf(self, value):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if value <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    value -= self.tree[left_child_idx]
                    parent_idx = right_child_idx
        
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]


    @property
    def total_priority(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=0.01):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon


    def push(self, transition):
        max_priority = np.max(self.tree.tree[-self.capacity:]) if self.tree.size > 0 else 1.0
        self.tree.add(max_priority, transition)


    def sample(self, batch_size):
        batch = []
        indices = []
        weights = np.empty(batch_size, dtype=np.float32)
        segment = self.tree.total_priority / batch_size
        
        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            value = random.uniform(a, b)
            tree_idx, priority, data = self.tree.get_leaf(value)
            
            sampling_prob = priority / self.tree.total_priority
            weight = (self.tree.size * sampling_prob) ** -self.beta
            weights[i] = weight
            
            indices.append(tree_idx)
            batch.append(data)
        
        weights /= weights.max()
        
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones)), indices, weights


    def update_priorities(self, tree_indices, td_errors):
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        for idx, p in zip(tree_indices, priorities):
            self.tree.update(idx, p)


    def __len__(self):
        return self.tree.size