import numpy as np
import sys
sys.path.append("..")  # Add the parent directory to the path

import agent
import environment
from training.plot import plot_training_progress

env = environment.SpaceInvadersEnv()
agent = agent.Agent(action_space_size=env.action_space.n)



def get_eps(episode, decay_rate=0.99, min_eps=0.1):
    return max(min_eps, decay_rate ** episode)

def train_agent(episodes=1000, max_steps=10000):
    scores = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        for t in range(max_steps):
            action = agent.act(state, epsilon=get_eps(episode))
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        scores.append(total_reward)
        print(f"Episode {episode + 1}/{episodes} - Total Reward: {total_reward}")

    return np.array(scores)


def main():
    print("Starting training...")
    scores = train_agent()
    plot_training_progress(scores)
