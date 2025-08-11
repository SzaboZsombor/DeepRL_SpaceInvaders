import numpy as np
import os
import sys

# Add current directory to path so we can import utils
sys.path.insert(0, os.path.dirname(__file__))
import utils

import agent
import environment
import plot

env = environment.SpaceInvadersEnv()
agent = agent.Agent(action_space_size=env.action_space.n)



def get_eps(episode, decay_rate=0.99, min_eps=0.1):
    return max(min_eps, decay_rate ** episode)

def train_agent(episodes=1000, max_steps=10000):
    scores = []
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        for t in range(max_steps):
            action = agent.act(state, epsilon=get_eps(episode))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break

        scores.append(total_reward)
        print(f"Episode {episode + 1}/{episodes} - Total Reward: {total_reward}")

    return np.array(scores)


def main():
    print("Starting training...")
    scores = train_agent()
    plot.plot_training_progress(scores)

if __name__ == "__main__":
    main()