import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import agent
import environment
from plot import plot_training_progress

LEARNING_RATE = 1e-4
GAMMA = 0.99
TAU = 1e-3
BATCH_SIZE = 64

env = environment.SpaceInvadersEnv()

agent = agent.Agent(action_space_size=env.action_space.n, learning_rate=LEARNING_RATE, gamma=GAMMA, tau=TAU, batch_size=BATCH_SIZE)
agent.local_model.load_model_weights("best_model.pth")

def get_eps(step, decay_rate=0.999999, min_eps=0.1, starting_eps=1.0):
    return max(min_eps, starting_eps * (decay_rate ** step))


def train_agent(episodes=1000, max_steps=10000, weights_output_name="best_model.pth"):
    best_score = -np.inf
    scores = []

    for episode in range(episodes):

        state, _ = env.reset()
        total_reward = 0

        for t in range(max_steps):

            action = agent.act(state, epsilon=get_eps(agent.time_step))

            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated
            clipped_reward = np.clip(reward, -1, 1)

            agent.step(state, action, clipped_reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break

        if best_score < total_reward:
            best_score = total_reward
            agent.local_model.save_model_weights(weights_output_name)

        scores.append(total_reward)
        print(f"Episode {episode + 1}/{episodes} - Total Reward: {total_reward}")

    return np.array(scores)


def main():
    print("Starting training...")

    weights_output_name = "best_model.pth"

    scores = train_agent(weights_output_name=weights_output_name)
    plot_training_progress(scores)


if __name__ == "__main__":
    main()