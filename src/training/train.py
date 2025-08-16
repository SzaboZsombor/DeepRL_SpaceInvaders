import numpy as np
import os
import sys
import yaml
from tqdm import trange

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import agent
import environment
from plot import plot_training_progress
from utils import get_logs_dir


with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

LEARNING_RATE = config["LEARNING_RATE"]
GAMMA = config["GAMMA"]
TAU = config["TAU"]
BATCH_SIZE = config["BATCH_SIZE"]
DECAY_RATE = config["DECAY_RATE"]
MIN_EPS = config["MIN_EPS"]
STARTING_EPS = config["STARTING_EPS"]
CAPACITY = config["CAPACITY"]

env = environment.SpaceInvadersEnv()

agent = agent.Agent(action_space_size=env.action_space.n, learning_rate=LEARNING_RATE, gamma=GAMMA, tau=TAU, batch_size=BATCH_SIZE, capacity=CAPACITY)
agent.local_model.load_model_weights("best_ddqn_agent.pth")

def get_eps(step, decay_rate, min_eps, starting_eps):
    return max(min_eps, starting_eps * (decay_rate ** step))


def train_agent(episodes=10000, max_steps=10000, weights_output_name="best_ddqn_agent.pth"):
    best_score = -np.inf

    if os.path.exists(f"{get_logs_dir()}/training_scores.npy"):
        scores = np.load(f"{get_logs_dir()}/training_scores.npy").tolist()
    else:
        scores = []

    for episode in trange(len(scores), episodes, desc="Training Progress"):

        state, _ = env.reset()
        total_reward = 0

        for t in range(max_steps):

            action = agent.act(state, epsilon=get_eps(agent.time_step, decay_rate=DECAY_RATE, min_eps=MIN_EPS, starting_eps=STARTING_EPS))

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
        print(f" Episode {episode + 1}/{episodes} - Total Reward: {total_reward}", flush=True)

        np.save(f"{get_logs_dir()}/training_scores.npy", np.array(scores))


def main():
    print("Starting training...")
    episodes = 12000
    max_steps = 10000

    weights_output_name = "best_ddqn_agent.pth"

    train_agent(episodes=episodes, max_steps=max_steps, weights_output_name=weights_output_name)

    scores = np.load(f"{get_logs_dir()}/training_scores.npy")
    plot_training_progress(scores, moving_average_window=100, file_name='training_rewards.png')


if __name__ == "__main__":
    main()