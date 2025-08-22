import numpy as np
import os
import yaml
from tqdm import trange

from src.agent import Agent
from src.environment import create_env
from src.training.plot import plot_training_progress
from src.utils import get_logs_dir


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
NUM_STACK = config["NUM_STACK"]
LIFE_LOST_PENALTY = config["LIFE_LOST_PENALTY"]
MISSED_SHOT_PENALTY = config["MISSED_SHOT_PENALTY"]


env = create_env(env_id="ALE/SpaceInvaders-v5", render_mode=None, num_stack=NUM_STACK, life_lost_penalty=LIFE_LOST_PENALTY, missed_shot_penalty=MISSED_SHOT_PENALTY)

agent = Agent(action_space_size=env.action_space.n, learning_rate=LEARNING_RATE, gamma=GAMMA, tau=TAU, batch_size=BATCH_SIZE, capacity=CAPACITY)
agent.local_model.load_model_weights("best_ddqn_agent.pth")

def get_eps(step, decay_rate, min_eps, starting_eps):
    return max(min_eps, starting_eps * (decay_rate ** step))


def train_agent(episodes=10000, max_steps=10000, weights_output_name="best_ddqn_agent.pth"):
    best_score = -np.inf

    if os.path.exists(f"{get_logs_dir()}/training_game_scores.npy") and os.path.exists(f"{get_logs_dir()}/training_custom_scores.npy"):
        scores = np.load(f"{get_logs_dir()}/training_game_scores.npy").tolist()
        custom_scores = np.load(f"{get_logs_dir()}/training_custom_scores.npy").tolist()
    else:
        scores = []
        custom_scores = []

    for episode in trange(len(scores), episodes, desc="Training Progress"):

        state, _ = env.reset()
        total_score_reward = 0.
        total_custom_reward = 0.

        for t in range(max_steps):

            action = agent.act(state, epsilon=get_eps(agent.time_step, decay_rate=DECAY_RATE, min_eps=MIN_EPS, starting_eps=STARTING_EPS))

            next_state, total_reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            scaled_reward = np.sign(total_reward) * np.log1p(np.abs(total_reward))

            agent.step(state, action, scaled_reward, next_state, done)
            state = next_state

            total_score_reward += info.get('score_reward', 0)
            total_custom_reward += info.get('custom_reward', 0)

            if done:
                break

        if best_score < total_score_reward:
            best_score = total_score_reward
            agent.local_model.save_model_weights(weights_output_name)

        scores.append(total_score_reward)
        custom_scores.append(total_custom_reward)
        print(f" Episode {episode + 1}/{episodes} - Score Reward: {total_score_reward} - Custom Reward: {total_custom_reward}", flush=True)

        np.save(f"{get_logs_dir()}/training_custom_scores.npy", np.array(custom_scores))
        np.save(f"{get_logs_dir()}/training_game_scores.npy", np.array(scores))


def main():
    print("Starting training...")
    episodes = 20000
    max_steps = 10000

    weights_output_name = "best_ddqn_agent.pth"

    train_agent(episodes=episodes, max_steps=max_steps, weights_output_name=weights_output_name)

    scores = np.load(f"{get_logs_dir()}/training_game_scores.npy")
    custom_scores = np.load(f"{get_logs_dir()}/training_custom_scores.npy")

    plot_training_progress(scores, custom_scores, moving_average_window=100, file_name='training_rewards.png')


if __name__ == "__main__":
    main()