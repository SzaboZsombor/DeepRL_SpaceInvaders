import sys
import os

from src.environment import create_env
from src.agent import Agent
import time


def main():
    env_manager = create_env(env_id="ALE/SpaceInvaders-v5", render_mode='human')
    agent = Agent(action_space_size=env_manager.action_space.n, eval_mode=True)
    agent.local_model.load_model_weights('best_ddqn_agent.pth')

    observation, info = env_manager.reset()
    done = False
    truncated = False
    total_reward = 0

    while not done and not truncated:
        action = agent.act(observation)
        next_observation, reward, done, truncated, info = env_manager.step(action)
        observation = next_observation
        total_reward += reward

        print(f"Action: {action}, Reward: {reward}, Done: {done}, Truncated: {truncated}")
        time.sleep(0.1)

    print(f"Total reward: {total_reward}")
    env_manager.close()

if __name__ == "__main__":
    main()