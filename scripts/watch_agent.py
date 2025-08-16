import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from environment import SpaceInvadersEnv
from agent import Agent
from utils import get_models_dir

def main():
    env_manager = SpaceInvadersEnv(render_mode='human')
    agent = Agent(env_manager)
    agent.load_model(os.path.join(get_models_dir(), 'best_ddqn_model.pth'))

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

    print(f"Total reward: {total_reward}")
    env_manager.close()

if __name__ == "__main__":
    main()