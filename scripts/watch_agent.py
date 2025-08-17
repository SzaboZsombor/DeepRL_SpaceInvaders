import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.environment import SpaceInvadersEnv
from src.agent import Agent
from src.utils import get_models_dir


def main():
    env_manager = SpaceInvadersEnv(render_mode='human')
    agent = Agent(action_space_size=env_manager.action_space.n, eval_mode=True)
    agent.local_model.load_model_weights('best_ddqn_model.pth')

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