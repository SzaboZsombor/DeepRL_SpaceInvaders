import time
import imageio

from src.environment import create_env
from src.agent import Agent
from src.utils import get_plots_path


def save_gif(frames: list[float, float, float], filename: str, fps: int = 10):
    file_path = get_plots_path(filename)
    imageio.mimsave(file_path, frames, duration=1/fps)
    print(f"GIF saved to {file_path}")


def main():
    env_manager = create_env(env_id="ALE/SpaceInvaders-v5", render_mode='human')
    agent = Agent(action_space_size=env_manager.action_space.n, eval_mode=True)
    agent.local_model.load_model_weights('best_ddqn_agent.pth')

    output_gif_name = "agent_playing.gif"

    observation, info = env_manager.reset()
    done = False
    truncated = False
    total_reward = 0
    frames = []

    while not done and not truncated:
        frames.append(env_manager.render(mode='rgb_array'))

        action = agent.act(observation)
        next_observation, reward, done, truncated, info = env_manager.step(action)
        observation = next_observation

        total_reward += reward

        print(f"Action: {action}, Reward: {reward}, Done: {done}, Truncated: {truncated}")
        time.sleep(0.1)

    save_gif(frames, output_gif_name)

    print(f"Total reward: {total_reward}")
    env_manager.close()


if __name__ == "__main__":
    main()