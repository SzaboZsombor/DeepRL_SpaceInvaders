import time
import imageio
import os

from src.environment import create_env
from src.agent import Agent
from src.utils import get_plots_dir


def save_gif(frames: list[float, float, float], filename: str, fps: int = 15):
    file_path = os.path.join(get_plots_dir(), filename)
    imageio.mimsave(file_path, frames, fps = fps, loop = 0)
    print(f"GIF saved to {file_path}")


def main():
    env_manager = create_env(env_id="ALE/SpaceInvaders-v5", render_mode='rgb_array')
    agent = Agent(action_space_size=env_manager.action_space.n, eval_mode=True)
    agent.local_model.load_model_weights('best_ddqn_agent.pth')

    output_gif_name = "agent_playing.gif"

    observation, info = env_manager.reset()
    done = False
    truncated = False
    total_reward = 0
    frames = []
    step_count = 0

    start_recording_after = 35  # Skipping first 35 steps
    max_frames = 250  # Limiting video to 250 frames

    while not done and not truncated:
        frame = env_manager.render()

        if step_count >= start_recording_after:
            frames.append(frame)


        if len(frames) >= max_frames:
            print(f"Reached maximum frames ({max_frames}), stopping recording...")
            break

        action = agent.act(observation)
        next_observation, reward, done, truncated, info = env_manager.step(action)
        observation = next_observation

        total_reward += reward
        step_count += 1

        print(f"Action: {action}, Reward: {reward}, Done: {done}, Truncated: {truncated}")
        time.sleep(0.1)

    save_gif(frames, output_gif_name)

    print(f"Total reward: {total_reward}")
    env_manager.close()


if __name__ == "__main__":
    main()