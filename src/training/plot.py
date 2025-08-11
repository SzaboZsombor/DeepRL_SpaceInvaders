import matplotlib.pyplot as plt
import numpy as np
from utils import get_plots_dir

def plot_training_progress(episode_rewards, moving_average_window=100, file_name=None):
    plt.figure(figsize=(12, 6))
    plt.title("Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    # Plotting the raw episode rewards
    plt.plot(episode_rewards, label="Episode Reward", color="blue")

    # Plotting the moving average
    if moving_average_window > 0:
        moving_avg = np.convolve(episode_rewards, np.ones(moving_average_window)/moving_average_window, mode='valid')
        plt.plot(np.arange(moving_average_window-1, len(episode_rewards)), moving_avg, label="Moving Average", color="orange")

    plt.legend()
    
    if file_name:
        file_path = f"{get_plots_dir()}/{file_name}.png"
        plt.tight_layout()
        plt.savefig(file_path)
        print(f"Plot saved to {file_path}")

    plt.show()
