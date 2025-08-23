import matplotlib.pyplot as plt
import numpy as np
import optuna

from src.utils import get_plots_dir

def plot_training_progress(episode_rewards: list[float], episode_custom_rewards: list[float], moving_average_window: int = 100, file_name: str = None):
    plt.figure(figsize=(12, 6))
    plt.title("Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    # Plotting the raw episode rewards
    plt.plot(episode_rewards, label="Game Reward", color="blue")
    plt.plot(episode_custom_rewards, label="Custom Reward", color="green")

    # Plotting the moving average rewards
    if moving_average_window > 0:
        moving_avg = np.convolve(episode_rewards, np.ones(moving_average_window)/moving_average_window, mode='valid')
        plt.plot(np.arange(moving_average_window-1, len(episode_rewards)), moving_avg, label="Game Reward Moving Average", color="orange")
        custom_moving_avg = np.convolve(episode_custom_rewards, np.ones(moving_average_window)/moving_average_window, mode='valid')
        plt.plot(np.arange(moving_average_window-1, len(episode_custom_rewards)), custom_moving_avg, label="Custom Reward Moving Average", color="red")

    plt.legend()

    if file_name:
        file_path = f"{get_plots_dir()}/{file_name}.png"
        plt.tight_layout()
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {file_path}")

    plt.show()


def plot_hyperparameter_optimization(study: optuna.Study):
    history_fig = optuna.visualization.plot_optimization_history(study)
    history_fig.show()

    importance_fig = optuna.visualization.plot_param_importances(study)
    importance_fig.show()

    try:
        history_fig.write_html(f"{get_plots_dir()}/optimization_history.html")
        print(f"Optimization history saved to {get_plots_dir()}/optimization_history.html")
    except Exception as e:
        print(f"[WARNING] Could not save optimization_history.html: {e}")

    try:
        importance_fig.write_html(f"{get_plots_dir()}/param_importances.html")
        print(f"Parameter importances saved to {get_plots_dir()}/param_importances.html")
    except Exception as e:
        print(f"[WARNING] Could not save param_importances.html: {e}")

    try:
        history_fig.write_image(f"{get_plots_dir()}/optimization_history.png")
        print(f"Optimization history saved to {get_plots_dir()}/optimization_history.png")
    except Exception as e:
        print(f"[WARNING] Could not save optimization_history.png: {e}")

    try:
        importance_fig.write_image(f"{get_plots_dir()}/param_importances.png")
        print(f"Parameter importances saved to {get_plots_dir()}/param_importances.png")
    except Exception as e:
        print(f"[WARNING] Could not save param_importances.png: {e}")
