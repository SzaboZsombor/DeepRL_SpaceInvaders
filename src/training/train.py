import numpy as np
import os
import yaml
from tqdm import trange
import time

from src.agent import Agent
from src.environment import create_env
from src.training.plot import plot_training_progress
from src.utils import get_logs_dir, get_plots_dir
from src.training.metrics import MetricsTracker


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

metrics = MetricsTracker(
    log_dir=get_logs_dir(),
    plot_dir=get_plots_dir(), 
    use_wandb=True,
    wandb_project="space-invaders-ddqn"
)

def get_eps(step: int, decay_rate: float, min_eps: float, starting_eps: float) -> float:
    return max(min_eps, starting_eps * (decay_rate ** step))


def train_agent(episodes: int = 10000, max_steps: int = 10000, weights_output_name: str = "best_ddqn_agent.pth") -> MetricsTracker:
    best_score = -np.inf

    if os.path.exists(f"{get_logs_dir()}/training_game_scores.npy") and os.path.exists(f"{get_logs_dir()}/training_custom_scores.npy"):
        scores = np.load(f"{get_logs_dir()}/training_game_scores.npy").tolist()
        custom_scores = np.load(f"{get_logs_dir()}/training_custom_scores.npy").tolist()
    else:
        scores = []
        custom_scores = []

    for episode in trange(len(scores), episodes, desc="Training Progress"):
        episode_start_time = time.time()

        state, _ = env.reset()
        total_score_reward = 0.
        total_custom_reward = 0.
        steps = 0
        loss = 0.

        for t in range(max_steps):

            action = agent.act(state, epsilon=get_eps(agent.time_step, decay_rate=DECAY_RATE, min_eps=MIN_EPS, starting_eps=STARTING_EPS))

            next_state, total_reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            scaled_reward = np.sign(total_reward) * np.log1p(np.abs(total_reward))

            agent.step(state, action, scaled_reward, next_state, done)
            state = next_state
            
            metrics.log_training_step(
                gradient_norm=agent.get_grad_norm(),
                q_value_estimate=agent.get_q_value_estimate(),
                replay_buffer_size=len(agent.memory),
                step=agent.time_step
            )

            total_score_reward += info.get('score_reward', 0)
            total_custom_reward += info.get('custom_reward', 0)
            steps += 1

            loss += agent.get_loss()

            if done:
                break
        
        
        episode_time = time.time() - episode_start_time
        current_epsilon = get_eps(agent.time_step, decay_rate=DECAY_RATE, min_eps=MIN_EPS, starting_eps=STARTING_EPS)
        
        metrics.log_system_metrics(episode)

        metrics.log_episode(
            episode=episode,
            score_reward=total_score_reward, 
            custom_reward=total_custom_reward,
            length=steps,
            loss=loss / steps if steps > 0 else 0,
            epsilon=current_epsilon,
            episode_time=episode_time,
            lives_lost=info.get('lives_lost', 0),
            shots_fired=info.get('shots_fired', 0),
            enemies_killed=info.get('enemies_killed', 0)
        )

        if best_score < total_score_reward:
            best_score = total_score_reward
            agent.local_model.save_model_weights(weights_output_name)

            metrics.log_checkpoint(
                episode=episode,
                model_path=weights_output_name,
                score_reward=total_score_reward,
                custom_reward=total_custom_reward,
                is_best=True
            )

        scores.append(total_score_reward)
        custom_scores.append(total_custom_reward)
        print(f" Episode {episode + 1}/{episodes} - Score Reward: {total_score_reward} - Custom Reward: {total_custom_reward}", flush=True)

        np.save(f"{get_logs_dir()}/training_custom_scores.npy", np.array(custom_scores))
        np.save(f"{get_logs_dir()}/training_game_scores.npy", np.array(scores))
    
    return metrics


def main():
    print("Starting training...")
    episodes = 20000
    max_steps = 10000

    weights_output_name = "best_ddqn_agent.pth"

    hyperparams = {
        "learning_rate": LEARNING_RATE,
        "gamma": GAMMA,
        "tau": TAU,
        "batch_size": BATCH_SIZE,
        "decay_rate": DECAY_RATE,
        "min_eps": MIN_EPS,
        "starting_eps": STARTING_EPS,
        "capacity": CAPACITY,
        "num_stack": NUM_STACK,
        "life_lost_penalty": LIFE_LOST_PENALTY,
        "missed_shot_penalty": MISSED_SHOT_PENALTY,
        "episodes": episodes,
        "max_steps": max_steps,
        "weights_output_name": weights_output_name
    }
    
    metrics.log_hyperparameters(hyperparams)

    metrics = train_agent(episodes=episodes, max_steps=max_steps, weights_output_name=weights_output_name)

    report = metrics.generate_training_report()
    print(report)

    training_metrics_plot_path = os.path.join(get_plots_dir(), 'training_progress.png')
    metrics.plot_training_progress(save_path=training_metrics_plot_path)

    metrics.finish_wandb()

if __name__ == "__main__":
    main()