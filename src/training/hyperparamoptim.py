import os
import torch
import numpy as np
from tqdm import trange
import optuna
from optuna.storages import RDBStorage
import gymnasium as gym

from src.environment import create_env
from src.agent import Agent
from src.utils import get_model_path, get_study_storage_path
from src.training.plot import plot_hyperparameter_optimization

TEMP_MODEL_FILENAME = get_model_path("temp_agent_hyperparam_optim.pth")
BEST_MODEL_FILENAME = get_model_path("best_ddqn_agent_hyperparam_optim.pth")


NUM_EPISODES = 750
MAX_STEPS_PER_EPISODE = 10000
CAPACITY = 200000
LIFE_LOST_PENALTY = -10.0
MISSED_SHOT_PENALTY = -1.0
NUM_STACK = 4
FRAME_SKIP = 4


def save_best_model_callback(study_: optuna.Study, trial: optuna.Trial):
    if study_.best_trial == trial:
        if os.path.exists(TEMP_MODEL_FILENAME):
            os.replace(TEMP_MODEL_FILENAME, BEST_MODEL_FILENAME)
            print(f"\nTrial {trial.number} is the best. Model saved to {BEST_MODEL_FILENAME}\n")


def setup_study(study_name: str, storage_path: str) -> optuna.Study:
    os.makedirs(os.path.dirname(storage_path), exist_ok=True)
    db_url = f"sqlite:///{storage_path}"
    storage = RDBStorage(url=db_url)
    
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30),
        storage=storage,
        study_name=study_name,
        load_if_exists=True
    )
    return study


def get_eps(step: int, decay_rate: float = 0.999999, min_eps: float = 0.1, starting_eps: float = 1.0) -> float:
    return max(min_eps, starting_eps * (decay_rate ** step))


def train_agent(agent: Agent, env: gym.Env, episodes: int, max_steps: int, eps_decay: float, min_eps: float, trial: optuna.Trial = None) -> np.ndarray:
    scores = []

    for episode in trange(episodes, desc=f"Trial {trial.number if trial else '-'}"):
        state, _ = env.reset()
        total_score_reward = 0.0

        for t in range(max_steps):

            action = agent.act(state, epsilon=get_eps(agent.time_step, decay_rate=eps_decay, min_eps=min_eps))

            next_state, total_reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            scaled_reward = np.sign(total_reward) * np.log1p(np.abs(total_reward))

            agent.step(state, action, scaled_reward, next_state, done)
            state = next_state

            total_score_reward += info.get('score_reward', 0)

            if done:
                break
        
        scores.append(total_score_reward)

        if trial:
            trial.report(total_score_reward, episode)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return np.array(scores)


def objective(trial: optuna.Trial) -> float:
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'gamma': trial.suggest_float('gamma', 0.9, 0.9999),
        'tau': trial.suggest_float('tau', 1e-4, 1e-2, log=True),
        'epsilon_end': trial.suggest_float('epsilon_end', 0.01, 0.1),
        'epsilon_decay': trial.suggest_float('epsilon_decay', 0.999, 0.99999),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128])
    }

    env = create_env(env_id="ALE/SpaceInvaders-v5", render_mode=None, frame_skip=FRAME_SKIP, num_stack=NUM_STACK, life_lost_penalty=LIFE_LOST_PENALTY, missed_shot_penalty=MISSED_SHOT_PENALTY)
    
    agent_instance = Agent(action_space_size=env.action_space.n,
                             learning_rate=params['learning_rate'],
                             gamma=params['gamma'],
                             tau=params['tau'],
                             batch_size=params['batch_size'],
                             capacity=CAPACITY)

    try:
        scores = train_agent(agent=agent_instance, env=env,
                             episodes=NUM_EPISODES, max_steps=MAX_STEPS_PER_EPISODE,
                             eps_decay=params['epsilon_decay'], 
                             min_eps=params['epsilon_end'],
                             trial=trial)

    except optuna.exceptions.TrialPruned:
        return trial.last_step_value

    torch.save(agent_instance.local_model.state_dict(), TEMP_MODEL_FILENAME)
    env.close()

    return np.mean(scores[-100:]) if len(scores) >= 100 else -1.0


def optimize_hyperparameters(n_trials: int = 100):
    study_name = "ddqn_hyperparam_optimization"
    storage_path = get_study_storage_path("ddqn_study.db")
    
    study = setup_study(study_name, storage_path)
    
    if n_trials > len(study.trials):
        n_trials = n_trials - len(study.trials)
        study.optimize(objective, n_trials=n_trials, callbacks=[save_best_model_callback])

    print("\n--- Optimization Finished ---")
    print(f"Best score: {study.best_value}")
    print("Best hyperparameters found:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    plot_hyperparameter_optimization(study)


def main():
    n_trials = 45
    print("Starting hyperparameter optimization...")
    optimize_hyperparameters(n_trials=n_trials)
    print("Hyperparameter optimization completed.")

if __name__ == "__main__":
    main()