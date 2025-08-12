import os
import sys
import torch
import numpy as np
from tqdm import trange
import optuna
from optuna.storages import RDBStorage

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from environment import SpaceInvadersEnv
from agent import Agent
from utils import get_model_path, get_study_storage_path

TEMP_MODEL_FILENAME = get_model_path("temp_agent.pth")
BEST_MODEL_FILENAME = get_model_path("best_ddqn_agent.pth")


def save_best_model_callback(study, trial):
    if study.best_trial == trial:
        if os.path.exists(TEMP_MODEL_FILENAME):
            os.replace(TEMP_MODEL_FILENAME, BEST_MODEL_FILENAME)
            print(f"\nTrial {trial.number} is the best. Model saved to {BEST_MODEL_FILENAME}\n")


def setup_study(study_name, storage_path):
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


def get_eps(step, decay_rate=0.999999, min_eps=0.1, starting_eps=1.0):
    return max(min_eps, starting_eps * (decay_rate ** step))


def train_agent(agent, env, episodes, max_steps, eps_decay, min_eps, trial=None):
    scores = []
    for episode in trange(episodes, desc=f"Trial {trial.number if trial else '-'}"):
        state, _ = env.reset()
        total_reward = 0
        for t in range(max_steps):
            action = agent.act(state, epsilon=get_eps(agent.time_step, decay_rate=eps_decay, min_eps=min_eps))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            clipped_reward = np.clip(reward, -1, 1)
            agent.step(state, action, clipped_reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
        
        scores.append(total_reward)

        if trial:
            trial.report(total_reward, episode)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return np.array(scores)


def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'gamma': trial.suggest_float('gamma', 0.9, 0.9999),
        'tau': trial.suggest_float('tau', 1e-4, 1e-2, log=True),
        'epsilon_end': trial.suggest_float('epsilon_end', 0.01, 0.1),
        'epsilon_decay': trial.suggest_float('epsilon_decay', 0.999, 0.99999),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128])
    }

    env = SpaceInvadersEnv()
    agent_instance = Agent(action_space_size=env.action_space.n,
                             learning_rate=params['learning_rate'],
                             gamma=params['gamma'],
                             tau=params['tau'],
                             batch_size=params['batch_size'])
    
    try:
        scores = train_agent(agent=agent_instance, env=env, 
                             episodes=100, max_steps=10000,
                             eps_decay=params['epsilon_decay'], 
                             min_eps=params['epsilon_end'],
                             trial=trial)

    except optuna.exceptions.TrialPruned:
        return trial.last_step_value

    torch.save(agent_instance.local_model.state_dict(), TEMP_MODEL_FILENAME)
    
    return np.mean(scores[-50:]) if len(scores) >= 50 else -1.0


def optimize_hyperparameters(n_trials=100):
    study_name = "ddqn_hyperparam_optimization"
    storage_path = get_study_storage_path("ddqn_study.db")
    
    study = setup_study(study_name, storage_path)
    
    study.optimize(objective, n_trials=n_trials, callbacks=[save_best_model_callback])

    print("\n--- Optimization Finished ---")
    print(f"Best score: {study.best_value}")
    print("Best hyperparameters found:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")


def main():
    n_trials = 100
    print("Starting hyperparameter optimization...")
    optimize_hyperparameters(n_trials=n_trials)
    print("Hyperparameter optimization completed.")

if __name__ == "__main__":
    main()