import optuna
import environment
import agent
import torch
from utils import get_model_path, get_study_storage_path
import numpy as np
from optuna.storages import JournalStorage, JournalFileStorage
from plot import plot_training_progress

def save_best_model_callback(study, trial):

    if study.best_trial == trial:
        agent_to_save = trial.user_attrs["agent"]
        
        model_path = get_model_path("best_weights_hyperparamoptim.pth")

        torch.save(agent_to_save.local_model.state_dict(), model_path)
        print(f"Trial {trial.number} is the best yet. Model saved to {model_path}")



def get_eps(step, decay_rate=0.999999, min_eps=0.1, starting_eps=1.0):
    return max(min_eps, starting_eps * (decay_rate ** step))


def train_agent(agent, env, episodes, max_steps, eps_decay, min_eps, trial=None):
    scores = []

    for episode in range(episodes):

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

        if trial:
            trial.report(total_reward, episode)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        scores.append(total_reward)

    return np.array(scores)


def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'gamma': trial.suggest_float('gamma', 0.9, 0.9999),
        'tau': trial.suggest_float('tau', 1e-4, 1e-2, log=True),
        'epsilon_end': trial.suggest_float('epsilon_end', 0.01, 0.1),
        'epsilon_decay': trial.suggest_float('epsilon_decay', 0.99, 0.9999999),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
    }

    env = environment.SpaceInvadersEnv()

    agent = agent.Agent(action_space_size=env.action_space.n,
                         learning_rate=params['learning_rate'],
                         gamma=params['gamma'],
                         tau=params['tau'],
                         batch_size=params['batch_size'])
    
    trial.set_user_attr("agent", agent)
    

    try:
        scores = train_agent(agent, env, 
                             episodes=100, max_steps=10000,
                             eps_decay=params['epsilon_decay'], 
                             min_eps=params['epsilon_end'],
                             trial=trial)

    except optuna.exceptions.TrialPruned:
        return trial.last_step_value

    return np.mean(scores[-50:]) if len(scores) >= 50 else -1.0


def optimize_hyperparameters(n_trials=100):

    storage = JournalStorage(JournalFileStorage(get_study_storage_path("ddqn_study.log")))

    study = optuna.create_study(direction='maximize', 
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30),
                                storage=storage,
                                study_name="ddqn_hyperparam_optimization",
                                load_if_exists=True)

    study.optimize(objective, n_trials=n_trials, callbacks=[save_best_model_callback])

    print("Best hyperparameters found:")
    print(study.best_params)
    print("Best score:", study.best_value)
    print("Best trial number:", study.best_trial.number)

    print("Best trial user attributes:")
    if "agent" in study.best_trial.user_attrs:
        best_agent = study.best_trial.user_attrs["agent"]
        print(f"Best agent's model weights saved to {get_model_path('best_weights_hyperparamoptim.pth')}")
        best_agent.local_model.save_model_weights("best_weights_hyperparamoptim.pth")

    plot_training_progress(study)



def main():
    n_trials = 100
    print("Starting hyperparameter optimization...")
    optimize_hyperparameters(n_trials=n_trials)
    print("Hyperparameter optimization completed.")


if __name__ == "__main__":
    main()