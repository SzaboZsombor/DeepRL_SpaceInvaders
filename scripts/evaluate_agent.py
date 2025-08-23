import numpy as np

from src.environment import create_env
from src.agent import Agent


def evaluate_agent(agent, env, num_episodes=10, render=False):
    scores = []
    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(obs, epsilon=0.0)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            done = done or truncated
            
        scores.append(total_reward)
    
    return {
        'mean_reward': np.mean(scores),
        'std_reward': np.std(scores),
        'min_reward': np.min(scores),
        'max_reward': np.max(scores)
    }


def main():
    env_manager = create_env(env_id="ALE/SpaceInvaders-v5", render_mode='human')
    agent = Agent(action_space_size=env_manager.action_space.n, eval_mode=True)
    agent.local_model.load_model_weights('best_ddqn_agent.pth')

    evaluation_results = evaluate_agent(agent, env_manager)
    print("Evaluation results:")
    for key, value in evaluation_results.items():
        print(f"  {key}: {value}")

    env_manager.close()