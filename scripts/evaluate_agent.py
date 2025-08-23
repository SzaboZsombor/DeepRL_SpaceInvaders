import numpy as np


def evaluate_agent(agent, env, num_episodes=10, render=False):
    """
    Evaluate trained agent performance
    
    Returns:
        dict: Evaluation metrics (mean_reward, std_reward, success_rate)
    """
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
