import gymnasium as gym
from gymnasium.wrappers import FrameStack, AtariPreprocessing

class SpaceInvadersEnv:
    def __init__(self, env_id = "ALE/SpaceInvaders-v5", render_mode = None):

        print(f"Initializing environment: {env_id}")

        self.env = gym.make(env_id, render_mode=render_mode)

        self.env = AtariPreprocessing(self.env, frame_skip=1) #because of the v5 version it skips 4 frames by default

        self.env = FrameStack(self.env, num_stack=4)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self):
        print("Closing environment")
        return self.env.close()



if __name__ == '__main__':
    # Initialize the environment with a render mode to see the game
    env_manager = SpaceInvadersEnv(render_mode='human')
    
    # Get the initial state of the environment
    observation, info = env_manager.reset()
    
    done = False
    truncated = False
    total_reward = 0
    
    # Run a simple loop with random actions
    print("Starting a short test run with random actions...")
    while not done and not truncated:
        action = env_manager.action_space.sample()  # Take a random action
        observation, reward, done, truncated, info = env_manager.step(action)
        total_reward += reward

        print(observation)
        
    print(f"Test run finished. Total reward: {total_reward}")
    env_manager.close()