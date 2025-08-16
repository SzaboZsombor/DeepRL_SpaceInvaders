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