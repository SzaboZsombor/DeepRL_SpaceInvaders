import gymnasium as gym
from gymnasium.wrappers import FrameStack, AtariPreprocessing


class SpaceInvadersEnv(gym.Wrapper):

    def __init__(self, env, life_lost_penalty: float, missed_shot_penalty: float):
        super().__init__(env)
        self.env = env
        self.life_lost_penalty = life_lost_penalty
        self.missed_shot_penalty = missed_shot_penalty
        self.current_lives = 0
        self.bullet_visibility = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.current_lives = info.get("ale.lives", 0)
        self.bullet_visibility = False

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        score_reward = reward
        custom_reward = 0.

        ram = self.env.unwrapped.ale.getRAM()
        is_bullet_currently_visible = ram[82] < 192

        if self.bullet_visibility and not is_bullet_currently_visible:
            custom_reward += self.missed_shot_penalty
            self.bullet_visibility = False

        new_lives = info.get("ale.lives", 0)
        if new_lives < self.current_lives:
            custom_reward += self.life_lost_penalty
            self.current_lives = new_lives

        self.bullet_visibility = is_bullet_currently_visible

        info['score_reward'] = score_reward
        info['custom_reward'] = custom_reward

        return obs, score_reward + custom_reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
    

def create_env(env_id="ALE/SpaceInvaders-v5", render_mode=None, num_stack=4, life_lost_penalty=-10.0, missed_shot_penalty=-1.0):

    env = gym.make(env_id, render_mode=render_mode)

    env = AtariPreprocessing(env, frame_skip=1)
    env = SpaceInvadersEnv(env, life_lost_penalty=life_lost_penalty, missed_shot_penalty=missed_shot_penalty)
    env = FrameStack(env, num_stack=num_stack)

    print(f"Environment created: {env_id} with render mode: {render_mode}")

    return env