import gymnasium as gym
import time
import numpy as np
import ale_py
gym.register_envs(ale_py)
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

class AddChannelDim(gym.ObservationWrapper):
    """Add a channel dimension to grayscale images.""" # Needed for env = VecTransposeImage(env) to work to allow CnnPolicy to work on PPO agent
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0], obs_shape[1], 1),
            dtype=np.uint8
        )

    def observation(self, obs):
        return np.expand_dims(obs, axis=-1)

def run_and_render_agent(agent, obs_type, num_episodes_to_render: int = 3, delay: float = 0.5):
    """
    Runs the trained agent in a Gymnasium environment with rendering enabled.
    
    Args:
        agent: A trained and loaded agent.
        num_episodes_to_render (int): Number of episodes to render.
        delay (float): Sleep delay between steps (seconds).
    """
    print(f"\n--- Demonstrating agent on Frogger for {num_episodes_to_render} episodes ---")
    
    # Create a new environment instance with render_mode='human'
    # Make sure to pass the same parameters as the training environment if they are specific.
    render_env = gym.make("ALE/Frogger-v5", obs_type=obs_type, render_mode='human', frameskip=4 if obs_type =="grayscale" else 1)

    if obs_type == "grayscale":
        render_env = AddChannelDim(render_env)
    
    for episode in range(num_episodes_to_render):
        if obs_type == "grayscale":
            obs, _ = render_env.reset()
        else:
            obs, info = render_env.reset()
        terminated, truncated = False, False
        total_reward = 0
        print(f"Episode {episode + 1}/{num_episodes_to_render}")

        while not (terminated or truncated):
            
            if obs_type == "grayscale":
                obs = np.transpose(obs, (2, 0, 1))
                obs = np.expand_dims(obs, axis=0)   
                action, _ = agent.predict(obs, deterministic=True)
                action = np.squeeze(action).item()
                obs, reward, terminated, truncated, _ = render_env.step(action)
            else:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = render_env.step(action)

            total_reward += reward

        print(f"Episode finished with total reward: {total_reward}")
        time.sleep(1) # Pause before the next episode starts

    render_env.close() # Close the rendering environment
    print("Demonstration finished")