import gymnasium as gym
import time
import numpy as np
import ale_py
gym.register_envs(ale_py)

def run_and_render_agent(agent, num_episodes_to_render: int = 3, delay: float = 0.5):
    """
    Runs the trained agent in a Gymnasium environment with rendering enabled.
    
    Args:
        agent: A trained DQN model loaded via DQN.load("path/to/model").
        num_episodes_to_render (int): Number of episodes to render.
        delay (float): Sleep delay between steps (seconds).
    """
    print(f"\n--- Demonstrating agent on FrozenLake for {num_episodes_to_render} episodes ---")
    
    # Create a new environment instance with render_mode='human'
    # Make sure to pass the same parameters as the training environment if they are specific.
    render_env = gym.make("ALE/Frogger-v5",obs_type="ram", render_mode='human')
    
    # Store current epsilon and set to 0 for evaluation (no exploration)

    for episode in range(num_episodes_to_render):
        state, info = render_env.reset()
        terminated, truncated = False, False
        total_reward = 0
        print(f"Episode {episode + 1}/{num_episodes_to_render}")

        while not terminated and not truncated:
            action, state = agent.predict(state, deterministic=True)
            state, reward, terminated, truncated, info = render_env.step(action)
            total_reward += reward

        print(f"Episode finished with total reward: {total_reward}")
        time.sleep(1) # Pause before the next episode starts

    # Restore original epsilon
    render_env.close() # Close the rendering environment
    print("Demonstration finished")