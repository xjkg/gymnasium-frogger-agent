import ale_py
import gymnasium as gym
import optuna
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import pandas as pd
import os
import shutil
gym.register_envs(ale_py)
from render_agent import run_and_render_agent 


def plot_learning_curve(log_path, window_size=100):
    """Generates and saves a plot of the learning curve."""
    if log_path == "logs/final_model_logs":
        agent = "DQN"
    else:
        agent = "PPO"
    try:
        log_data = pd.read_csv(log_path + ".monitor.csv", skiprows=1)
        cumulative_timesteps = log_data['l'].cumsum()
        moving_avg = log_data['r'].rolling(window=window_size).mean()
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_timesteps, log_data['r'], alpha=0.3, label='Per-Episode Reward')
        plt.plot(cumulative_timesteps, moving_avg, color='red', linewidth=2, label=f'Moving Average (window={window_size})')
        plt.title(f"Learning Curve of the {agent} Agent")
        plt.xlabel("Timesteps")
        plt.ylabel("Episode Reward")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"learning_curve_{agent}.png")
        print(f"\nLearning curve plot saved to learning_curve_{agent}.png")
    except FileNotFoundError:
        print("\nCould not find monitor log file. Skipping learning curve plot.")

def plot_optuna_study(study):
    """Generates and saves a plot of the Optuna study."""
    if not study.trials:
        print("No trials found in the study to plot.")
        return
    trial_values = [t.value for t in study.trials if t.value is not None]
    if not trial_values:
        print("No successful trials with values to plot.")
        return
    trial_numbers = [t.number for t in study.trials if t.value is not None]
    plt.figure(figsize=(10, 6))
    plt.plot(trial_numbers, trial_values, marker='o', linestyle='--')
    if study.best_trial and study.best_trial.value is not None:
        best_trial_num = study.best_trial.number
        best_trial_val = study.best_trial.value
        plt.scatter(best_trial_num, best_trial_val, s=120, c='red', zorder=5, label=f'Best Trial (#{best_trial_num})')
    plt.title("Optuna Hyperparameter Optimization History")
    plt.xlabel("Trial Number")
    plt.ylabel("Mean Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig("optuna_study.png")
    print("Optuna study plot saved to optuna_study.png")
    plt.close()

def objective(trial):
    """The objective function for Optuna to maximize."""
    trial_env = gym.make("ALE/Frogger-v5", obs_type="ram")
    trial_env = Monitor(trial_env)
    
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999999)
    layer_size = trial.suggest_categorical("layer_size", [32, 64, 128, 256])
    buffer_size = trial.suggest_int("buffer_size", 500000,2000000)
    batch_size= trial.suggest_categorical("batch_size", [32, 64])
    net_arch = [layer_size, layer_size]
    policy_kwargs = dict(net_arch=net_arch)
    
    model = DQN(
        "MlpPolicy", 
        trial_env, 
        learning_rate=learning_rate, 
        gamma=gamma,
        buffer_size=buffer_size,
        batch_size=batch_size,
        policy_kwargs=policy_kwargs,
        verbose=0
    )
    
    model.learn(total_timesteps=10000)
    
    mean_reward, _ = evaluate_policy(model, trial_env, n_eval_episodes=50)
    trial_env.close()
    return mean_reward

def train_ppo():
    env = gym.make("ALE/Frogger-v5", obs_type="ram")
    env = Monitor(env, "ppo_logs/PPO")
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, gamma=0.99)
    model.learn(total_timesteps=30000)
    model.save("ppo_frogger_model")
    env.close()
    eval_env = gym.make("ALE/Frogger-v5", obs_type="ram")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
    eval_env.close()
    print(f"PPO: Mean reward = {mean_reward:.2f} ± {std_reward:.2f}")
    return mean_reward, std_reward

if __name__ == "__main__":
    RENDER = True
    
    # Check and create logs
    LOG_DIR = "logs/"
    if os.path.exists("logs/"):
        shutil.rmtree("logs/")
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Runtime variables
    STORAGE_PATH = "sqlite:///frogger_study_2.db" 
    STUDY_NAME = "frogger-optimization_2"
    NUM_TRIALS_TO_RUN = 100

    study = optuna.create_study(
        direction="maximize",
        study_name=STUDY_NAME,
        storage=STORAGE_PATH,
        load_if_exists=True
    )

    # Avoid additional trials | Increase variable for more runs
    if len(study.trials) < NUM_TRIALS_TO_RUN:
        study.optimize(objective, n_trials=NUM_TRIALS_TO_RUN - len(study.trials))
    else:
        print(f"Study already has {len(study.trials)} trials. Skipping optimization.")

    print("\n--- Best Trial Information ---")
    best_trial = study.best_trial
    if not RENDER:
        if best_trial:
            print(f"  Value (Mean Reward): {best_trial.value:.2f}")
            print("  Params: ")
            for key, value in best_trial.params.items():
                print(f"    {key}: {value}")
                
            print("\n--- Training the final, best model ---")
            best_params = best_trial.params.copy()
            final_layer_size = best_params.pop('layer_size')
            final_policy_kwargs = dict(net_arch=[final_layer_size, final_layer_size])
            
            # Create and wrap the environment with Monitor for logging
            final_env = gym.make("ALE/Frogger-v5", obs_type="ram")
            final_log_path = os.path.join(LOG_DIR, "final_model_logs")
            final_env = Monitor(final_env, final_log_path)
            
            final_model = DQN("MlpPolicy", final_env, policy_kwargs=final_policy_kwargs,
                            **best_params, verbose=0)
            
            final_model.learn(total_timesteps=500000)
            final_model.save("best_frogger_model")
            print("\nFinal model saved to best_frogger_model.zip")
            
            # Built-in evaluation
            print("\n--- Evaluating Final Model Performance ---")
            eval_env = gym.make("ALE/Frogger-v5", obs_type="ram")
            mean_reward, std_reward = evaluate_policy(final_model, eval_env, n_eval_episodes=100)
            print(f"Final Model: Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
            eval_env.close()
            
            # Images & Plotting
            plot_optuna_study(study)
            plot_learning_curve(final_log_path)
        else:
            print("No successful trials were completed. Cannot train or evaluate a final model.")

        # PPO
        train_ppo()
        plot_learning_curve("ppo_logs/PPO")
    elif RENDER == True:
        best_agent = DQN.load("best_frogger_model")
        run_and_render_agent(best_agent, num_episodes_to_render=5, delay=0.3)
