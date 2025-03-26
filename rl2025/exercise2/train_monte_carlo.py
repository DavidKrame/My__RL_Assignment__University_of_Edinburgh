import gymnasium as gym

from rl2025.constants import EX2_MC_CONSTANTS as CONSTANTS
from rl2025.exercise2.agents import MonteCarloAgent
from rl2025.exercise2.utils import evaluate
from tqdm import tqdm

from rl2025.util.result_processing import Run, get_best_saved_run
import copy
import numpy as np

HYPERPARAM_SWEEP = False # False to run just with the default CONFIG parameters
IS_SLIPPERY = True

CONFIG = {
    "eval_freq": 5000, # keep this unchanged
    "epsilon": 0.9,
    "gamma": 0.99,
    "save_filename": None,
}
CONFIG.update(CONSTANTS)


# List of hyperparameter configurations to test
hyperparameter_configs = [
    {"epsilon": 0.9, "gamma": 0.99},
    {"epsilon": 0.9, "gamma": 0.8},
]
NUM_SEEDS = 10
all_runs = []

def monte_carlo_eval(
        env,
        config,
        q_table,
        render=False):
    """
    Evaluate configuration of MC on given environment when initialised with given Q-table

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :param q_table (Dict[(Obs, Act), float]): Q-table mapping observation-action to Q-values
    :param render (bool): flag whether evaluation runs should be rendered
    :return (float, float): mean and standard deviation of returns received over episodes
    """
    eval_agent = MonteCarloAgent(
        action_space=env.action_space,
        obs_space=env.observation_space,
        gamma=CONFIG["gamma"],
        epsilon=0.0,
    )
    eval_agent.q_table = q_table
    if render:
        eval_env = gym.make(CONFIG["env"], render_mode="human", is_slippery=IS_SLIPPERY)
    else:
        eval_env = env
    return evaluate(eval_env, eval_agent, config["eval_eps_max_steps"], config["eval_episodes"])


def train(env, config):
    """
    Train and evaluate MC on given environment with provided hyperparameters

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :return (float, List[float], List[float], Dict[(Obs, Act), float]):
        returns over all episodes, list of means and standard deviations of evaluation
        returns, final Q-table, final state-action counts
    """
    agent = MonteCarloAgent(
        action_space=env.action_space,
        obs_space=env.observation_space,
        gamma=config["gamma"],
        epsilon=config["epsilon"],
    )

    step_counter = 0
    max_steps = config["total_eps"] * config["eps_max_steps"]

    total_reward = 0
    evaluation_return_means = []
    evaluation_negative_returns = []

    for eps_num in tqdm(range(1, config["total_eps"] + 1)):
        obs, _ = env.reset()

        t = 0
        episodic_return = 0

        obs_list, act_list, rew_list = [], [], []
        while t < config["eps_max_steps"]:
            agent.schedule_hyperparameters(step_counter, max_steps)
            act = agent.act(obs)

            n_obs, reward, terminated, truncated, _ = env.step(act)
            done = terminated or truncated

            obs_list.append(obs)
            rew_list.append(reward)
            act_list.append(act)

            t += 1
            step_counter += 1
            episodic_return += reward

            if done:
                break

            obs = n_obs

        agent.learn(obs_list, act_list, rew_list)
        total_reward += episodic_return

        if eps_num > 0 and eps_num % config["eval_freq"] == 0:
            mean_return, negative_returns = monte_carlo_eval(env, config, agent.q_table)
            tqdm.write(f"EVALUATION: EP {eps_num} - MEAN RETURN {mean_return}")
            evaluation_return_means.append(mean_return)
            evaluation_negative_returns.append(negative_returns)

    return total_reward, evaluation_return_means, evaluation_negative_returns, agent.q_table


if __name__ == "__main__":
    # env = gym.make(CONFIG["env"])
    # total_reward, _, _, q_table = train(env, CONFIG)
    
    if not HYPERPARAM_SWEEP:
        # Run with default parameters from CONFIG
        env = gym.make(CONFIG["env"], is_slippery=IS_SLIPPERY)
        total_reward, eval_return_means, _, q_table = train(env, CONFIG)
        env.close()
        mean_value = eval_return_means[-1] if eval_return_means else None
        print("------------------------------------------------------------------------------")
        print("Default Parameters Run - Last Mean Evaluation Return:", mean_value)
    else:
        # All hyperparameters
        all_runs = []
        for hp_config in hyperparameter_configs:
            current_config = copy.deepcopy(CONFIG)
            current_config.update(hp_config)
            
            run_instance = Run(current_config)
            run_instance.run_name = f"Env:{current_config['env']}_eps:{current_config['epsilon']}_gamma:{current_config['gamma']}"
            
            # on those 10 seeds
            for seed in range(NUM_SEEDS):
                np.random.seed(seed)
                env = gym.make(current_config["env"], is_slippery=IS_SLIPPERY)
                total_reward, eval_return_means, eval_negative_returns, q_table = train(env, current_config)
                env.close()
                run_instance.update(
                    eval_returns=eval_return_means,
                    eval_timesteps=eval_negative_returns,
                    times=None,
                    run_data={"total_reward": total_reward, "q_table": q_table}
                )
            all_runs.append(run_instance)
            print("------------------------------------------------------------------------------")
        
        print("------------------------------------------------------------------------------")
        print("------------------------------------------------------------------------------")
        for run in all_runs:
            print(f"Run: {run.run_name}, Mean Evaluation Return: {np.mean(run.final_returns)}")
        print("------------------------------------------------------------------------------")
        print(f"The Best Run is : {get_best_saved_run(all_runs)}")
        
"""
    RESULTS : Slippery=True
    ------------------------------------------------------------------------------
    Run: Env:FrozenLake8x8-v1_eps:0.9_gamma:0.99, Mean Evaluation Return: 0.6906000000000001
    Run: Env:FrozenLake8x8-v1_eps:0.9_gamma:0.8, Mean Evaluation Return: 0.265
    ------------------------------------------------------------------------------
    The Best Run is : 0.6906000000000001 (this is Run: Env:FrozenLake8x8-v1_eps:0.9_gamma:0.99)
"""        
