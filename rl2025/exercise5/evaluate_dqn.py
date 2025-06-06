import gymnasium as gym
import pickle
from typing import List, Tuple

from rl2025.exercise5.agents import DQNModified
from rl2025.exercise5.train_dqn import play_episode, CARTPOLE_CONFIG, MOUNTAINCAR_CONFIG, SWEEP_RESULTS_FILE_MOUNTAINCAR
#SWEEP_RESULTS_FILE_CARTPOLE
from rl2025.util.result_processing import Run, get_best_saved_run

ENV = "MOUNTAINCAR" # "CARTPOLE" OR "MOUNTAINCAR"
RENDER = True
def evaluate(env: gym.Env, config, output: bool = True) -> Tuple[List[float], List[float]]:
    """
    Execute training of DQN on given environment using the provided configuration

    :param env (gym.Env): environment to train on
    :param config: configuration dictionary mapping configuration keys to values
    :param output (bool): flag whether evaluation results should be printed
    :return (Tuple[List[float], List[float]]): eval returns during training, times of evaluation
    """
    timesteps_elapsed = 0

    agent = DQN(
        action_space=env.action_space, observation_space=env.observation_space, **config
    )
    # try:
    agent.restore(config['save_filename'])
    # except:
    #     raise ValueError(f"Could not find model to load at {config['save_filename']}")

    eval_returns_all = []
    eval_times_all = []


    eval_returns = 0
    for _ in range(config["eval_episodes"]):
        _, episode_return, _ = play_episode(
            env,
            agent,
            0,
            train=False,
            explore=False,
            render=RENDER,
            max_steps=config["episode_length"],
            batch_size=config["batch_size"],
        )
        eval_returns += episode_return / config["eval_episodes"]

    return eval_returns


if __name__ == "__main__":
    if ENV == "CARTPOLE":
        CONFIG = CARTPOLE_CONFIG
        SWEEP_RESULTS_FILE = SWEEP_RESULTS_FILE_CARTPOLE
    elif ENV == "MOUNTAINCAR":
        CONFIG = MOUNTAINCAR_CONFIG
        SWEEP_RESULTS_FILE = SWEEP_RESULTS_FILE_MOUNTAINCAR
    else:
        raise(ValueError(f"Unknown environment {ENV}"))

    env = gym.make(CONFIG["env"])
    results = pickle.load(open(SWEEP_RESULTS_FILE, 'rb'))
    best_run, best_run_filename = get_best_saved_run(results)
    print(f"Best run was {best_run_filename}")
    CONFIG.update(best_run.config)
    CONFIG['save_filename'] = best_run_filename
    returns = evaluate(env, CONFIG)
    print(returns)
    env.close()
