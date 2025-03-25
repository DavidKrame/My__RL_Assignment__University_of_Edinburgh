from abc import ABC, abstractmethod
from copy import deepcopy
import gymnasium as gym
from collections import defaultdict
import numpy as np
import os.path
from torch import Tensor
from torch.distributions.categorical import Categorical
import torch.nn
from torch.optim import Adam
from typing import Dict, Iterable, List

from rl2025.exercise3.networks import FCNetwork
from rl2025.exercise3.replay import Transition


class Agent(ABC):
    """Base class for Deep RL Exercise 3 Agents

    **DO NOT CHANGE THIS CLASS**

    :attr action_space (gym.Space): action space of used environment
    :attr observation_space (gym.Space): observation space of used environment
    :attr saveables (Dict[str, torch.nn.Module]):
        mapping from network names to PyTorch network modules

    Note:
        see https://gymnasium.farama.org/api/spaces/ for more information on Gymnasium spaces
    """

    def __init__(self, action_space: gym.Space, observation_space: gym.Space):
        """The constructor of the Agent Class

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        """
        self.action_space = action_space
        self.observation_space = observation_space

        self.saveables = {}

    def save(self, path: str, suffix: str = "") -> str:
        """Saves saveable PyTorch models under given path

        The models will be saved in directory found under given path in file "models_{suffix}.pt"
        where suffix is given by the optional parameter (by default empty string "")

        :param path (str): path to directory where to save models
        :param suffix (str, optional): suffix given to models file
        :return (str): path to file of saved models file
        """
        torch.save(self.saveables, path)
        return path

    def restore(self, save_path: str):
        """Restores PyTorch models from models file given by path

        :param save_path (str): path to file containing saved models
        """
        dirname, _ = os.path.split(os.path.abspath(__file__))
        save_path = os.path.join(dirname, save_path)
        checkpoint = torch.load(save_path)
        for k, v in self.saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())

    @abstractmethod
    def act(self, obs: np.ndarray):
        ...

    @abstractmethod
    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ...

    @abstractmethod
    def update(self):
        ...


class DQNModified(Agent):
    """From the DQN agent in 3

    :attr critics_net (FCNetwork): fully connected DQN to compute Q-value estimates
    :attr critics_target (FCNetwork): fully connected DQN target network
    :attr critics_optim (torch.optim): PyTorch optimiser for DQN critics_net
    :attr learning_rate (float): learning rate for DQN optimisation
    :attr update_counter (int): counter of updates for target network updates
    :attr target_update_freq (int): update frequency (number of iterations after which the target
        networks should be updated)
    :attr batch_size (int): size of sampled batches of experience
    :attr gamma (float): discount rate gamma
    """

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.Space,
        learning_rate: float,
        hidden_size: Iterable[int],
        target_update_freq: int,
        batch_size: int,
        gamma: float,
        epsilon_start: float,
        epsilon_min: float,
        epsilon_decay_strategy: str = "constant",
        epsilon_decay: float = None,
        exploration_fraction: float = None,
        **kwargs,
        ):
        """The constructor of the DQN agent class

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param learning_rate (float): learning rate for DQN optimisation
        :param hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected DQNs
        :param target_update_freq (int): update frequency (number of iterations after which the target
            networks should be updated)
        :param batch_size (int): size of sampled batches of experience
        :param gamma (float): discount rate gamma
        :param epsilon_start (float): initial value of epsilon for epsilon-greedy action selection
        :param epsilon_min (float): minimum value of epsilon for epsilon-greedy action selection
        :param epsilon_decay (float, optional): decay rate of epsilon for epsilon-greedy action. If not specified,
                                                epsilon will be decayed linearly from epsilon_start to epsilon_min.
        """
        super().__init__(action_space, observation_space)

        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.n

        # ######################################### #
        #  BUILD YOUR NETWORKS AND OPTIMIZERS HERE  #
        # ######################################### #
        self.critics_net = FCNetwork(
            (STATE_SIZE, *hidden_size, ACTION_SIZE), output_activation=None
            )

        self.critics_target = deepcopy(self.critics_net)

        self.critics_optim = Adam(
            self.critics_net.parameters(), lr=learning_rate, eps=1e-3
            )

        # ############################################# #
        # WRITE ANY HYPERPARAMETERS YOU MIGHT NEED HERE #
        # ############################################# #
        self.learning_rate = learning_rate
        self.update_counter = 0
        self.last_timestep = 0 ############################################################################
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min

        self.epsilon_decay_strategy = epsilon_decay_strategy
        if epsilon_decay_strategy == "constant":
            assert epsilon_decay is None, "epsilon_decay should be None for epsilon_decay_strategy == 'constant'"
            assert exploration_fraction is None, "exploration_fraction should be None for epsilon_decay_strategy == 'constant'"
            self.epsilon_exponential_decay_factor = None
            self.exploration_fraction = None
        elif self.epsilon_decay_strategy == "linear":
            assert epsilon_decay is None, "epsilon_decay is only set for epsilon_decay_strategy='exponential'"
            assert exploration_fraction is not None, "exploration_fraction must be set for epsilon_decay_strategy='linear'"
            assert exploration_fraction > 0, "exploration_fraction must be positive"
            self.epsilon_exponential_decay_factor = None
            self.exploration_fraction = exploration_fraction
        elif self.epsilon_decay_strategy == "exponential":
            assert epsilon_decay is not None, "epsilon_decay must be set for epsilon_decay_strategy='exponential'"
            assert exploration_fraction is None, "exploration_fraction is only set for epsilon_decay_strategy='linear'"
            self.epsilon_exponential_decay_factor = epsilon_decay
            self.exploration_fraction = None
        else:
            raise ValueError("epsilon_decay_strategy must be either 'linear' or 'exponential'")
        # ######################################### #
        self.saveables.update(
            {
                "critics_net"   : self.critics_net,
                "critics_target": self.critics_target,
                "critic_optim"  : self.critics_optim,
                }
            )

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**
        ** Implement both epsilon_linear_decay() and epsilon_exponential_decay() functions **
        ** You may modify the signature of these functions if you wish to pass additional arguments **

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """

        def epsilon_linear_decay(*args, **kwargs):
            ### PUT YOUR CODE HERE ###
            # raise(NotImplementedError)
            decay_steps = self.exploration_fraction * max_timestep
            if timestep < decay_steps:
                new_epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_min) * (timestep / decay_steps)
            else:
                new_epsilon = self.epsilon_min
            return new_epsilon

         # def epsilon_exponential_decay(*args, **kwargs):
        #     ### PUT YOUR CODE HERE ###
        #     # raise(NotImplementedError)
        #     ratio = timestep / max_timestep
        #     new_epsilon = self.epsilon_start * (self.epsilon_exponential_decay_factor ** ratio)
        #     if new_epsilon < self.epsilon_min:
        #         new_epsilon = self.epsilon_min
        #     return new_epsilon
        def epsilon_exponential_decay(*args, **kwargs):
            ### PUT YOUR CODE HERE ###
            # raise(NotImplementedError)
            new_epsilon = self.epsilon * (self.epsilon_exponential_decay_factor ** (timestep / max_timestep))
            new_epsilon = max(new_epsilon, self.epsilon_min)
            self.epsilon = new_epsilon
            
            return new_epsilon        

        if self.epsilon_decay_strategy == "constant":
            pass
        elif self.epsilon_decay_strategy == "linear":
            # linear decay
            ### PUT YOUR CODE HERE ###
            # self.epsilon = epsilon_linear_decay(...)
            self.epsilon = epsilon_linear_decay()
            
        elif self.epsilon_decay_strategy == "exponential":
            # exponential decay
            ### PUT YOUR CODE HERE ###
            # self.epsilon = epsilon_exponential_decay(...)
            self.epsilon = epsilon_exponential_decay()
            
        else:
            raise ValueError("epsilon_decay_strategy must be either 'constant', 'linear' or 'exponential'")


    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        When explore is False you should select the best action possible (greedy). However, during
        exploration, you should be implementing an exploration strategy (like e-greedy). Use
        schedule_hyperparameters() for any hyperparameters that you want to change over time.

        :param obs (np.ndarray): observation vector from the environment
        :param explore (bool): flag indicating whether we should explore
        :return (sample from self.action_space): action the agent should perform
        """
        ### PUT YOUR CODE HERE ###
        # raise NotImplementedError("Needed for Q3")
        
        # important conversion to tensors
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        
        self.critics_net.eval() # Eval mode
        with torch.no_grad(): # Eval mode
            q_values = self.critics_net(obs_tensor)
        
        self.critics_net.train()  # training mode back
        
        greedy_action = int(torch.argmax(q_values, dim=1).item())

        if explore:
            if np.random.rand() < self.epsilon:
                return self.action_space.sample()
            else:
                return greedy_action
        else:
            return greedy_action
        
    def update(self, batch: Transition) -> Dict[str, float]:
        """Update function for DQNModified
            # --------------------------------------------------------------------------------------------------
            We realized that classical Q-learning update rule often leads to suboptimal results, 
            requiring significant adjustments in hyperparameters to achieve satisfactory performance. 
            To address these limitations, I propose a topology-inspired correction to the update mechanism. 
            This correction is based on the mathematical analysis of the Bellman Operator and
            the study of its modifications, as detailed in my work: 
            [Topological Foundations of Reinforcement Learning](https://arxiv.org/pdf/2410.03706).

            In the classical Q-learning update, the target is computed as:
                target = reward + gamma * (1 - done) * max_next_q

            My proposed modification introduces a correction term that penalizes the discrepancy between 
            the maximum Q-value at a state and the Q-value for the selected action. 
            This correction encourages smoother Q-functions and better alignment with the underlying topology 
            of the state–action space.

            The correction term is defined (for simplification here) as:
                correction = beta * (max_current_q - current_q)

            where beta is a decaying coefficient. To simplify here we took:
                beta = 1 / ((update_counter + 2)^2)

            As training progresses, beta decreases, reducing the influence of the correction term and 
            allowing the agent to fine-tune its policy based on accumulated experience.
            
            So, in essence, this modification integrates the concept of advantage learning directly into the 
            Bellman operator. While this integration might initially seem counterintuitive, our theoretical 
            analysis in the tabular case demonstrated that it leads to a well-behaved operator. 
            Moreover, empirical implementations have confirmed that this approach yields improved results.
            # --------------------------------------------------------------------------------------------------
            
            :param batch (Transition): A batch of transitions sampled from the replay buffer.
            :return (Dict[str, float]): A dictionary mapping loss names to their corresponding values.
            
        """
        # Initialize update_counter if it doesn't exist (we'll use this to decay the correction coefficient beta)
        if not hasattr(self, 'update_counter'):
            self.update_counter = 0

        # states, actions, next_states, rewards, dones = batch
        
        # states = torch.FloatTensor(states)
        # actions = torch.LongTensor(actions).unsqueeze(1)
        # rewards = torch.FloatTensor(rewards).unsqueeze(1)
        # dones = torch.FloatTensor(dones).unsqueeze(1)
        # next_states = torch.FloatTensor(next_states)
        
        # # current Q-values for all actions using the critics network
        # current_q_values = self.critics_net(states)
        
        # current_q = current_q_values.gather(1, actions)
        
        # max_current_q, _ = torch.max(current_q_values, dim=1, keepdim=True)
        
        # Unpack the batch of transitions : "states", "actions", "next_states", "rewards", "done"
        states, actions, next_states, rewards, dones  = batch
        # Conversions
        # actions = torch.FloatTensor(actions).unsqueeze(1)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        
        # After these conversion let us compute the current Q-values using critics_net
        current_q_values = self.critics_net(states)
        actions = actions.long()
        current_q = current_q_values.gather(1, actions)
        
        # max_current_q, _ = torch.max(current_q_values, dim=1, keepdim=True)
        with torch.no_grad():
            max_current_q, _ = current_q.max(dim=1, keepdim=True)
            
            next_q_values = self.critics_target(next_states)
            max_next_q, _ = next_q_values.max(dim=1, keepdim=True)
            
        # the decaying beta now (beta decreases over time)
        beta = 1.0 / ((self.update_counter + 2) ** 2)
        
        # topological correction term according
        topo_correction = beta * (max_current_q - current_q)
        
        # My proposed target: classical target - topo_correction
        target_q = rewards + self.gamma * (1 - dones) * max_next_q - topo_correction

        q_loss = torch.mean((current_q - target_q) ** 2)
        
        self.critics_optim.zero_grad()
        q_loss.backward()
        self.critics_optim.step()

        self.update_counter += 1

        if self.update_counter % self.target_update_freq == 0:
            self.critics_target.load_state_dict(self.critics_net.state_dict())

        return {"q_loss": q_loss.item()}