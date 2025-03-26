
############################################################################################################
##########################            RL2023 Assignment Answer Sheet              ##########################
############################################################################################################

# **PROVIDE YOUR ANSWERS TO THE ASSIGNMENT QUESTIONS IN THE FUNCTIONS BELOW.**

############################################################################################################
# Question 2
############################################################################################################

def question2_1() -> str:
    """
    (Multiple choice question):
    For the Q-learning algorithm, which value of gamma leads to the best average evaluation return?
    a) 0.99
    b) 0.8
    return: (str): your answer as a string. accepted strings: "a" or "b"
    """
    answer = "a"  # TYPE YOUR ANSWER HERE "a" or "b"
    return answer


def question2_2() -> str:
    """
    (Multiple choice question):
    For the Every-visit Monte Carlo algorithm, which value of gamma leads to the best average evaluation return?
    a) 0.99
    b) 0.8
    return: (str): your answer as a string. accepted strings: "a" or "b"
    """
    answer = "a"  # TYPE YOUR ANSWER HERE "a" or "b"
    return answer


def question2_3() -> str:
    """
    (Multiple choice question):
    Between the two algorithms (Q-Learning and Every-Visit MC), whose average evaluation return is impacted by gamma in
    a greater way?
    a) Q-Learning
    b) Every-Visit Monte Carlo
    return: (str): your answer as a string. accepted strings: "a" or "b"
    """
    answer = "b"  # TYPE YOUR ANSWER HERE "a" or "b"
    return answer


def question2_4() -> str:
    """
    (Short answer question):
    Provide a short explanation (<100 words) as to why the value of gamma affects more the evaluation returns achieved
    by [Q-learning / Every-Visit Monte Carlo] when compared to the other algorithm.
    return: answer (str): your answer as a string (100 words max)
    """
    answer = "I think the change impacts Monte Carlo every-visit more because it is applied over the entire episode, which influences the aggregated return significantly more than in Q-learning. Changing gamma as we did for example, in Monte Carlo methods even a slight adjustment alters the cumulative sum of rewards by affecting every step in the trajectory. But, in Q-learning, gamma applies only to the immediate next states maximum Q-value, limiting its influence to just one lookahead step."  # TYPE YOUR ANSWER HERE (100 words max)
    return answer

def question2_5() -> str:
    """
    (Short answer question):
    Provide a short explanation (<100 words) on the differences between the non-slippery and the slippery varian of the problem.
    by [Q-learning / Every-Visit Monte Carlo].
    return: answer (str): your answer as a string (100 words max)
    """
    answer = "In the FrozenLake environment we used, the slippery version introduces randomness in the moves (meaning the agent might not act as intended or go in the intended direction) while the non-slippery version is deterministic. Consequently, the slippery version is more challenging. Our experiments revealed that Q-Learning handles randomness well, but surprisingly Monte Carlo outperforms it in the predictable (much simpler, non-slippery) setup, where Q-Learning sometimes got stuck in suboptimal moves (due to its max choice probably)."  # TYPE YOUR ANSWER HERE (100 words max)
    return answer


############################################################################################################
# Question 3
############################################################################################################

def question3_1() -> str:
    """
    (Multiple choice question):
    In the DiscreteRL algorithm, which learning rate achieves the highest mean returns at the end of training?
    a) 2e-2
    b) 2e-3
    c) 2e-4
    return: (str): your answer as a string. accepted strings: "a", "b" or "c"
    """
    answer = "a"  # TYPE YOUR ANSWER HERE "a", "b" or "c"
    return answer


def question3_2() -> str:
    """
    (Multiple choice question):
    When training DQN using a linear decay strategy for epsilon, which exploration fraction achieves the highest mean
    returns at the end of training?
    a) 0.99
    b) 0.75
    c) 0.01
    return: (str): your answer as a string. accepted strings: "a", "b" or "c"
    """
    answer = "c"  # TYPE YOUR ANSWER HERE "a", "b" or "c"
    return answer


def question3_3() -> str:
    """
    (Multiple choice question):
    When training DQN using an exponential decay strategy for epsilon, which epsilon decay achieves the highest
    mean returns at the end of training?
    a) 1.0
    b) 0.5
    c) 1e-5
    return: (str): your answer as a string. accepted strings: "a", "b" or "c"
    """
    answer = "c"  # TYPE YOUR ANSWER HERE "a", "b" or "c"
    return answer


def question3_4() -> str:
    """
    (Multiple choice question):
    What would the value of epsilon be at the end of training when employing an exponential decay strategy
    with epsilon decay set to 1.0?
    a) 0.0
    b) 1.0
    c) epsilon_min
    d) approximately 0.0057
    e) it depends on the number of training timesteps
    return: (str): your answer as a string. accepted strings: "a", "b", "c", "d" or "e"
    """
    answer = "f:Unchanged:epsilon_start"  # TYPE YOUR ANSWER HERE "a", "b", "c", "d" or "e"
    return answer


def question3_5() -> str:
    """
    (Multiple choice question):
    What would the value of epsilon be at the end of  training when employing an exponential decay strategy
    with epsilon decay set to 0.95?
    a) 0.95
    b) 1.0
    c) epsilon_min
    d) approximately 0.0014
    e) it depends on the number of training timesteps
    return: (str): your answer as a string. accepted strings: "a", "b", "c", "d" or "e"
    """
    answer = "e"  # TYPE YOUR ANSWER HERE "a", "b", "c", "d" or "e"
    return answer


def question3_6() -> str:
    """
    (Short answer question):
    Based on your answer to question3_5(), briefly  explain why a decay strategy based on an exploration fraction
    parameter (such as in the linear decay strategy you implemented) may be more generally applicable across
    different environments  than a decay strategy based on a decay rate parameter (such as in the exponential decay
    strategy you implemented).
    return: answer (str): your answer as a string (100 words max)
    """
    answer = "I think linear decay is more adaptable to different environments because its decay is more closely tied to the maximum training duration than exponential decay, which is based on a fixed rate that controls much of the process and can decay more quickly or slowly just by setting that rate, regardless of the total training time."  # TYPE YOUR ANSWER HERE (100 words max)
    return answer


def question3_7() -> str:
    """
    (Short answer question):
    In DQN, explain why the loss is not behaving as in typical supervised learning approaches
    (where we usually see a fairly steady decrease of the loss throughout training)
    return: answer (str): your answer as a string (150 words max)
    """
    answer = "In DQN, we saw, the loss function measures the difference between predicted and target Q-values. Unlike supervised learning, where loss just decreases steadily due to a fixed dataset, DQNs loss fluctuates because the agent continuously gathers new experiences and in our case a replay buffer was used, leading to a very non-stationary dataset, with this constant sampling. Additionally, as suggested by the DQN paper, the target networks in DQN update periodically, not every time, causing shifts in target values and contributing to loss variability. Thus, the loss patterns don't exhibit the steady decline seen in supervised learning scenarios and, in fact, don't have the same significance. In RL, these kind of losses serves almost just as a guide before a new target update."  # TYPE YOUR ANSWER HERE (150 words max)
    return answer


def question3_8() -> str:
    """
    (Short answer question):
    Provide an explanation for the spikes which can be observed at regular intervals throughout
    the DQN training process.
    return: answer (str): your answer as a string (100 words max)
    """
    answer = "The spikes in the image (see the assignment sheet), which occur at almost regular intervals during DQN training, are due to the periodic updates of the target network. In DQN, the target network is updated less frequently than the main Q-network to stabilize learning, and this leads to those spikes."  # TYPE YOUR ANSWER HERE (100 words max)
    return answer


############################################################################################################
# Question 5
############################################################################################################

def question5_1() -> str:
    """
    (Short answer question):
    Provide a short description (200 words max) describing your hyperparameter turning and scheduling process to get
    the best performance of your agents
    return: answer (str): your answer as a string (200 words max)
    """
    # answer = """
    #     Wanting to improve the mean final scores in Q3, I made changes in the Bellman update of the DQN algorithm.
    #     The changes are inspired by the theoretical analysis I did in point 5.1.3 of my document on the 
    #     "Topological Foundations of Reinforcement Learning" (https://arxiv.org/pdf/2410.03706). In that
    #     document I was just exploring well-known results but in a much more mathematically clear way before
    #      going beyond but without neural-networks like for this assignment.
    #     I introduced a topological correction term to penalize the discrepancy between the expected 
    #     Q-value and the Q-value corresponding to the chosen action. For the particular case of this assignment,
    #     I defined this term as:correction =  beta x (max_current_q - current_q), with  beta decaying as 
    #      1/((iteration + 2)^2).
    #     See the function description in the code but the idea is to study near-optimal but simple Bellman
    #      operators theoretically (proof of convergence) and then say something empirically too. 
    #     I did empirical tests here too (using a constant epsilon of 0.05 over 10 trials) and got a mean final score 
    #      of -121.11 ± 4.13, compared to -131.11 ± 3.93 for the classical DQN, that we got in Q3, 
    #     demonstrating both improved performance and enhanced training stability (see files in Q5 folder).
    # """  # TYPE YOUR ANSWER HERE (200 words max)
    answer = "Wanting to improve the mean final scores in Q3, I made changes in the Bellman Operator in general. The changes are inspired by the theoretical analysis I did in point 5.1.3 of my work on the 'Topological Foundations of Reinforcement Learning' (https://arxiv.org/pdf/2410.03706). In that document I was just exploring well-known results but in a much more mathematically clear way before going beyond but without neural-networks like for this assignment. I introduced a topological correction term to penalize the discrepancy between the expected Q-value and the Q-value corresponding to the chosen action. For the particular case of this assignment, I defined this term as: correction =  beta x (max_current_q - current_q), with  beta decaying as 1/((iteration + 2)^2). See the function description in the code but the idea is to study near-optimal and simple Bellman Operators theoretically (proof of convergence) and then say something empirically too. I did empirical tests here too (using a constant epsilon of 0.05 over 10 trials) and got a mean final score of -121.11 (err=4.13), compared to -131.11 (err=3.93) for the classical DQN, demonstrating both improved performance and enhanced training stability (see terminal output in Q5 folder)."
    
    return answer
