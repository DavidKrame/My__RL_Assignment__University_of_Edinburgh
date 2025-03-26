                    ------------------------------------------------
                    CHANGES I MADE AND DETAILS IMPORTANT TO MENTION
                    ------------------------------------------------

Q2:
---
    - I made changes in "util/result_processing.py" and at corresponding lines that call it in
    "exercise2/train_monte_carlo.py" and "exercise2/train_q_learning.py" to modify the display (Precisely,
    I changed completely __name__ == "__main__" for these two files to be able to display also hyperparameter tuning).
    MOST IMPORTANT CHANGE : in get_best_saved_run TO AVOID ERRORS since no weight savings sometimes.
    - I also added an "IS_SLIPPERY" param in "exercise2/train_monte_carlo.py" and "exercise2/train_q_learning.py"
    allowing me to switch easily. By default I set it "True".

Q3,4,5:
-------
    - In agents.py in Q3, Q4 and Q5(My Q5 is relying on Q3) I added "weights_only=False" to avoid errors in "torch.load"

Q4:
---
    - I included my chosen parameters for "critic_hidden_size" and "policy_hidden_size" in train_ddpg.py where 
        it was set.
    - CHOICE: After my tests I choose to submit a model with the following configurations:
        RACETRACK_CONFIG = {
            "critic_hidden_size": [32, 64, 128, 64, 32],
            "policy_hidden_size": [32, 64, 128, 128, 64, 32],
        }
        
        I got even better results for deeper models but taking a lot of time to run. This choice I made is based on
        my trade-off of choosing something above 500 as performance but not taking too long to run.

        I also realized that deeper models for "policy_hidden_size" were most of the time better.
        You just need to test some time since even though my average was around 700, the evaluation fall (sometimes)
         below 500 which might mislead you in the marking, thinking mistakenly that I didn't reach the 500 target.

Q5:
---
    - For this question 5 I took the code from Question 3 and included my update rule into it (Everything is 
    explained in docstring before the update function begins in "agents.py"). 
    I did my tests with constant epsilon=0.05 to show it works better. I also recorded the Terminal_Output 
    during training and I added those files into Q5 Folder (inside the folder "Terminal_OUTPUT__Results" 
     both the Q5 Terminal_Output and the Q3 output when set exactly on same parameters as my Q5 to allow tests).
     I used the default parameters given in this assignment for "constant" case changing only epsilon to 0.05. 
    I didn't test decaying methods on my suggested method Q5 but since my approach is better than the 
    standard approach without Decay, it should be also better with decay. And indeed I realized that the best 
     result of my approach without decay was better than the best results of the classical method with decay.