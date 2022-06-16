# Anomaly Detection with Reinforcement Learning

## Intro

The anomaly detection problem cab be modeled as a markov decision process (MDP) in RL.
An MPD is represented by $\langle S, A, P_a, R_a, \gamma \rangle$, where:
- $S$ is the set of environment states (at $t$ the system is in $s_t \in S$)
- $A$ is the set of actions (at $t$ the agents chooses the action $a_t \in A$)
- $P_a$ is the probability to do the action $a$ which leads to the state $s_{t+1}$ $\rightarrow$ $P_a = P(s_{t+1}| S=s_t, A=a_t)$
- $R_a$ is the instant reward received by the agent from taking the action $a$
- $\gamma \in [0,1]$ is the discount factor

The agent tries to learn the policy $\pi:S \to A$ that maximizes the future reward $\sum_t^{T_\infty} \gamma^t R_t$.

---

## RL Elements Definitions

Time series anomaly detection could be considered as an MDP
because the decision of normal or abnormal at the current time step will change the environment by whether it triggers an anomaly detection or not. And the next decision will be influenced by the changing environment. 

- **State**: a single state is a time series with $n$ time steps [$s_i, \dots, s_{i+t}$] with an associated label 0/1 that represents a normal/abnormal behaviour. Maybe it is also possible to construct a second part of the state $s_{actions} = [a_i, \dots, a_{i+n}]$ which collects the sequence of previous actions.  
In our case, the idea is for instance to aggregate the dataset in 1 minute timestamps, and then to create time series of 1 hour of duration.  
    - *How to define the label? Either use timeseries of Sunday [9:00-16:00] as anomalous or just look at the last timestamp if it is in the givern range* 
    - *Or we can assign a binary label to each timestamp of the timeseries*
- **Action**: the action space is $A=\{0,1\}$, where 0 is normal and 1 is anomaly.
- **Reward**: 
$$
R(s,a) = 
\begin{cases}
     A & \text{if action is TP} \\
    -B & \text{if action is FP} \\
    -C & \text{if action is FN} \\
     D & \text{if action is TN} \\
\end{cases}
$$

---

## DQN Algorithm

Points are taken from slide 17 (`ML4MobCom_04c`).

1. *Preprocess and feed the state s to our DQN, which will return the Q-values of all possible actions in the state*
    - state *s* is a time series
    - DQN is LSTM Network with binary prob output $\to$ it returns Q-values for the 2 possible actions (or a binary value for each timestamp?)
2. *Select an action using the epsilon-greedy policy*
    - Actions are $\{0,1\}$: with probability $1-\epsilon$ choose max Q-value from DQN, otherwise random
3. *Perform action in s and move to s' to receive reward. Store transition $\langle s, a, r, s' \rangle$ to replay buffer*.
    - how do I select next state *s'* given *a*, since our states are time series?
        - Do I need to calculate a sort of similarity between them?
        - if action is 1 go to similar ts, otherwise to the most different? Need to create pseudo-labels to decide where to move?
        - need ton populate replay buffer before starting RL algorithm (maybe using other methods - `luminol`/`prophet`)?
4. *sample some random batches of transitions from the replay buffer and calculate the loss*
    - target DQN is copy of DQN but updated less frequent
    - need to pre-populate replay buffer (otherwise in first episodes I can't sample batches)?
5. *gradient descent with respect to our actual network parameters*
    - update DQN
6. *After every N iterations, copy our actual network weights to the target network weights*
    - update target DQN copying DQN state every N iters
7. *Repeat these steps for M number of episodes*

---

## References

- ***Papers***:
    - [Toward Deep Supervised Anomaly Detection: Reinforcement Learning from Partially Labeled Anomaly Data](https://arxiv.org/abs/2009.06847) - [Unofficial DPLAN implementation](https://github.com/lflfdxfn/DPLAN-Implementation)
    - [Policy-based reinforcement learning for time series anomaly detection](https://www.sciencedirect.com/science/article/pii/S0952197620302499)
    - [RLAD: Time Series Anomaly Detection through Reinforcement Learning and Active Learning](https://arxiv.org/abs/2104.00543)
- ***Code Examples***:
    - [Keras-rl - Deep Reinforcement Learning for Keras](https://github.com/keras-rl/keras-rl)
    - [Introduction to Reinforcement Learning (RL) in PyTorch](https://medium.com/analytics-vidhya/introduction-to-reinforcement-learning-rl-in-pytorch-c0862989cc0e)
    - [Deep Q-network with Pytorch and Gym to solve the Acrobot game](https://towardsdatascience.com/deep-q-network-with-pytorch-and-gym-to-solve-acrobot-game-d677836bda9b)