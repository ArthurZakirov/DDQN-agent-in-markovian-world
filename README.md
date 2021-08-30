# Reinforcement Learning

<img src="https://www.kdnuggets.com/images/reinforcement-learning-fig1-700.jpg" width="600" height="230">

---

### Table of Contents
You're sections headers will be used to reference location of destination.

- [Description](#description)
- [How To Use](#how-to-use)
- [References](#references)
- [License](#license)
- [Author Info](#author-info)

---

## Description

This project contains several code implementations of current research papers in Reinforcement Learning and Deep Reinforcement Learning. 

#### World
![MDP](https://artint.info/2e/html/x438.png)

<img src="https://static.packt-cdn.com/products/9781838649777/graphics/494d0f6c-dc6c-4851-81ef-5a2756e178ec.png" width="400" height="50">
The world is implemented as a deterministic Markov Decison Process characterized by an adjustable number of states S and actions A. Transition probabilities and rewards are created randomly. Model-based methods such as Policy-Iteration and Value Iteration are implemented to calculate the optimal policy. This allows to precisely evaluate the later implemented model-free RL methods. The Agents policy can be plugged into the Bellmann equation of the world to see how close it is to optimal.

#### Q-Table Agent
In order to demonstrate the effectivenes of Deep Neural Networks in RL, the following Methods are implemented without a Q-function for comparison. Instead they rely on a table of Q values, which is not scalable but reasonable for the model choice in this project. 
The following methods are compared:
- Monte Carlo (MC)
- Temporal Difference (TD)
- Q-Learning (DQL)

#### DQN Agent
<img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2019/04/Screenshot-2019-04-16-at-5.46.01-PM.png" width="400" height="250">
Instead of relying on a Table, in many scenarios it is more feasable to use a Q function. In this project multiple variants of RL Agents using Deep neural Networks as a Q function are compared:

- Monte Carlo (MC)
- Temporal Difference (TD)
- Q-Learning (DQL)

Also the following research papers are implemented in order to improve Q-Learning
-  Double Q Learning (https://arxiv.org/abs/1509.06461)
-  Prioritized Experience Replay (https://arxiv.org/abs/1511.05952)

---

## How To Use

#### Installation
