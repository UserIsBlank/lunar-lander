#  OpenAI Gym's LunarLander-v3 Implementation
Authors: \<[Joshua Ha](https://github.com/UserIsBlank)\>

## Overview

This project implements a Deep Q-Learning (DQN) algorithm to solve the LunarLander-v3 environment from OpenAI's Gymnasium. The goal is to train an AI agent to successfully land a lunar module on the moon's surface using reinforcement learning. The project demonstrates the application of deep reinforcement learning techniques, including experience replay, target networks, and epsilon-greedy exploration.

---

## Table of Contents

1. Project Description
2. Key Features
3. Technologies Used
4. Installation
5. Usage
6. Code Structure
7. Results
8. Future Improvements
9. License

---

## Project Description

The LunarLander-v3 environment simulates a lunar landing scenario where the agent must control a spacecraft to land safely on a landing pad. The agent receives observations about the spacecraft's position, velocity, and orientation, and it must choose between four discrete actions (do nothing, fire left engine, fire main engine, fire right engine) to achieve a safe landing.

This project uses Deep Q-Learning, a reinforcement learning algorithm that combines Q-learning with deep neural networks. The key components of the implementation include:

- Neural Network: A feedforward neural network approximates the Q-value function.
- Experience Replay: A replay buffer stores past experiences to break correlations between consecutive updates.
- Target Network: A separate target network is used to stabilize training.
- Epsilon-Greedy Exploration: The agent balances exploration and exploitation during training.

The agent is trained for 2000 episodes, and the training process stops early if the agent achieves an average score of 200 over 100 consecutive episodes, indicating that the environment has been solved.

---

## Key Features

- Deep Q-Learning Implementation: A complete implementation of the DQN algorithm, including experience replay and target networks.
- Modular Code: The code is organized into classes for the neural network, replay memory, and agent, making it easy to extend and reuse.
- Training Visualization: The training progress is visualized by printing the average score over 100 episodes.
- Result Visualization: After training, the agent's performance is visualized by rendering a video of the lunar landing.

---

## Technologies Used

- Python: The primary programming language used for the project.
- PyTorch: A deep learning framework used to build and train the neural network.
- Gymnasium: A toolkit for developing and comparing reinforcement learning algorithms.
- NumPy: A library for numerical computations.
- Matplotlib: Used for visualizing training progress (optional).
- ImageIO: Used to create a video of the agent's performance.

---

## Installation

To run this project locally, follow these steps:

1. Clone the Repository:
  git clone https://github.com/UserIsBlank/lunar-lander.git
  cd deep-q-learning-lunar-landing

2. Install Dependencies:
Ensure you have Python 3.8 or higher installed. Create a virtual environmentn an install the required packages:
  pip install -r requirements.txt

3. Run the Training Script:
  python3 train.py
