#  OpenAI Gym's LunarLander-v3 Implementation
Authors: \<[Joshua Ha](https://github.com/UserIsBlank)\>

## Overview

This project implements a Deep Q-Learning (DQN) algorithm to solve the LunarLander-v3 environment from OpenAI's Gymnasium. The goal is to train an AI agent to successfully land a lunar module on the moon's surface using reinforcement learning. The project demonstrates the application of deep reinforcement learning techniques, including experience replay, target networks, and epsilon-greedy exploration.

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

## Installation

To run this project locally, follow these steps:

1. Clone the Repository:
  git clone https://github.com/UserIsBlank/lunar-lander.git
  cd deep-q-learning-lunar-landing
2. Install Dependencies:
Ensure you have Python 3.8 or higher installed. Create a virtual environment to install the required dependencies:
  pip install -r requirements.txt
3. Run the Training Script:
  python3 train.py

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
- Matplotlib: Used for visualizing training progress.
- ImageIO: Used to create a video of the agent's performance.

---

## Usage

### Training the Agent

The training process is handled by the `train.py` script. The script initializes the environment, creates the DQN agent, and trains it for 2000 episodes. The training progress is printed to the console, showing the average score over the last 100 episodes.

### Visualizing the Results

After training, you can visualize the agent's performance by running the `visualize.py` script. This script renders a video of the lunar landing using the trained agent.

---

## Results

### Training Progress

During training, the agent's performance is monitored by calculating the average score over the last 100 episodes. The environment is considered solved when the agent achieves an average score of 200 or higher.

Example output during training:
Episode 100	Average Score: -162.92
Episode 200	Average Score: -103.80
Episode 300	Average Score: -41.63
Episode 400	Average Score: -43.84
Episode 500	Average Score: -14.88
Episode 600	Average Score: 106.19
Episode 691	Average Score: 200.03
Environment solved in 591 episodes!	Average Score: 200.03


### Visualizing the Agent's Performance

After training, the agent's performance can be visualized by rendering a video of the lunar landing. The video shows the agent successfully landing the lunar module on the landing pad.

---

## Future Improvements

This project can be extended in several ways:

1. Hyperparameter Tuning: Experiment with different hyperparameters (e.g., learning rate, discount factor) to improve training efficiency.
2. Advanced Algorithms: Implement advanced reinforcement learning algorithms such as Double DQN, Dueling DQN, or A3C.
3. Continuous Action Space: Adapt the algorithm to work with environments that have a continuous action space.
4. Parallel Training: Use parallel training techniques to speed up the training process.
5. Deployment: Deploy the trained model to a real-world application or simulation.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Conclusion

This project demonstrates my ability to implement and train a deep reinforcement learning agent using PyTorch and Gymnasium. It showcases my understanding of key reinforcement learning concepts, including Q-learning, experience replay, and neural network approximation. The modular code structure and clear documentation make it easy to extend and adapt for other reinforcement learning tasks.

Feel free to explore the code, experiment with different parameters, and contribute to the project! If you have any questions or feedback, please don't hesitate to reach out.

---

>GitHub Repository: https://github.com/UserIsBlank/lunar-lander
>LinkedIn: [https://www.linkedin.com/in/your-profile](https://www.linkedin.com/in/joshua-ha-805879280/)
>Email: josh4329ln@gmail.com

