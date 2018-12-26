[//]: # (Image References)

[image1]: https://s3.amazonaws.com/video.udacity-data.com/topher/2018/June/5b1ea778_reacher/reacher.gif "Trained Agent"

# Learning to Reach: Policy-based optimization for a single-armed agent
[Unity ML-Agents](https://unity3d.com/machine-learning) + [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/abs/1509.02971) using [PyTorch](https://pytorch.org/).

Project on use of policy-based methods for continuous control for partial fulfillment of [Udacity's Deep Reinforcement Learning Nanodegree.](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)

### Background

See attached ```report.pdf``` for background information on reinforcement learning and the DDPG architecture utilized in this repository. DDPG utilizes an "Actor" (which learns which actions to take) and a "Critic" (which informs the Actor on the value of these actions)

### Introduction

The goal of the Reacher agent is to learn to control its arm to reach a sphere moving around the agent

![Trained Agent][image1]

In this modified version of [ML-Agent's Reacher example](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md) provided by Udacity, the agent is rewarded (+0.1 score) every time its hand is inside the moving sphere. The Reacher agent can observe 33 variables (e.g., position, rotation, velocity, angular velocity, as well as the position of the sphere).

In this training environment, we will train 20 Reacher agents in parallel. The environment is considered solved if we can train our agents to achieve a net score of +30 over 100 consecutive episodes.

Each Reacher agent's arm consists of two joints. For each joint, two torque values need to be controlled whose values can range from -1 to 1 (total action space = 4 dimensions).

### Getting Started
1. [Download the Anaconda Python Distribution](https://www.anaconda.com/download/)

2. Once installed, open the Anaconda command prompt window and create a new conda environment that will encapsulate all the required dependencies. Replace **"desired_name_here"** with whatever name you'd like. Python 3.6 is required.

    `conda create --name "desired_name_here" python=3.6`  
    `activate "desired_name_here"`

3. Clone this repository on your local machine.

    `git clone https://github.com/ShortFox/Learning-to-Reach.git`  

4. Navigate to the `python/` subdirectory in this repository and install all required dependencies

    `cd Learning-to-Reach/python`  
    `pip install .`  

    **Windows Users:** When installing the required dependencies, you may receive an error when trying to install "torch." If that is the case, then install pytorch manually, and then run `pip install .` again. Note that pytorch version 0.4.0 is required:

    `conda install pytorch=0.4.0 -c pytorch`

5. Download the Reacher Unity environment from one of the links below.  Note you do not need to have Unity installed for this repository to function.

    Download the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

6. Place the unzipped file inside the repository location. Note that the Windows (x86_64) version of the environment is already included (must be unzipped).

### Instructions to train the Reacher agent

1. Open `main.py` and edit the `file_name` variable to correctly locate the location of the Banana Man Unity environment.

2. Within the same anaconda prompt, return to the `Learning-to-Reach` subfolder, and then train the agent:

    `cd ..`  
    `python main.py`

3. Following training, `DDPG_Actor.pth` and `DDPG_Critic.pth` will be created which represents the neural network weights for the Actor and Critic, respectively. Additionally, `reacher_ddpg_scores.csv` will contain the scores from training (with the row indicating episode number). `reacher_ddpg_average.csv` is the average values of `reacher_ddpg_scores.csv`.

4. To watch your trained agent, first open `main.py` and replace the `train()` function call with `train()`. By default, `train()` will load the `DDPG_Actor.pth` and `DDPG_Critic.pth` files.

5. Then, execute `main.py` again:

    `python main.py`

### Interested to make your own changes?

- `model.py` - defines the neural network architecture for the Actor and Critic.
- `ddpg_agent.py` - defines the agent training program. Contains the classes `Agent()`, which defines the agent's behavior and neural network update rules. Also contains the `save` and `load` functions to save/load the network's weights.
- `memory.py` - contains the class `ReplayBuffer()`, which keeps track of the agent's state-action-reward experiences in memory for randomized batch learning for the neural network.
- `noise.py` - simple noise class that adds random variation to the Reacher agent's movements to improve exploration.
- `main.py` - simple script defining the Unity environment and command to run the Reacher environment.

See comments in associated files for more details regarding specific functions/variables.
