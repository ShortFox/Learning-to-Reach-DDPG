import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic
from memory import ReplayBuffer
from noise import OUNoise

import torch
import torch.nn.functional as F
import torch.optim as optim

EPISODES = 200          # total number of episodic experiences
BATCH_SIZE = 128        # minibatch size
BUFFER_SIZE = int(1e5)  # replay buffer size
GAMMA = 0.99            # discount factor
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
TAU = 1e-3              # for soft update of target parameters
WEIGHT_DECAY = 0        # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment. Code adapted from Udacity Deep Reinforcement Learning Nanodegree (http://www.udacity.com)"""
    
    def __init__(self, environment, agent_name,train_agent, random_seed):
        """Initialize an Agent object.

        Params
        ======
            environment (UnityEnvironment): Agent's environment
            agent_name: The name of the agent's "brain" (a Unity ML
            Agent's construct)
            train_agent (bool): Determine if agent will be trained, or if
            model weights will be loaded.
            random_seed (int): random_seed
        """
        
        #Set Environment
        self.env = environment

        #Set Brain and name
        self.brain_name = agent_name
        self.brain = self.env.brains[self.brain_name]

        #Set model to train model, or to load previous model.
        self.train_agent = train_agent

        #Set random seed.
        self.seed = random.seed(random_seed)

        #Get environment information.
        self.env_info = self.env.reset(train_mode=self.train_agent)[self.brain_name]
        self.agent_length = len(self.env_info.agents)
        self.state_size = len(self.env_info.vector_observations[0])
        self.action_size = self.brain.vector_action_space_size
        
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(self.state_size, self.action_size, random_seed).to(device)
        self.actor_target = Actor(self.state_size, self.action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(self.state_size, self.action_size, random_seed).to(device)
        self.critic_target = Critic(self.state_size, self.action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise((self.agent_length, self.action_size), random_seed)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed, device)
    
    def run(self):
        #If training set to false, load previous model weights.
        if self.train_agent == False:
            self.load(self.actor_local, 'DDPG_Actor.pth')
            self.load(self.critic_local, 'DDPG_Critic.pth')
            
        scores = []
        scores_window = deque(maxlen=100)
        
        for i_episode in range(1,EPISODES+1):
            
            #Get environment information and current state.
            self.env_info = self.env.reset(train_mode=self.train_agent)[self.brain_name]
            states = self.env_info.vector_observations
            
            #Update mean of noise distribution.
            self.reset()
            score = np.zeros(self.agent_length)
            
            while True:
                
                #Given state, select action
                actions = self.act(states,add_noise=self.train_agent)
                
                #Get next state information, rewards received from action and whether the episode ended
                self.env_info = self.env.step(actions)[self.brain_name]
                next_states = self.env_info.vector_observations
                rewards = self.env_info.rewards
                dones = self.env_info.local_done
                
                #If agent set to training mode, take a step. This will add experience to memory and perform a learning iteration.
                if self.train_agent:
                    self.step(states,actions,rewards, next_states,dones)
                
                #Update score, and set next_state to be the new state.
                score += rewards
                states = next_states
                
                if np.any(dones):
                    break
                    
            score = np.transpose(score)        
            scores.append(score)
            scores_window.append(np.mean(score))
            
            #Print statements, save score and model weights periodically.
            print('Episode', i_episode, 'Score:', np.mean(scores[-1]), end='\r')
            if (i_episode)%10 == 0:
                self.save_scores(scores, scores_window)
                if self.train_agent:
                    self.save(self.actor_local, 'DDPG_Actor.pth')
                    self.save(self.critic_local, 'DDPG_Critic.pth')
            if i_episode >= 100:
                # Display most recent score every 100 episodes.
                if (i_episode)%100 ==0:
                    print("Episode: {0:d}, Average score {1:f}".format(i_episode,np.mean(scores_window)))  
                #If environment is solved early, break
                if np.mean(scores_window) >= 30:
                    break
        self.save_scores(scores,scores_window)
        if self.train_agent:
            print("Saving final model...")
            self.save(self.actor_local, 'DDPG_Actor.pth')
            self.save(self.critic_local, 'DDPG_Critic.pth')

    def save_scores(self,output, output_average):
        #Save scores to a .csv file
        np.savetxt('reacher_ddpg_scores.csv',output, delimiter=',')
        np.savetxt('reacher_ddpg_average.csv',output_average,delimiter=',')
        
    def save(self, network, file_name):
        """Save model parameters to file"""
        torch.save(network.state_dict(), file_name)

    def load(self, network, file_name):
        """If possible, load model parameters from file (extension .pth)"""
        try:
            state_dict = torch.load(file_name)
        except FileNotFoundError as err:
            raise Exception('No file named ',file_name,' was found')
        else:
            network.load_state_dict(state_dict)
          
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward for each agent seperately.
        for i in range(self.agent_length):
            self.memory.add(state[i,:], action[i,:], reward[i], next_state[i,:], done[i])

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to(device)
        self.actor_local.eval()
        
        #Obtain action given state input.
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train(mode = self.train_agent)
        
        #Add noise to action.
        if add_noise:
            actions += self.noise.sample()
        #Clip actions to be within appropriate range.
        return np.clip(actions, -1, 1)

    def reset(self):
        """Updates mean of noise distribution"""
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)