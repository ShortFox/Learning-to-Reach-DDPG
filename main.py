from ddpg_agent import Agent
from unityagents import UnityEnvironment
import numpy as np


#Uncomment version that matches your operating system.

#Mac:
    #filename = "Banana.app"
#Windows (x86):
    #file_name = Reacher_Windows_x86/Reacher.exe
#Windows (x86_64):
file_name = "Reacher_Windows_x86_64/Reacher.exe"
#Linux (x86):
    #file_name = "Reacher_Linux/Reacher.x86"
#Linux (x86_64)
    #file_name = "Reacher_Linux/Reacher.x86_64"



#Define the Unity environment
env = UnityEnvironment(file_name)

#Get the "brain" name of Reachers
brain_name = env.brain_names[0]

#Create Agent, Run Training, and then Close Environment.
def train():
    agent = Agent(environment = env, agent_name = brain_name, train_agent = True, random_seed = 0)
    agent.run()
    env.close()

#This function loads previous training weights and plays trained agent.
def watch():
    agent = Agent(environment = env, agent_name = brain_name, train_agent = False, random_seed = 0)
    agent.run()
    env.close()
    
train()