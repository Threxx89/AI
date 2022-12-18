# AI for Self Driving Car

# Importing the libraries
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as Funtional
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Creating the architecture of the Neural Network

class NeuralNetwork(nn.Module):
    
    def __init__(self, input_action_size, output_action_size):
        super(NeuralNetwork, self).__init__()
        self.input_action_size = input_action_size
        self.output_action_size = output_action_size
        self.fc_input_hidden = nn.Linear(input_action_size, 30) # connection between Input Layer and Hidden layers
        self.fc_output_hidden = nn.Linear(30, output_action_size) # connection between Output Layer and Hidden layers
    
    def forward(self, state):
        x = Funtional.relu(self.fc_input_hidden(state)) #recitivy activate hidden layer
        q_values = self.fc_output_hidden(x)
        return q_values

# Implementing Experience Replay
# Based on mark decision process. look at previose states or events to learn from.
# Experience Replay looks at a series of events to learn from, long term memory

class NeuralNetworkEventMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, event_batch_size):
        #zip reshapes list {[1,2,3],[4,5,6]} if you zip it becomes {[1,4],[2,5],[3,6]}
        
        samples = zip(*random.sample(self.memory, event_batch_size))
        #Variable converts a torch tensor to tensor and gradient, used to diferenciate
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

# Implementing Deep Q Learning