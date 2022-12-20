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

class DeepQLearning():
    
    def __init__(self, input_action_size, output_action_size, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.neural_network_model = NeuralNetwork(input_action_size, output_action_size)
        self.event_memory = NeuralNetworkEventMemory(100000)
        self.optimizer = optim.Adam(self.neural_network_model.parameters(), lr = 0.001) #lr = learn_rate
        self.last_state = torch.Tensor(input_action_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
    
    def select_action(self, state):
        probs = Funtional.softmax(self.neural_network_model(Variable(state, volatile = True))*100) # T=100
        action = probs.multinomial(num_samples=1)
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.neural_network_model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.neural_network_model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = Funtional.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_graph = True)
        self.optimizer.step()
    
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.event_memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.event_memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.event_memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    def save(self):
        torch.save({'state_dict': self.neural_network_model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.neural_network_model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")