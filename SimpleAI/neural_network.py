
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self,number_of_input, number_of_output):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(number_of_input, 14)
        self.linear2 = nn.Linear(14, number_of_output)


    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x
