import torch.nn as nn

class Net(nn.Module):
    def __int__(self, input_size, hidden_size, output_size):
        super(Net,self).__init__()
        self.fully_connect_layer1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.fully_connect_layer2 = nn.Linear(hidden_size,hidden_size)
        self.fully_connect_layer3 = nn.Linear(hidden_size,output_size)

    def forward(self, x):
        output = self.fully_connect_layer1(x)
        output = self.relu(output)
        output = self.fully_connect_layer2(output)
        output = self.relu(output)
        output = self.fully_connect_layer3(output)
        return  output