import torch
import torch.nn as nn

# create layer

layer = nn.Linear(5, 5)

# Access weights
print(layer.weight.data)

# initialise weight between 0 and 3
nn.init.uniform_(tensor=layer.weight.data, a=0, b=3)
print(layer.weight.data)
# normal distribution weight initialisation adjust value between 0 and 1
nn.init.normal_(tensor=layer.weight.data,mean=0.0,std=0.2)
print(layer.weight.data)
# initialize weight for constant don't do it for weight but baises

nn.init.constant_(tensor=layer.bias,val=8.0)

# automatically set zerios
nn.init.zeros_(layer.bias)


# Xavier initialize from a uniform distribution

nn.init.xavier_uniform_(layer.weight.data,gain=1.0)
print(layer.weight.data)