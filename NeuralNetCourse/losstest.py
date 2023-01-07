import torch
import torch.nn as nn
import numpy as np

prediction = torch.randn(4, 5)

# Simple of lable using the meanSquare Loss function
lable = torch.randn(4, 5)

# none sum mean
# Turn loss into single value into single value you can do mean or sum
mseLoss = nn.MSELoss(reduction='mean')

# Calculates the loss between actual values and outputs
loss = mseLoss(prediction, lable)
print(loss)
# Self implementation of mse
print(((prediction - lable) ** 2).mean())

# Binary Cross Entropy Loss BCE Loss

prediction2 = torch.randn(4, 5)

lableBCE = torch.zeros(4, 5).random_(0, 2)
print(lableBCE)

# When using BCE straight we need a sigmoid layer to convert to 0 or 1

sigmoid = nn.Sigmoid()

bceLoss = nn.BCELoss(reduction='mean')

sigmoidPred = sigmoid(prediction2)
sigmoidPredX = sigmoid(prediction2)
print(sigmoidPred)
print(sigmoidPredX)

lossBce = bceLoss(sigmoidPred, lableBCE)

print('bce')
print(lossBce)

# BCEWithLogicloss or bce with signmod include

bces = nn.BCEWithLogitsLoss(reduction='mean')

print('bces')
print(bces(prediction2, lableBCE))

# Pure Python Implementation


y = lableBCE.numpy()

x = sigmoid(prediction2)
x = x.numpy()

loss_value = []

for i in range(len(y)):
    batch_loss = []
    for j in range(len(y[0])):
        if y[i][j] == 1:
            loss = -np.log(x[i][j])
        else:
            loss = -np.log(1 - x[i][j])
        batch_loss.append(loss)

    loss_value.append(batch_loss)

print(np.mean(loss_value))