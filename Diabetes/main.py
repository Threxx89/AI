import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

# Load the dataset using Pandas
data = pd.read_csv('diabetes.csv')

# For x: Extract out the dataset from all the rows (all samples) and all columns except last column (all features).
# For y: Extract out the last column (which is the label)
# Convert both to numpy using the .values method
x = data.iloc[:,0:-1].values
y_string= list(data.iloc[:,-1])

# Lets have a look some samples from our data
print(x[:3])
print(y_string[:3])

# Our neural network only understand numbers! So convert the string to labels
y_int = []
for string in y_string:
    if string == 'positive':
        y_int.append(1)
    else:
        y_int.append(0)

# Now convert to an array
y = np.array(y_int, dtype = 'float64')

# Feature Normalization. All features should have the same range of values (-1,1)
sc = StandardScaler()
x = sc.fit_transform(x)
# Now we convert the arrays to PyTorch tensors
x = torch.tensor(x)
# We add an extra dimension to convert this array to 2D
y = torch.tensor(y).unsqueeze(1)
print(x.shape)
print(y.shape)


class Dataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        # Get one item from the dataset
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


dataset = Dataset(x, y)
len(dataset)
# Load the data to your dataloader for batch processing and shuffling
train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=32,
                                           shuffle=True)
# Let's have a look at the data loader
print("There is {} batches in the dataset".format(len(train_loader)))
for (x, y) in train_loader:
    print("For one iteration (batch), there is:")
    print("Data:    {}".format(x.shape))
    print("Labels:  {}".format(y.shape))
    break


# Now let's build the above network
class Model(nn.Module):
    def __init__(self, input_features):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_features, 5)
        self.fc2 = nn.Linear(5, 4)
        self.fc3 = nn.Linear(4, 3)
        self.fc4 = nn.Linear(3, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.fc3(out)
        out = self.tanh(out)
        out = self.fc4(out)
        out = self.sigmoid(out)
        return out


# Create the network (an object of the Net class)
net = Model(x.shape[1])
# In Binary Cross Entropy: the input and output should have the same shape
# size_average = True --> the losses are averaged over observations for each minibatch
criterion = torch.nn.BCELoss(size_average=True)
# We will use SGD with momentum with a learning rate of 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
# Train the network
num_epochs = 200
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        inputs = inputs.float()
        labels = labels.float()
        # Feed Forward
        output = net(inputs)
        # Loss Calculation
        loss = criterion(output, labels)
        # Clear the gradient buffer (we don't want to accumulate gradients)
        optimizer.zero_grad()
        # Backpropagation
        loss.backward()
        # Weight Update: w <-- w - lr * gradient
        optimizer.step()

    # Accuracy
    # Since we are using a sigmoid, we will need to perform some thresholding
    output = (output > 0.5).float()
    # Accuracy: (output == labels).float().sum() / output.shape[0]
    accuracy = (output == labels).float().mean()
    # Print statistics
    print("Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}".format(epoch + 1, num_epochs, loss, accuracy))
