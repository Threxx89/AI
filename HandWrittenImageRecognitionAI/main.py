import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable

from NeuralNet import Net

# input + output /2
input_size = 784
hidden_size = 400
output_size = 10
epochs = 10
batch_size = 100
learning_rate = 0.0001

# root location of where you want to store image
training_dataset = datasets.MNIST(root='./data',
                                  train=True,
                                  transform=transforms.ToTensor(),
                                  download=True)

test_dataset = datasets.MNIST(root='./data',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True)

# Make Data Iterable by loading it into dataloader
training_data_loader = torch.utils.data.DataLoader(dataset=training_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)

# Create an object of the class, which represents our network
net = Net(input_size, hidden_size, output_size)
CUDA = torch.cuda.is_available()
if CUDA:
    net = net.cuda()
# The loss function. The Cross Entropy loss comes along with Softmax. Therefore, no need to specify Softmax as well
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# Train the network
for epoch in range(epochs):
    correct_train = 0
    running_loss = 0
    for i, (images, labels) in enumerate(training_data_loader):
        # Flatten the image from size (batch,1,28,28) --> (100,1,28,28) where 1 represents the number of channels (grayscale-->1),
        # to size (100,784) and wrap it in a variable
        images = images.view(-1, 28 * 28)
        if CUDA:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        correct_train += (predicted == labels).sum()
        loss = criterion(outputs, labels)  # Difference between the actual and predicted (loss function)
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()  # Backpropagation
        optimizer.step()  # Update the weights

    print('Epoch [{}/{}], Training Loss: {:.3f}, Training Accuracy: {:.3f}%'.format
          (epoch + 1, epochs, running_loss / len(training_data_loader),
           (100 * correct_train.double() / len(training_dataset))))
print("DONE TRAINING!")

with torch.no_grad():
    correct = 0
    for images, labels in test_data_loader:
        if CUDA:
            images = images.cuda()
            labels = labels.cuda()
        images = images.view(-1, 28*28)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / len(test_dataset)))