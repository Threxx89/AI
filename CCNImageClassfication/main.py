import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import torch.utils.data as data

mean_gray = 0.1307
stddev_gray = 0.3081
# transform image to tensor and then normalized the image
# formula that gets calculated
# input[channel] = (input[channel] - mean[channel]) /std[channel]
transforms = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((mean_gray), (stddev_gray))])

train_dataset = datasets.MNIST(root='./data',
                               train=True,
                               transform=transforms,
                               download=True)

test_dataset = datasets.MNIST(root='./data',
                              train=False,
                              transform=transforms)

# Visualize data
random_img = train_dataset[20][0].numpy() * stddev_gray + mean_gray
plt.imshow(random_img.reshape(28, 28), cmap='gray')
plt.show()

print(train_dataset[20][1])

batch_size = 100
train_loader = data.DataLoader(dataset=train_dataset,
                               batch_size=batch_size,
                               shuffle=True)

test_loader = data.DataLoader(dataset=test_dataset,
                              batch_size=batch_size,
                              shuffle=False)

print(len(train_loader))


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Layer 1
        # Same padding input size will = outputsize
        # Same Padding  = filter size  -1 /2
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        # Formula for output size for each feature map
        # input size - filter size + 2(padding)/ stride +1 =  (28 - 3 + 2(1)) /1 +1 = 28
        self.batchnorm1 = nn.BatchNorm2d(num_features=8)
        self.relu = nn.ReLU()
        # Max Pooling
        # how to determine max pool kern size check
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        # output size is output size /2  28 /2 = 14

        # Layer 2
        # Same Padding
        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=5, stride=1, padding=2)
        # Formula for output size for 32 feature map [(14 - 5 + 2(2)/1 + 1)] = 14
        self.batchnorm2 = nn.BatchNorm2d(num_features=32)
        # Layer 3 fully connected layer 1
        # Flatten the 32 feature maps 7*7*32=1568
        self.fc1 = nn.Linear(1568, 600)
        # Add drop out
        self.dropout = nn.Dropout(p=0.5)
        # Layer 4 fully connected layer 2 output or to label or prediction answers
        self.fc2 = nn.Linear(600, 10)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.cnn2(out)
        out = self.batchnorm2(out)
        out = self.relu(out)
        out = self.maxpool(out)

        # Flatten output -1 figure out what value will be but -1 is usually batch size which is 100
        out = out.view(-1, 1568)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


model = CNN()

CUDA = torch.cuda.is_available()
if CUDA:
    model = model.cuda()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
