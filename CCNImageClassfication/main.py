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
                                           shuffle= True)

test_loader = data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle= False)


print(len(train_loader))