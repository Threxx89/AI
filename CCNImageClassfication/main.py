import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

mean_gray = 0.1307
stddev_gray = 0.3081
# transform image to tensor and then normalized the image
#formula that gets calculated
# input[channel] = (input[channel] - mean[channel]) /std[channel]
transforms = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((mean_gray),(stddev_gray))])

train_dataset = datasets.MNIST(root='./data',
                               train= True,
                               transform=transforms,
                               download=True)

test_dataset = datasets.MNIST(root='./data',
                               train= False,
                               transform=transforms)

