import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets

input, expectedOutput = sklearn.datasets.make_moons(200, noise = 0.5)

# plot data
plt.scatter(input[:,0],input[:,1], c=expectedOutput)


input_neurons_length = 2
output_neurons_length = 2 #cross entrype 2 otherwise1
number_of_sample = input.shape[0]
learning_rate = 0.001;
regular_expression_constant = 0.01