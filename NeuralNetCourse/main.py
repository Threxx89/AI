# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import numpy as np

# This is a 1-D Tensor
OneD = torch.tensor([2, 2, 1]);
print(OneD);

# This is a 2-D Tensor
TwoD = torch.tensor([[2, 1, 4], [3, 5, 4], [1, 2, 0], [4, 3, 2]])
print(TwoD)

# The size of the tensors
print(OneD.shape)
print(TwoD.shape)
print(OneD.size())
print(TwoD.size())

# How to get the height/number of rows
print(TwoD.shape[0])

floatTensor = torch.FloatTensor([[2, 1, 4], [3, 5, 4], [1, 2, 0], [4, 3, 2]])

doubleTensor = torch.DoubleTensor([[2, 1, 4], [3, 5, 4], [1, 2, 0], [4, 3, 2]])

print(floatTensor)
print(floatTensor.dtype)

print(doubleTensor)
print(doubleTensor.dtype)

# Calculate mean and standard deviation

print(floatTensor.mean())
print(doubleTensor.mean())
print(floatTensor.std())
print(doubleTensor.std())

# Reshape a Tensor
# Note: If one of the dimensions is -1 the size can be inferred
print(TwoD.view(-1,1))
print(TwoD.view(12))
print(TwoD.view(-1,4))
print(TwoD.view(3,4))
# Assign TwoD a new Shape
TwoD = TwoD.view(-1,1)

print(TwoD)
print(TwoD.shape)

# Create a 3-D Tensor with 2 channels. 3 rows and 4 columns
ThreeD = torch.randn(2,3,4)
print('\n')
print(ThreeD)
print(ThreeD.view(2,12))
print(ThreeD.view(2,-1))

# Create a matrix with random numbers between 0 and 1
matrix = torch.rand(4,4)
print(matrix)

# create a matrix with random number taken from a normal distribute with mean 0 and veriation 1

matrixTwo = torch.rand(4,4)
print(matrix)
print(matrix.dtype)

# create a matrix with random int between 6 and 9
intMatrix = torch.randint(6,10,(5,))
print(intMatrix)
print(intMatrix.dtype)

# create a 3*3 matrix with random int between 6 and 9
intThreeMatrix = torch.randint(6,10,(3,3))
print(intThreeMatrix)
print(intThreeMatrix.dtype)

# get number of elements in matrix
print(torch.numel(intMatrix))

print(torch.numel(intThreeMatrix))

# construct 3*3 matrix of type long

longMatrix = torch.zeros(3,3, dtype=torch.long)
print(longMatrix)

oneMatrix = torch.ones(3,3)
print(oneMatrix)
print(oneMatrix.dtype)

# reference size of tensor from another tensor

refsize = torch.randn_like(matrixTwo, dtype=torch.double)
print(refsize)

# add two tensor together

add_result = torch.add(matrix,matrixTwo)
print(matrixTwo)

# matrix slicing
print(matrixTwo[:,1])
print(matrixTwo[:,:2])
print(matrixTwo[:3,:])
num_ten = matrixTwo[2,3]
print(num_ten)
print(num_ten.item())
print(matrix[2,:])

#Converting a Torch Tensor to a NumPy Array
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
#See how the numpy array changed their value.
a.add_(1)
print(a)
print(b)

#Converting NumPy Array to Torch Tensor
#See how changing the np array changed the Torch Tensor automatically
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

#Move the tensor to the GPU
#r2 = matrixTwo.cuda()
#print(r2)

#Provide Easy switching between CPU and GPU
CUDA = torch.cuda.is_available()
print(CUDA)
if CUDA:
    add_result = add_result.cuda()
    print(add_result)

#You can also convert a list to a tensor
a = [2,3,4,1]
print(a)
to_list = torch.tensor(a)
print(to_list, to_list.dtype)

data =  [[1., 2.], [3., 4.],
         [5., 6.], [7., 8.]]
T = torch.tensor(data)
print(T, T.dtype)

#Tensor Concatenation
first_1 = torch.randn(2, 5)
print(first_1)
second_1 = torch.randn(3, 5)
print(second_1)
#Concatenate along the 0 dimension (concatenate rows)
con_1 = torch.cat([first_1, second_1])
print('\n')
print(con_1)
print('\n')
first_2 = torch.randn(2, 3)
print(first_2)
second_2 = torch.randn(2, 5)
print(second_2)
# Concatenate along the 1 dimension (concatenate columns)
con_2 = torch.cat([first_2, second_2], 1)
print('\n')
print(con_2)
print('\n')

#Adds a dimension of 1 along a specified index
tensor_1 = torch.tensor([1, 2, 3, 4])
tensor_a = torch.unsqueeze(tensor_1, 0)
print(tensor_a)
print(tensor_a.shape)
tensor_b = torch.unsqueeze(tensor_1,1)
print(tensor_b)
print(tensor_b.shape)
print('\n')
tensor_2 = torch.rand(2,3,4)
print(tensor_2)
print('\n')
tensor_c = tensor_2[:,:,2]
print(tensor_c)
print(tensor_c.shape)
print('\n')
tensor_d = torch.unsqueeze(tensor_c,2)
print(tensor_d)
print(tensor_d.shape)

#Remember, If requires_grad=True, the Tensor object keeps track of how it was created.
x = torch.tensor([1., 2., 3], requires_grad=True)
y = torch.tensor([4., 5., 6], requires_grad=True)
#Notice that both x and y have their required_grad set to true, therefore we an compute gradients with respect to them
z = x + y
print(z)
# z knows that is was created as a result of addition of x and y. It knows that it wasn't read in from a file
print(z.grad_fn)
#And if we go further on this
s = z.sum()
print(s)
print(s.grad_fn)

#Now if we backpropagate on s, we can find the gradients of s with respect to x
s.backward()
print(x.grad)

# By default, Tensors have `requires_grad=False`
x = torch.randn(2, 2)
y = torch.randn(2, 2)
print(x.requires_grad, y.requires_grad)
z = x + y
# So you can't backprop through z
print(z.grad_fn)
#Another way to set the requires_grad = True is
x.requires_grad_()
y.requires_grad_()
# z contains enough information to compute gradients, as we saw above
z = x + y
print(z.grad_fn)
# If any input to an operation has ``requires_grad=True``, so will the output
print(z.requires_grad)
# Now z has the computation history that relates itself to x and y

new_z = z.detach()
print(new_z.grad_fn)
# z.detach() returns a tensor that shares the same storage as ``z``, but with the computation history forgotten.
#It doesn't know anything about how it was computed.In other words, we have broken the Tensor away from its past history

#You can also stop autograd from tracking history on Tensors. This concept is useful when applying Transfer Learning
print(x.requires_grad)
print((x+10).requires_grad)

with torch.no_grad():
    print((x+10).requires_grad)

#Let's walk in through one last example
x = torch.ones(2, 2, requires_grad=True)
print(x)
y = x + 2
print(y)
print(y.grad_fn)
z = y * y * 3
out = z.mean()
print(z, out)
out.backward()
print(x.grad)

m1 = torch.ones(5,5)
m2 = torch.zeros(5,5)
#Perform element-wise multiplaction
mul = torch.mul(m1,m2)
#Another way to perform element-wise multiplaction
mul_another = m1*m2
print(mul)
print(mul_another)

