"""
- includes commonly used operations on tensor i.e.
    - Addition, Subtraction, Division, Matrix Multiplication, Elementwise multiplication, Batch matrix multiplication
    - max(), min(), argmax(), argmin(), 
    - clamp()
    - cat()
- suitable for beginners willing to learn deep learning using pytorch
- requirements:
    - pip install torch
- References:
    - https://pytorch.org/docs/stable/torch.html
"""
import torch

# check imported torch version
print(f"torch version: {torch.__version__}")

# setup code to reproduce exact same output
# set the seed
torch.manual_seed(50)

# ------------------------------------------------------------------------
#                       Pytorch Tensor Operations
# ------------------------------------------------------------------------
#
# 1.Addition
tensor1 = torch.randn(3, 3) # 3x3 shape
tensor2 = torch.randn(3, 3) # 3x3 shape
addition = tensor1 + tensor2 # or addition = torch.add(tensor1 + tensor2)
print("\nAddition of two tensor")
print(f"tensor 1: \n{tensor1}")
print(f"tensor2: \n{tensor2}")
print(f"1.tensor1+tensor2: \n{addition}")
#
# 2.Subtraction
subtraction = torch.empty(3, 3)
torch.sub(tensor1, tensor2, out=subtraction) # output result in pre-initialized tensor (i.e. subtraction)
print(f"2.tensor1-tensor2:\n{subtraction}")
# 3.Division
division = torch.true_divide(tensor1, tensor2) # true_divide() function is an alias for divide()
print(f"3.tensor1/tensor2:\n{division}")
#
# 4.Matrix multiplication
# torch.matmul: performs the matrix product over two tensors
# torch.mm: perforsm the matrix product, doesn't support broadcasting
x = torch.rand(2, 3) #input vector, shape: (2, 3)
W = torch.rand(3, 3) # random weight, shape: (3, 3)
h = torch.matmul(x, W) # activation output, shape: (2, 3)
print(f"3.Matrix multiplication i.e. matmul(x, W):\n{h}")
#
# 5. Elementwise multplication
# shape of two tensors must be same
elementwise_product = tensor1*tensor2
print(f"4.Elementwise multiplication i.e. (tensor1*tensor2):\n{elementwise_product}")
#
# 6. Batch matrix multiplication
# performs a batch matrix-matrix product of matrices 
# batch matrix 1 shape: (bxnxm)
# batch matrix 2 shape: (bxmxp)
# out matrix shape: (bxnxp)
b = 2
n = 3
m = 2
p = 3
batch_mat_1 = torch.randn((b, n, m))
batch_mat_2 = torch.randn((b, m, p))
out_matrix = torch.bmm(batch_mat_1, batch_mat_2) 
print(f'\nbatch matrix 1 shape: {batch_mat_1.shape}')
print(f'batch matrix 2 shape: {batch_mat_2.shape}')
print(f"output matrix shape: {out_matrix.shape}")
print(f"5.Output of Batch matrix multiplication:\n {out_matrix}")
#
# 7. max, min, argmax, argmin
# max: returns the maximum value of all elements in the input tensor
# min: returns the minimum value of all elements in the input tensor
# argmax: returns index of maximum value
# argmin: returns index of minimum value
print("7.max/min/argmax operations")
print(f"Tensor1: {tensor1}")
tensor_max = tensor1.max(axis=1)
tensor_min = tensor1.min(axis=1)
tensor_argmax = tensor1.argmax(axis=1)
tensor_argmin = tensor1.argmin(axis=1)
print(f"tensor1.max(axis=1): {tensor_max}")
print(f"\ntensor1.min(axis=1): {tensor_min}")
print(f"\ntensor1.argmax(axis=1): {tensor_argmax}")
print(f"\ntensor1.argmin(axis=1): {tensor_argmin}")
#
# 8. torch.clamp()
# ReLU is special case of Clamp() with min=0 and max=None i.e. 
# clamp(tensor, min=0, max=None): Clamps all elements in input into the range [0, x].
print("8.Clamp operations")
min_clamp = 0
max_clamp = None
tensor_to_clamp = torch.tensor([[-5, 0, 1],
                                [2, 3, 5],
                                [10, -2, 3]])
print(f"Tensor to be clammed:\n {tensor_to_clamp}")

tensor_to_clamp = torch.clamp(tensor_to_clamp, min=min_clamp, max=max_clamp) # its similar to ReLU functo

print(f"Tensor after clammed operations, clamp(0):\n {tensor_to_clamp}")
#
# 9. torch.cat()
# useful in concatenating two or more tensors in a specified dimensions
print("9.Concatenation operations")
print(f"Tensor1 shape: {tensor1.shape}") # tensor1 shape: (3, 3)
print(f"Tensor2 shape: {tensor2.shape}") # tensor2 shape: (3, 3)
concat_axis_0 = torch.cat((tensor1, tensor2), axis=0) # concat axis=0
concat_axis_1 = torch.cat((tensor1, tensor2), axis=1) # concat axis=1
print(f"Concatenation axis=0 shape: {concat_axis_0.shape}") # expected shape: (6, 3)
print(f"Concatenation axis=1 shape: {concat_axis_1.shape}") # expected shape: (3, 6)
