"""
- includes commonly used tensor initialization technique in pytorch
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
#                             Tensor Initialzation
#-------------------------------------------------------------------------
#
# Tensor: equivalent to numpy array with additional GPU support
#
# create tensor: most common --> torch.Tensor() or torch.tensor()
# torch.Tensor(): pass tuple i.e. desired tensor shape. e.g (3, 3)
# torch.tensor(): pass list i.e. converted to tensor
sample_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print("\n3x3 tensor initialization using torch.Tensor()")
print(torch.Tensor(3, 3))
print("\n3x3 tensor initialization using torch.tensor() from python list")
print(torch.tensor(sample_list))

# Others initialization 
# 1. torch.eye: creates a tensor equivalent to identity matrix
# 2. torch.ones: creates a tensor filled with ones.
# 3. torch.rand: creates a tensor random values uniformly sampled between 0 and 1
# 4. torch.zeros: creates a tensor with all zeros.
# 5. torch.randn: creates a tensor with random values sampled from standard normal dist
# 6. torch.arange: creates a tensor with values in specified range
print("\n***torch.eye, shape:(3, 3)***")
print(torch.eye(3, 3))
print("\n***torch.ones, shape:(3, 3)***")
print(torch.ones((3, 3)))
print("\n***torch.rand, shape:(3, 3)***")
print(torch.rand((3, 3)))
print("\n***torch.zeros, shape:(3, 3)***")
print(torch.zeros((3, 3)))
print("\n***torch.randn, shape:(3, 3)***")
print(torch.randn((3, 3)))
print("\n***torch with values range in [0, 9]***")
print(torch.arange(start=0, end=10, step=1))