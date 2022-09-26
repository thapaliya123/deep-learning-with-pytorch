"""
- includes commonly used operations such as 
    - shape, reshape, view, and permute
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
#                  Tensor Shape, Reshape, view, and premute
# ------------------------------------------------------------------------
#
# use either tensor.shape or tensor.size
# shape is an alias of size()
# view(): useful in reshaping, acts on contiguous memory block
# reshape(): useful in reshaping, acts on both contiguous or non-contiguous block
# permute(): returns a view of the original tensor input with its dimension permuted
# -----------useful in tensor manulpulation when dealing with image dataset
sample_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
sample_tensor = torch.tensor(sample_list)
tensor_shape = sample_tensor.shape # out: (3, 3)
tensor_size = sample_tensor.size() # out: (3, 3)
print(f"\ntensor.shape: {tensor_shape}")
print(f"tensor.size: {tensor_size}")

print(f"\nReshape tensor to shape of 1x9:")
view_tensor = sample_tensor.view(1, 9) # out: (1, 9), contiguous memory block
reshape_tensor = sample_tensor.reshape(1, 9) # out: (1, 9), both contiguous or non-contiguous block
print(f"Shape of Tensor after view(): {view_tensor.shape}")
print(f"Shape of Tensor after reshape(): {reshape_tensor.shape}")
#
# permute tensor of shape (1, 3, 3) to (3, 3, 1)
before_permute_tensor = sample_tensor.reshape(1, 3, 3) # original shape: (1, 3, 3)
after_permute_tensor = torch.permute(before_permute_tensor, (2, 1, 0)) # permuted shape: (3, 3, 1)
print(f'\nShape before permute: {before_permute_tensor.shape}')
print(f"Shape after permute: {after_permute_tensor.shape}")
