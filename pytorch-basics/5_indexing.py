"""
- includes about normal indexing and fancy indexing on torch tensor
- Indexing is similar to numpy array
- requirements:
    - pip install torch
"""

import torch

# check imported torch version
print(f"torch version: {torch.__version__}")

# setup code to reproduce exact same output
# set the seed
torch.manual_seed(50)

# ------------------------------------------------------------------------#
#                           Pytorch Indexing                              #
# ------------------------------------------------------------------------#
# 
# Indexing is similar to numpy array
sample_tensor = torch.rand((1, 16)).view(4, 4)

# extract third column
third_col = sample_tensor[:, 3]
print(f"\nextracting third column: {third_col}")
# extract extract third row
third_row = sample_tensor[3, :]
print(f"extracting third row: {third_row}")
# extract first three rows of last column
three_rows_last_col = sample_tensor[:3, -1]
print(f'extracting first three rows of last column: {three_rows_last_col}')

# Fancy Indexing
#
# slice rows=[1, 4] and cols = [1, 4]
rows_index = [0, 3]
col_index = [0, 3]
fancy_sliced = sample_tensor[rows_index, col_index]
print(f"\nFancy Indexing: Sliced rows=[1, 4] and cols=[1, 4]\n {fancy_sliced}")
