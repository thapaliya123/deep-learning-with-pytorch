"""
- includes about squeeze() and unsqueeze() pytorch tensor method
- handy methods when dealing with multi-dimensional tensor.
    - Task such as: Image Captioning
- requirements:
    - pip install torch
- References:
    - https://pytorch.org/docs/stable/torch.html
    - https://www.geeksforgeeks.org/how-to-squeeze-and-unsqueeze-a-tensor-in-pytorch/
"""
import torch

# check imported torch version
print(f"torch version: {torch.__version__}")

# setup code to reproduce exact same output
# set the seed
torch.manual_seed(50)

# ------------------------------------------------------------------------
#                      squeeze and unsqueeze operations
# ------------------------------------------------------------------------
# squeeze(): remove dimension 1 from the specified position of the input tensor
# unsqueeze(): add dimension 1 in the specified position of the input tensor
# Idea: adding or removing dimension 1 doesn't change tensor size
# Handy function when dealing with multi-dimensional tensor
#     e.g. NLP related task: Image captioning

input_tensor = torch.randn(3, 1, 2, 1, 4) # tensor shape: (3, 1, 2, 1, 4), for squeeze()
print("\n***squeeze()***")
print(f"Input tensor size (defined for squeeze() operation):\n {input_tensor.size()}")

# squeeze the tensor
squeezed_tensor = torch.squeeze(input_tensor) # remove all dimension=1, default, out shape: (3, 2, 4)
print(f"size after squeeze (remove all dimension=1):\n {squeezed_tensor.size()}")
# squeeze the tensor in  the dimension 0
squeezed_tensor = torch.squeeze(input_tensor, dim=0) # out shape: (3, 1, 2, 1, 4)
print("Size after squeeze with dim=0: ", squeezed_tensor.size())
# squeeze the tensor in the dimension 1
squeezed_tensor = torch.squeeze(input_tensor, dim=1) # out shape: (3, 2, 1, 4)
print("Size after squeeze with dim=1: ", squeezed_tensor.size())
# squeeze the tensor in the dimension 2
squeezed_tensor = torch.squeeze(input_tensor, dim=2) # out shape: (3, 1, 2, 1, 4)
print("Size after squeeze with dim=2: ", squeezed_tensor.size())
# squeeze the tensor in the dimension 3 
squeezed_tensor = torch.squeeze(input_tensor, dim=3) # out shape: (3, 1, 2, 4)
print("Size after squeeze with dim=3: ", squeezed_tensor.size())
# squeeze the tensor in the dimension 4
squeezed_tensor = torch.squeeze(input_tensor, dim=4) # out shape: (3, 1, 2, 1, 4)
print("Size after squeeze with dim=4:\n", squeezed_tensor.size())

# unsqueeze() tensor
# dim should be in range of [-input.dim()-1, input.dim()+1]
# negative dim will be changed to dim = dim + input.dim() + 1, i.e 0
input_tensor = torch.arange(10) # tensor shape: (10,)
print("\n***unsqueeze()****")
print("Input tensor size (defined for unsqueeze() operations): {}".format(input_tensor.size()))
print("Input tensor before unsqueeze():\n {}".format(input_tensor))

# unsqueeze(dim=0), add dimension=1 at position 0
unsqueezed_tensor = torch.unsqueeze(input_tensor, dim=0) # out tensor shape: (1, 10)
print("Size after unsqueeze with dim=0: {}".format(unsqueezed_tensor.size()))
print("Tensor after unsqueeze(dim=0):\n {}".format(unsqueezed_tensor))
# unsqueeze(dim=1), add dimension=1 at position 1
unsqueezed_tensor = torch.unsqueeze(input_tensor, dim=1) # out tensor shape: (10, 1)
print("Size after unsqueeze with dim=1: {}".format(unsqueezed_tensor.size()))
print("Tensor after unsqueeze(dim=1):\n {}".format(unsqueezed_tensor))
# unsqueeze(dim=0) followed by unsqueeze(dim=1)
unsqueezed_tensor = input_tensor.unsqueeze(dim=0).unsqueeze(dim=1) # (1, 1, 10)
print("Size after unsqueeeze(dim=0) followed by unsqueeze(dim=1): {}".format(unsqueezed_tensor.size()))
print("Tensor:\n {}".format(unsqueezed_tensor))