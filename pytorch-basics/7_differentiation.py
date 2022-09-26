"""
- consider equations: 
    - y = x * w + b, 
    - z = l * y + m, 
    - v = c * z + d
- we want gradients of w, b, l, m, c, and d
- We initialize w, b, l, m, c, and d with some random numbers
- we will use pytorch autograd package to compute automatic differentiation
- torch.autograd, is PyTorch's automatic differentiatio engine
- requirements:
    - pip install torch
- References:
    - https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
"""
from importlib.metadata import requires
import torch


# setup code to reproduce the same exact output
# set the seed
torch.manual_seed(50)

def main():
    """
    Given equations:
        - y = x * w + b, 
        - z = l * y + m, 
        - v = c * z + d
    we compute gradients of w, b, l, m, c, and d using pytorch autgrads package
    """
    #  y = x * w + b, z = l * y + m, v = c * z + d

    # use requires_grad argument to define 
    # if you want to compute gradients,
    # requires_grad=False -- no need to compute derivate
    # requires_grad=False -- need to compute derivative
    x = torch.tensor([1., 2, 3, 4], requires_grad = False) # no derivate to compute
    w = torch.tensor([5., 6, 7, 8], requires_grad = True) # need to compute derivative
    b = torch.tensor([10.], requires_grad = True) # need to compute derivative

    # first equations 
    y = x * w + b

    # initialize second equations input
    l = torch.tensor([10., 11, 12, 13], requires_grad = True) # need to compute derivative
    m = torch.tensor([11.], requires_grad = True) # need to compute derivative

    # second equations
    z = l * y + m 

    # initialize third equations input
    c = torch.tensor([14., 15, 16, 17], requires_grad = True) # need to compute derivative
    d = torch.tensor([13.], requires_grad = True) # need to compute derivative

    # third equations
    v = c * z + d # final equation

    # sum, to get scalar
    v = v.sum() # backpropagate v
    
    # compute derivatives
    v.backward()

    # print(w.grad)
    # Compute derivatives,
    # v w.r.t w, b, l, m, c, d
    # grad are stored on .grad attribute i.e.
    # w.grad, b.grad, l.grad, m.grad, c.grad, d.grad
    print("\n***Gradient Computation using torch autograd***")
    print('dv/dx:', x.grad) # grad=None, since requires_grad = False
    print(f'dv/dw at w={w.detach()}: ', w.grad)
    print(f'dv/dl at l = {l.detach()}: ', l.grad)
    print(f'dv/dm at m = {m.detach()}: ', m.grad)
    print(f'dv/dc at c = {c.detach()}: ', c.grad)
    print(f'dv/dd at d = {d.detach()}: ', d.grad)


if __name__ == "__main__":
    main()