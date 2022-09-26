# Deep Learning with PyTorch
- According to Wikipedia, PyTorch is an open source machine learning framework based on the Torch library, used for applications such as computer vision and natural language processing, originally developed by Meta AI and now part of the Linux Foundation Umbrella. 
- It is one of the preferred platforms for deep learning research. 
- It is very famous because:
    - It provdies flexibility to a programmer about how to create, combine, and process tensors as they flow through a network (called computation graph).
    - It provides easy access to a computational graph.

# [Pytorch Basics](https://github.com/thapaliya123/deep-learning-with-pytorch/tree/master/pytorch-basics)
**For python examples:** [click here](https://github.com/thapaliya123/deep-learning-with-pytorch/tree/master/pytorch-basics)
<table>
    <caption><b><i>Tensor Initailization</i></b></caption>
    <tr>
        <th>methods</th>
        <th>descriptions</th>
    </tr>
    <tr>
        <td>torch.tensor([1, 2, 3])</td>
        <td>Initializes torch tensor tensor from the python list</td>
    </tr>
    <tr>
        <td>torch.eye(3, 3)</td>
        <td>Initializes 3x3 indentity matrix tensor</td>
    </tr>
    <tr>
        <td>torch.ones((3, 3))</td>
        <td>Initializes 3x3 tensor will all 1's</td>
    </tr>
    <tr>
        <td>torch.rand((3, 3))</td>
        <td>Initializes 3x3 tensor with random values uniformly sampled between 0 and 1</td>
    </tr>
    <tr>
        <td>torch.zeros((3, 3))</td>
        <td>Initializes 3x3 tensor with all zeros</td>
    </tr>
    <tr>
        <td>torch.randn((3, 3))</td>
        <td>Initializes 3x3 tensor with random values sampled from standard normal distribution</td>
    </tr>
    <tr>
        <td>torch.arange(start=0, end=10, step=1)</td>
        <td>Initializes 1 D tensor with start value=0 and end value=9 in a unit interval i.e. step=1</td>
    </tr>
</table>
