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
        <td>Initializes 3x3 tensor with all 1's</td>
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

<table>
    <caption><b><i>Tensor Operations</i></b></caption>
    <tr>
        <th>methods</th>
        <th>example</th>
        <th>descriptions</th>
    </tr>
    <tr>
        <td>squeeze()</td>
        <td>torch.squeeze(input_tensor)</td>
        <td>remove all dimension=1</td>
    </tr>
    <tr>
        <td>unsqueeze()</td>
        <td>torch.unsqueeze(input_tensor, dim=0)</td>
        <td>add dimension=1 at position 0. e.g. out tensor shape = (1, *)</td>
    </tr>
    <tr>
        <td>add()</td>
        <td>out = torch.add(tensor1, tensor2)</td>
        <td>Adds two tensor and store result in out</td>
    </tr>
    <tr>
        <td>sub()</td>
        <td>torch.sub(tensor1, tensor2, out=subtraction)</td>
        <td>Subtract tensor1 and tensor2, and output result in pre-initialized tensor (i.e. subtraction)</td>
    </tr>
    <tr>
        <td>true_divide()</td>
        <td>torch.true_divide(tensor1, tensor2)</td>
        <td>Divides two tensor passed as an argument i.e. tensor1/tensor2. true_divide() function is an alias for divide(). </td>
    </tr>
    <tr>
        <td>matmul()</td>
        <td>torch.matmul(matrix_1, matrix_2)</td>
        <td>performs the matrix multiplication over two tensor passed as an arguments.</td>
    </tr>
    <tr>
        <td>bmm()</td>
        <td>
            <ul>
                <li>batch_mat_1 shape = (2, 3, 2)</li>
                <li>batch_mat_2 shape = (2, 2, 3)</li>
                <li>out_mat shape = (2, 3, 3)</li>
                <li>torch.bmm(batch_mat_1, batch_mat_2)</li>
            </ul>
        </td>
        <td>performs a batch matrix-matrix product of matrices.</td>
    </tr>
    <tr>
        <td>min()</td>
        <td>tensor.min(axis=1)</td>
        <td>returns minimum value of all elements in the input tensor. If axis is passed (axis=1) then returns minimum value with indices along the passed axis.</td>
    </tr>
    <tr>
        <td>max()</td>
        <td>torch.max(axis=1)</td>
        <td>returns maximum value of all elements in the input tensor. If axis is passed (axis=1) then returns maximum value with indices along the passed axis.</td>
    </tr>
    <tr>
        <td>argmin()</td>
        <td>tensor.argmin(axis=1)</td>
        <td>returns the index of minimum value along the passed axis (axis=1/axis=0).</td>
    </tr>
    <tr>
        <td>argmax()</td>
        <td>torch.argmax(axis=1)</td>
        <td>returns the index of maximum value along the passed axis (axis=1/axis=0).</td>
    </tr>
    <tr>
        <td>torch.clamp()</td>
        <td>torch.clamp(tensor_to_clamp, min=0, max=None)</td>
        <td>clamp operation clamps the input tensor in the specified value range passed in min and max attribute. ReLU is a special case of clamp operations.</td>
    </tr>
    <tr>
        <td>cat()</td>
        <td>
            <ol>
                <li>torch.cat((tensor1, tensor2), axis=0)</li>
                <li>torch.cat((tensor1, tensor2), axis=1)</li>
            </ol>
        </td>
        <td>
            <ol>
                <li>concatenates two tensor along axis = 0</li>
                <li>concatenates two tensor along axis = 1</li>
            </ol>
        </td>
    </tr>
</table>
