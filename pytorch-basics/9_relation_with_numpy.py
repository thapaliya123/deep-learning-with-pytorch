"""
- NumPy is useful pacakge on dealing with matrix operations, fourier transform.
- When performing deep learning we will process dataset using numpy,
- Do visualization using matplotlib, use pandas for analysis
- Pytorch interoperates really with NumPY.
    - torch.from_numpy(numpy_arr)
    - torch_tensor.numpy()
- References:
    - https://github.com/yunjey/pytorch-tutorial
"""
import numpy as np
import torch

# setup code to reproduce exact same output
# set the seed
torch.manual_seed(50)

def define_device():
    """
    defines device object that points to GPU if you have one 
    otherwise points to the CPU.
    """
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return dev

def main():
    # initialize numpy array
    numpy_arr = np.array([[1, 2], [3, 4]])

    # convert numpy array to pytorch tensor,
    numpy_to_tensor_cpu = torch.from_numpy(numpy_arr)
    
    # push tensor to GPU
    dev = define_device()
    print("Putting tensor to GPU")
    numpy_to_tensor_gpu = numpy_to_tensor_cpu.to(dev)
    
    # convert pytorch tensor back to numpy array
    # 1. when device type = cpu
    cpu_tensor_to_numpy = numpy_to_tensor_cpu.numpy()

    # 2. when device type = cuda
    # first bring tensor to cpu and convert to numpy
    gpu_tensor_to_numpy = numpy_to_tensor_gpu.cpu().numpy()

    print(f"\nOriginal Numpy array data type: {numpy_arr.dtype}")
    print(f"Original Numpy array:\n {numpy_arr}")
    print(f"\nNumpy to pytorch tensor data type: {numpy_to_tensor_cpu.dtype}")
    print(f"Numpy 2 pytorch tensor:\n {numpy_to_tensor_cpu}")
    print(f"\nPytorch tensor(device=cpu) to Numpy array data type: {cpu_tensor_to_numpy.dtype}")
    print(f"Pytorch tensor(device=cpu) 2 numpy:\n {cpu_tensor_to_numpy}")
    print(f"\nPytorch tensor(device=gpu) to Numpy array data type: {gpu_tensor_to_numpy.dtype}")
    print(f"Pytorch tensor(device=gpu) 2 numpy:\n {gpu_tensor_to_numpy}")

if __name__ == "__main__":
    main()