"""
- Computing using GPU is essentially for achieving parallelism
- Pytorch supports GPU for processing the tensors.
- Here, we will see some handy methods when processing tensors using GPU.
- References:
    - https://pytorch.org/docs/stable/index.html#pytorch-documentation
    - https://pytorch.org/docs/stable/cuda.html
"""

import torch

# setup code to reproduce exact same output
# set the seed
torch.manual_seed(50)

def is_gpu_available():
    """
    checks whether you have GPU available.
    """
    gpu_avai = torch.cuda.is_available()
    return gpu_avai

def count_list_available_gpus():
    """
    Counts total number of GPUS available in your system.
    list name of all available gpus in the system.
    -------------
    Example:
    -------------
    >>>> Number of available gpus are: 1
    >>>> Your cuda devices are: ['cuda:0']
    """
    # count available gpus
    num_gpus = torch.cuda.device_count()
    # list available gpu names
    cuda_devices = [f"cuda:{i}" for i in range(num_gpus)] # ['cuda:0', 'cuda:1']
    print(f"Number of available gpus are: {num_gpus}")
    print(f"Number of: {cuda_devices}")

    return num_gpus

def define_device():
    """
    defines device object that points to GPU if you have one 
    otherwise points to the CPU.
    """
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return dev

def main():
    # Test GPU availability
    gpu_avai = is_gpu_available() # returns True if GPU
    print(f"Is GPU available? {gpu_avai}")

    # Count available gpus
    # list all the available gpus
    num_gpus = count_list_available_gpus()

    # Select GPU or CPU based on availability
    dev = define_device()
    print(f"Available device: {dev}")

    # All the created tensors are stored,
    # on cpu by default, but we need to push
    # them to GPU for computing using GPU
    # handy functions: .to(...) or .cuda()
    #
    # Now lets create a tensor and push it to device
    identity_mat = torch.eye(3, 3) # identity matrix
    print(f"Pushing tensor to device: {dev}")
    identity_mat = identity_mat.to(dev) # push to device
    print(f"Tensor after pushing to: {dev}\n {identity_mat}") # see attribute, device="..."


if __name__ == "__main__":
    main()

