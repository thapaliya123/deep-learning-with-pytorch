"""
- core idea about neural network is forward and backward propagation.
- here we will implement forward propagation through computation graph.
- computation graph can be found here: images/forward_prop.jpeg
- Steps:
    - initialize four inputs i.e.
        - x1=2, x2=3, x3=4, x4=5
    - initialize four weights i.e.
        - w1=0.1, w2=0.2, w3=0.3, w4=0.5
    - compute z1, z2, z3, z4 i.e.
        - z1 = x1*w1
        - z2 = x2*w2
        - z3 = x3*w3
        - z4 = x4*w4
    - compute output i.e. 
        - a = z1 + z2 + z3 + z4
- Note:
    - Initialization values are choosen randomly
- requirements:
    - pip install torch
    - pip install opencv-python
"""

import torch
# import cv2

# setup code to reproduce exact same output
# set the seed
torch.manual_seed(50)

def display_img(path_, title_):
    """
    Display image reading from the path passed as an argument using opencv
    """
    import cv2 
    
    # read image
    img = cv2.imread(path_)

    # output image with windows name 
    # as title
    cv2.imshow('title_', img)

    # maintain output window untill 
    # user presses a key
    cv2.waitkey(0)

    # destroy present windows on screen
    cv2.destroyAllWindows()

def main():
    """
    Here we will implement forward propagation through computation graph.
    Computation graph can be found in this path: images/forward_prop.jpeg
    """
    # display computation graph using opencv
    computation_graph = "./images/forward_prop.jpg"
    image_title = "FORWARD PROPAGATION IN COMPUTATION GRAPH"
    try:
        display_img(computation_graph, image_title)
    except:
        print("Unable to display computation graph. Manually visualize in path: '../images/forward_prop.jpg'")
    # initialize input tensor
    x1 = torch.Tensor([2]) # x1 = 2
    x2 = torch.Tensor([3]) # x2 = 3
    x3 = torch.Tensor([4]) # x3 = 4
    x4 = torch.Tensor([5]) # x4 = 5
    print(f"\nx1={x1}, x2={x2}, x3={x3}, x4={x4}")

    # initialize weight tensor
    w1 = torch.Tensor([0.1]) # w1 = 0.1
    w2 = torch.Tensor([0.2]) # w2 = 0.2
    w3 = torch.Tensor([0.3]) # w3 = 0.3
    w4 = torch.Tensor([0.4]) # w4 = 0.4
    print(f"w1={w1}, w2={w2}, w3={w3}, w4={w4}")

    # compute z1, z2, z3, z4
    z1 = x1 * w1 # z1 = 0.2
    z2 = x2 * w2 # z2 = 0.60
    z3 = x3 * w3 # z3 = 1.2
    z4 = x4 * w4 # z4 = 2.0

    # compute output, a=z1+z2+z3+z4
    # a = torch.add(z1, z2, z3, z4) # a = 4.0
    a = z1 + z2 + z3 + z4 # a = 4.0
    print(f"a = x1*w1+x2*w2+x3*w3+x4*w4 = {a}")

if __name__ == "__main__":
    main()