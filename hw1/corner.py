import numpy as np
from utils import filter2d, partial_x, partial_y
from skimage.feature import peak_local_max
from skimage.io import imread
import matplotlib.pyplot as plt

def harris_corners(img, window_size=3, k=0.04):
    """
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).
        
    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    """

    response = None
    
    ### YOUR CODE HERE

    ### END YOUR CODE

    return response

def main():
    img = imread('building.jpg', as_gray=True)

    ### YOUR CODE HERE
    
    # Compute Harris corner response

    # Threshold on response

    # Perform non-max suppression by finding peak local maximum

    # Visualize results
    
    ### END YOUR CODE
    
if __name__ == "__main__":
    main()
