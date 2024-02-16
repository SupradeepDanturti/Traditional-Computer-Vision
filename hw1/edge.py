import numpy as np
import matplotlib.pylab as plt
from skimage import io
from utils import gaussian_kernel, filter2d, partial_x, partial_y

def main():
    # Load image
    img = io.imread('iguana.png', as_gray=True)

    ### YOUR CODE HERE

    # Smooth image with Gaussian kernel
    kernel = gaussian_kernel()
    smooth_img = filter2d(img, kernel)

    # Compute x and y derivate on smoothed image
    dx = partial_x(smooth_img)
    dy = partial_y(smooth_img)

    # Compute gradient magnitude
    magnitude = (dx**2 + dy**2)**0.5

    # Visualize results
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original Image")

    plt.subplot(2, 2, 2)
    plt.imshow(dx, cmap="gray")
    plt.title("X Gradient")

    plt.subplot(2, 2, 3)
    plt.imshow(dy, cmap="gray")
    plt.title("Y Gradient")

    plt.subplot(2, 2, 4)
    plt.imshow(magnitude, cmap="gray")
    plt.title("Gradient Magnitude")

    plt.tight_layout()
    plt.show()
    
    ### END YOUR CODE
    
if __name__ == "__main__":
    main()

