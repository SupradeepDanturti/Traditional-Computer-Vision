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
    kernel = np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]])
    Ix = partial_x(img)
    Iy = partial_y(img)
    lam_x = Ix * Ix
    lam_y = Iy * Iy
    lam_xy = Ix * Iy
    lam_x = filter2d(lam_x, kernel)
    lam_y = filter2d(lam_y, kernel)
    lam_xy = filter2d(lam_xy, kernel)

    # Compute corner response
    detM = (lam_x * lam_y) - lam_xy ** 2
    traceM = lam_x + lam_y
    response = detM - k * traceM ** 2

    ### END YOUR CODE

    return response

def main():
    img = imread('building.jpg', as_gray=True)

    ### YOUR CODE HERE

    # Compute Harris corner response
    response = harris_corners(img)

    # Threshold on response
    threshold = 0.5 * response.max()
    corner_threshold = response > threshold
    # Perform non-max suppression by finding peak local maximum
    coordinates = peak_local_max(response, min_distance=12)

    # Visualize results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    ax1.imshow(response, cmap='gray')
    ax1.set_title('Response Map')

    ax2.imshow(corner_threshold, cmap='gray')
    ax2.set_title('Threshold Response')

    ax3.imshow(img, cmap='gray')
    ax3.autoscale(False)
    ax3.plot(coordinates[:, 1], coordinates[:, 0], 'rx')
    ax3.set_title('Detected Corners')

    plt.show()
    ### END YOUR CODE
    
if __name__ == "__main__":
    main()
