import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from utils import gaussian_kernel, filter2d


def main():
    # load the image
    im = imread('paint.jpg').astype('float')
    im = im / 255

    # number of levels for downsampling
    N_levels = 5

    # make a copy of the original image
    im_subsample = im.copy()

    # naive subsampling, visualize the results on the 1st row
    for i in range(N_levels):
        # subsample image
        im_subsample = im_subsample[::2, ::2, :]
        plt.subplot(2, N_levels, i + 1)
        plt.imshow(im_subsample)
        plt.axis('off')

    # subsampling without aliasing, visualize results on 2nd row

    #### YOUR CODE HERE
    im_subsample = im.copy()

    for i in range(N_levels):
        channels = []
        for channel in range(3):
            # Extract image of particular channel and applying gaussian kernel
            channel_data = filter2d(im_subsample[:, :, channel], gaussian_kernel())
            channels.append(channel_data[::2, ::2])  # Downsampling each channel
        im_subsample = np.stack(channels, axis=-1)  # Combining channels

        plt.subplot(2, N_levels, N_levels + i + 1)
        plt.imshow(im_subsample)
        plt.axis('off')
        plt.text(0.5, 1.05, f'Level {i + 1}', horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes)

    plt.show()
    #### END YOUR CODE


if __name__ == "__main__":
    main()
