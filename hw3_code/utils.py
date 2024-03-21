from skimage import io
from skimage.util import img_as_float
from skimage.color import rgb2gray
import numpy as np
import cv2
from scipy.ndimage import correlate
import sklearn.cluster
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


def applyFilterBank(img, F):
    """
    Apply a filter bank to an image and return the filter responses.

    Parameters:
    - img: The input image.
    - F: The filter bank, a list of filters.

    Returns:
    - A list of filter responses.
    """
    responses = []
    for filter in F:
        filtered_img = cv2.filter2D(img, -1, filter)
        responses.append(filtered_img.flatten())  # Flatten and store each response
    return np.array(responses).T  # Transpose to make rows correspond to pixels


def computeHistogram(img_file, F, textons):
    
    ### YOUR CODE HERE
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    responses = applyFilterBank(img, F)

    distances = cdist(responses, textons)

    closest_textons = np.argmin(distances, axis=1)

    histogram, _ = np.histogram(closest_textons, bins=np.arange(len(textons) + 1), density=True)
    return histogram
    ### END YOUR CODE
    
def createTextons(F, file_list, K):

    ### YOUR CODE HERE
    all_responses = []
    for filename in file_list:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        responses = applyFilterBank(img, F)
        all_responses.append(responses)
    all_responses = np.vstack(all_responses)

    kmeans = KMeans(n_clusters=K, random_state=0).fit(all_responses)
    return kmeans.cluster_centers_

    ### END YOUR CODE
