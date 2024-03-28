import numpy as np
import cv2
from scipy.spatial.distance import cdist


def computeHistogram(img_file, F, textons):
    ### YOUR CODE HERE
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    responses = np.array([cv2.filter2D(img, -1, filter).flatten() for filter in F]).T
    distances = cdist(responses, textons)

    close_val = np.argmin(distances, axis=1)

    histogram, _ = np.histogram(close_val, bins=np.arange(len(textons) + 1), density=True)
    return histogram
    ### END YOUR CODE


def createTextons(F, file_list, K):
    ### YOUR CODE HERE
    data = []
    for filename in file_list:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        responses = np.array([cv2.filter2D(img, -1, filter).flatten() for filter in F]).T
        data.append(responses)
    data = np.vstack(data)

    """Clustering"""
    centers = data[np.random.choice(data.shape[0], K, replace=False), :]
    dists = np.linalg.norm(data[:, None] - centers[None, :], axis=2)
    closest = np.argmin(dists, axis=1)
    centers = np.array([data[closest == k].mean(axis=0) for k in range(K)])

    return centers

    ### END YOUR CODE
