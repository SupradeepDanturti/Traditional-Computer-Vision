import cv2
import numpy as np

def matchPics(I1, I2):
    """https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html"""
    # Given two images I1 and I2, perform SIFT matching to find candidate match pairs

    ### YOUR CODE HERE
    ### You can use skimage or OpenCV to perform SIFT matching
    I1 = np.uint8(I1 * 255)
    I2 = np.uint8(I2 * 255)

    sift = cv2.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(I1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(I2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)

    locs1 = np.array([keypoints_1[m.queryIdx].pt for m in matches])
    locs2 = np.array([keypoints_2[m.trainIdx].pt for m in matches])

    print(locs1)
    ### END YOUR CODE
    return matches, locs1, locs2


def computeH_ransac(matches, locs1, locs2):
    # Compute the best fitting homography using RANSAC given a list of matching pairs

    ### YOUR CODE HERE
    ### You should implement this function using Numpy only

    ### END YOUR CODE

    return bestH, inliers


def compositeH(H, template, img):
    # Create a compositie image after warping the template image on top
    # of the image using homography

    # Create mask of same size as template

    # Warp mask by appropriate homography

    # Warp template by appropriate homography

    # Use mask to combine the warped template and the image

    return composite_img
