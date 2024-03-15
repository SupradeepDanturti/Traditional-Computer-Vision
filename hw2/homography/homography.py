import matplotlib.pyplot as plt
from skimage.feature import SIFT, match_descriptors
from skimage.color import rgb2gray
import numpy as np
import cv2


def matchPics(I1, I2):
    # Given two images I1 and I2, perform SIFT matching to find candidate match pairs

    ### YOUR CODE HERE
    ### You can use skimage or OpenCV to perform SIFT matching
    I2_gray = rgb2gray(I2)
    # I1 = cv2.rotate(I1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # Use skimage SIFT to detect and extract features
    descriptor_extractor = SIFT()

    descriptor_extractor.detect_and_extract(I1)
    locs1 = descriptor_extractor.keypoints
    descriptors1 = descriptor_extractor.descriptors

    descriptor_extractor.detect_and_extract(I2_gray)
    locs2 = descriptor_extractor.keypoints
    descriptors2 = descriptor_extractor.descriptors

    # Match descriptors using skimage's match_descriptors
    matches = match_descriptors(descriptors1, descriptors2, max_ratio=0.56, cross_check=True)

    ### END YOUR CODE
    return matches, locs1, locs2


def computeH(A):
    # Compute the homography matrix using SVD
    _, _, V = np.linalg.svd(A)
    H = V[-1].reshape((3, 3))
    return H / H[2, 2]  # Normalize so that the bottom-right value is 1

def computeH_ransac(matches, locs1, locs2):
    num_matches = matches.shape[0]
    max_inliers = 0
    bestH = np.empty((3, 3))
    best_inliers = np.array([])

    # Parameters for RANSAC
    num_iterations = 200
    tolerance = 10  # Tolerance for a point to be considered an inlier

    for _ in range(num_iterations):
        # Randomly select four matching pairs
        indices = np.random.choice(num_matches, 4, replace=False)
        points1 = locs1[matches[indices, 0]]
        points2 = locs2[matches[indices, 1]]

        # Construct the matrix A using the points
        A = np.zeros((8, 9))
        for i in range(4):
            x1, y1 = points1[i]
            x2, y2 = points2[i]
            A[2*i] = [-x1, -y1, -1, 0, 0, 0, x2*x1, x2*y1, x2]
            A[2*i + 1] = [0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2]

        # Estimate the homography
        H = computeH(A)

        # Apply homography on all points in locs2
        locs2_homogenous = np.concatenate([locs2[matches[:, 1]], np.ones((num_matches, 1))], axis=1)
        locs2_transformed = (H @ locs2_homogenous.T).T
        locs2_transformed /= locs2_transformed[:, [2]]

        # Calculate distances and find inliers
        locs1_points = locs1[matches[:, 0]]
        distances = np.linalg.norm(locs1_points - locs2_transformed[:, :2], axis=1)
        inliers = np.where(distances < tolerance)[0]

        # Update the best homography if the current one is better
        if len(inliers) > max_inliers:
            bestH = H
            max_inliers = len(inliers)
            best_inliers = inliers

    return bestH, best_inliers


def compositeH(H, template, img):
    # Create a compositie image after warping the template image on top
    # of the image using homography
    # Ensure that the template has the same number of color channels as img
    if len(template.shape) != len(img.shape):
        if len(img.shape) == 3:  # img is a color image
            # Convert template to a color image
            template = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)

    # Create mask of same size as template with single channel
    mask = np.ones(template.shape[:2], dtype=np.uint8)

    # Warp mask by the inverse of the homography
    H_inv = np.linalg.inv(H)
    warped_mask = cv2.warpPerspective(mask, H_inv, (img.shape[1], img.shape[0]))

    # Warp template by the inverse of the homography
    warped_template = cv2.warpPerspective(template, H_inv, (img.shape[1], img.shape[0]))

    # Invert the mask
    mask_inv = cv2.bitwise_not(warped_mask)

    # Create a hole in the image where the template will be placed
    img_hole = cv2.bitwise_and(img, img, mask=mask_inv)

    # Combine the hole image and the warped template
    composite_img = cv2.bitwise_or(img_hole, warped_template)

    return composite_img
