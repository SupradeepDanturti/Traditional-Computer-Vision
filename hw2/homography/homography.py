import matplotlib.pyplot as plt
from skimage.feature import SIFT, match_descriptors
from skimage.color import rgb2gray
import numpy as np
import cv2
from skimage.transform import ProjectiveTransform, warp, rotate


def matchPics(I1, I2):
    """Ref --> https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_sift.html"""
    # Given two images I1 and I2, perform SIFT matching to find candidate match pairs

    ### YOUR CODE HERE
    ### You can use skimage or OpenCV to perform SIFT matching

    I2_gray = rgb2gray(I2)

    # I1 = cv2.rotate(I1, cv2.ROTATE_90_COUNTERCLOCKWISE)

    descriptor_extractor = SIFT()

    descriptor_extractor.detect_and_extract(I1)
    locs1 = descriptor_extractor.keypoints
    descriptors1 = descriptor_extractor.descriptors

    descriptor_extractor.detect_and_extract(I2_gray)
    locs2 = descriptor_extractor.keypoints
    descriptors2 = descriptor_extractor.descriptors

    matches = match_descriptors(descriptors1, descriptors2, max_ratio=0.6, cross_check=True)

    ### END YOUR CODE
    return matches, locs1, locs2


def computeH_ransac(matches, locs1, locs2):

    best_inliers = []

    for iteration in range(1000):
        random_indices = np.random.choice(len(matches), 4, replace=False)
        selected_points1 = np.array(
            [locs1[matches[random_indices, 0], 1], locs1[matches[random_indices, 0], 0]]).T
        selected_points2 = np.array(
            [locs2[matches[random_indices, 1], 1], locs2[matches[random_indices, 1], 0]]).T

        point_count = selected_points1.shape[0]
        matrix_A = np.zeros((2 * point_count, 9))

        for idx in range(point_count):
            px1, py1 = selected_points1[idx]
            px2, py2 = selected_points2[idx]
            matrix_A[2 * idx] = [-px1, -py1, -1, 0, 0, 0, px2 * px1, px2 * py1, px2]
            matrix_A[2 * idx + 1] = [0, 0, 0, -px1, -py1, -1, py2 * px1, py2 * py1, py2]

        _, _, matrix_V = np.linalg.svd(matrix_A)
        homography_matrix = matrix_V[-1].reshape(3, 3)
        homography_matrix /= homography_matrix[2, 2]

        loc1_homogenous = np.hstack(
            [locs1[matches[:, 0], 1][:, np.newaxis], locs1[matches[:, 0], 0][:, np.newaxis],
             np.ones((len(matches), 1))])
        loc2_homogenous = np.hstack(
            [locs2[matches[:, 1], 1][:, np.newaxis], locs2[matches[:, 1], 0][:, np.newaxis],
             np.ones((len(matches), 1))])

        estimated_loc2 = (homography_matrix @ loc1_homogenous.T).T
        estimated_loc2 = estimated_loc2[:, :2] / estimated_loc2[:, 2][:, np.newaxis]

        distances = np.sqrt(np.sum((loc2_homogenous[:, :2] - estimated_loc2) ** 2, axis=1))

        inliers = np.where(distances < 5)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            bestH = homography_matrix

    print("Best Homography Matrix:\n", bestH)

    return bestH, best_inliers


def compositeH(H, template, img):
    # Create a compositie image after warping the template image on top
    # of the image using homography


    H_inv = np.linalg.inv(H)
    transform = ProjectiveTransform(H_inv)

    warped_template = warp(template, transform, output_shape=img.shape)

    mask = np.ones(template.shape[:2], dtype=np.float32)

    warped_mask = warp(mask, transform, output_shape=img.shape[:2])
    warped_mask = warped_mask > 0

    if len(img.shape) == 3 and img.shape[2] == 3:
        warped_mask = np.expand_dims(warped_mask, axis=-1)
        warped_mask = np.repeat(warped_mask, 3, axis=-1)

    # Use mask to combine the warped template and the image
    composite_img = img * (1 - warped_mask) + warped_template * warped_mask
    return composite_img
