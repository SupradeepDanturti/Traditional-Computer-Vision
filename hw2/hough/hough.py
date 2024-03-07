# import other necessary libaries
from utils import create_line, create_mask
import cv2
import numpy as np

# load the input image
img = cv2.imread("road.jpg")
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# run Canny edge detector to find edge points
edge_detector = cv2.Canny(img, 20, 200)
cv2.imshow('edge', edge_detector)

# create a mask for ROI by calling create_mask
h, w, c = img.shape
mask = create_mask(h, w)
cv2.imshow('mask', mask)

# extract edge points in ROI by multipling edge map with the mask
roi = edge_detector * mask
cv2.imshow('mask ROI', roi)


# perform Hough transform
HoughLinesP


# find the right lane by finding the peak in hough space

# zero out the values in accumulator around the neighborhood of the peak

# find the left lane by finding the peak in hough space

# plot the results
for rho, theta in peaks:
    xs, ys = create_line(rho, theta, img)
    # for x, y in zip(xs, ys):
    #     cv2.line(img, (xs[0], ys[0]), (xs[-1], ys[-1]), (255, 0, 0), 2)

cv2.imshow('Detected Lanes', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
