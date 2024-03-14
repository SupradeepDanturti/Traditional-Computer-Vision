# import other necessary libaries
from utils import create_line, create_mask
import cv2
import numpy as np
import matplotlib.pyplot as plt

# load the input image
img = cv2.imread("road.jpg")
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# run Canny edge detector to find edge points
edge_detector = cv2.Canny(img, 20, 200)
# cv2.imshow('edge', edge_detector)

# create a mask for ROI by calling create_mask
h, w, c = img.shape
mask = create_mask(h, w)
# cv2.imshow('mask', mask)

# extract edge points in ROI by multipling edge map with the mask
roi = edge_detector * mask
# cv2.imshow('mask ROI', roi)


# perform Hough transform
def hough_line(img):
    """https://medium.com/@alb.formaggio/implementing-the-hough-transform-from-scratch-09a56ba7316b"""
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    width, height = img.shape
    diag_len = int(np.ceil(np.sqrt(width ** 2 + height ** 2)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    accum = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img)

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            rho = round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len
            accum[int(rho), t_idx] += 1

    return accum, thetas, rhos


def find_peaks(accumulator, threshold, num_peaks):

    indices = np.argwhere(accumulator > threshold)
    peak = sorted(indices, key=lambda idx: accumulator[idx[0], idx[1]], reverse=True)

    return peak[:num_peaks]


accumulator, thetas, rhos = hough_line(roi)

# find the right lane by finding the peak in hough space
right_lane_peak = find_peaks(accumulator, threshold=100, num_peaks=1)[0]

print(right_lane_peak)


# zero out the values in accumulator around the neighborhood of the peak

def non_max_suppression(accumulator, idx, neighborhood_size=50):
    """https://gist.github.com/ri-sh/45cb32dd5c1485e273ab81468e531f09"""
    rho_idx, theta_idx = idx
    # Define the bounds of the neighborhood
    rho_start = max(0, rho_idx - neighborhood_size)
    rho_end = min(rho_idx + neighborhood_size + 1, accumulator.shape[0])
    theta_start = max(0, theta_idx - neighborhood_size)
    theta_end = min(theta_idx + neighborhood_size + 1, accumulator.shape[1])

    # Zero out the neighborhood in the accumulator
    accumulator[rho_start:rho_end, theta_start:theta_end] = 0
    return accumulator


non_max_suppression(accumulator, right_lane_peak)

# find the left lane by finding the peak in hough space
left_lane_peak = find_peaks(accumulator, threshold=50, num_peaks=1)[0]
print(left_lane_peak)

# plot the results
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# Plot the right lane
rho = rhos[right_lane_peak[0]]
theta = thetas[right_lane_peak[1]]
xs, ys = create_line(rho, theta, img)
plt.plot(xs, ys)

# Plot the left lane
rho = rhos[left_lane_peak[0]]
theta = thetas[left_lane_peak[1]]
xs, ys = create_line(rho, theta, img)
plt.plot(xs, ys)

plt.title("Detected Lanes")
plt.axis('off')
plt.show()
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()