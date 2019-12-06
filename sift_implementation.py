# ---------------------------------------------------------------#
# __name__ = "SIFT keypoint detection implementation in python"
# __author__ = "Shyama P"
# __version__ = "1.0"
# __email__ = "shyamap95@gmail.com"
# __status__ = "Development"
# ---------------------------------------------------------------#
import numpy as np
from numpy.linalg import norm
from scipy import misc
from scipy.ndimage import gaussian_filter
import cv2

from constant import *
from keypoints_class import *


# Get a set of 4 successively downsampled images from a given image
# Downsampling rate is defined by sample_rate. It is set to 2 by default
def get_downsampled_images(image, sample_rate=2):
    # downsample once by sample rate
    sample1 = image

    # downsample twice by sample rate
    sample2 = sample1[0::sample_rate, 0::sample_rate]

    # downsample thrice by sample rate
    sample3 = sample2[0::sample_rate, 0::sample_rate]

    # downsample four times by sample rate
    sample4 = sample3[0::sample_rate, 0::sample_rate]

    return sample1, sample2, sample3, sample4


# Obtain a blurred version of the image using a Gaussian kernel with standard deviation sd
def gaussian_blur(image, sd):
    # Blur given image using Gaussian filter from scipy library
    output = gaussian_filter(image, sigma=sd)
    return output


# Generate a scale space for the image with four octaves
# Each octave has OCTAVE_DIMENSION (defined in constant.py) images
def generate_scale_space(image):
    (base_image_octave1, base_image_octave2, base_image_octave3, base_image_octave4) = \
        get_downsampled_images(image, SAMPLE_RATE)

    sigma1 = SIGMA
    sigma2 = 2 * SIGMA
    sigma3 = 4 * SIGMA
    sigma4 = 8 * SIGMA

    s = OCTAVE_DIMENSION - 3
    k = np.power(2, 1 / s)

    octave1_dim = (base_image_octave1.shape[0], base_image_octave1.shape[1], OCTAVE_DIMENSION)
    octave1 = np.zeros(octave1_dim)
    for i in range(0, OCTAVE_DIMENSION):
        octave1[:, :, i] = gaussian_blur(base_image_octave1, sigma1 * (k ** i))

    octave2_dim = (base_image_octave2.shape[0], base_image_octave2.shape[1], OCTAVE_DIMENSION)
    octave2 = np.zeros(octave2_dim)
    for i in range(0, OCTAVE_DIMENSION):
        octave2[:, :, i] = gaussian_blur(base_image_octave2, sigma2 * (k ** i))

    octave3_dim = (base_image_octave3.shape[0], base_image_octave3.shape[1], OCTAVE_DIMENSION)
    octave3 = np.zeros(octave3_dim)
    for i in range(0, OCTAVE_DIMENSION):
        octave3[:, :, i] = gaussian_blur(base_image_octave3, sigma3 * (k ** i))

    octave4_dim = (base_image_octave4.shape[0], base_image_octave4.shape[1], OCTAVE_DIMENSION)
    octave4 = np.zeros(octave4_dim)
    for i in range(0, OCTAVE_DIMENSION):
        octave4[:, :, i] = gaussian_blur(base_image_octave4, sigma4 * (k ** i))

    return octave1, octave2, octave3, octave4


# Generate Difference of Gaussian for a each scale of a 4 scale scale-space
def generate_DoG(scale1, scale2, scale3, scale4):
    DoG1_dim = (scale1.shape[0], scale1.shape[1], OCTAVE_DIMENSION - 1)
    DoG1 = np.zeros(DoG1_dim)
    for i in range(0, OCTAVE_DIMENSION - 1):
        DoG1[:, :, i] = scale1[:, :, i + 1] - scale1[:, :, i]

    DoG2_dim = (scale2.shape[0], scale2.shape[1], OCTAVE_DIMENSION - 1)
    DoG2 = np.zeros(DoG2_dim)
    for i in range(0, OCTAVE_DIMENSION - 1):
        DoG2[:, :, i] = scale2[:, :, i + 1] - scale2[:, :, i]

    DoG3_dim = (scale3.shape[0], scale3.shape[1], OCTAVE_DIMENSION - 1)
    DoG3 = np.zeros(DoG3_dim)
    for i in range(0, OCTAVE_DIMENSION - 1):
        DoG3[:, :, i] = scale3[:, :, i + 1] - scale3[:, :, i]

    DoG4_dim = (scale4.shape[0], scale4.shape[1], OCTAVE_DIMENSION - 1)
    DoG4 = np.zeros(DoG4_dim)
    for i in range(0, OCTAVE_DIMENSION - 1):
        DoG4[:, :, i] = scale4[:, :, i + 1] - scale4[:, :, i]

    return DoG1, DoG2, DoG3, DoG4


def determine_initial_keypoints(DoG1, DoG2, DoG3, DoG4):
    (M, N) = (DoG1.shape[0], DoG1.shape[1])
    keypoint1 = []

    for k in range(1, 3):
        for i in range(1, M - 1):
            for j in range(1, N - 1):

                if np.abs(DoG1[i, j, k]) < THRESHOLD:
                    continue

                values = np.concatenate((DoG1[i - 1:i + 2, j - 1:j + 2, k - 1],
                        np.concatenate((DoG1[i - 1:i + 2, j - 1:j + 2, k], DoG1[i - 1:i + 2, j - 1:j + 2, k + 1]))))

                max_value = np.max(values)
                min_value = np.min(values)

                if (DoG1[i, j, k] == max_value) or (DoG1[i, j, k] == min_value):
                    # Find first derivatives approximated as difference (normalized)
                    dx = (DoG1[i, j + 1, k] - DoG1[i, j - 1, k]) * 0.5 / 255
                    dy = (DoG1[i + 1, j, k] - DoG1[i - 1, j, k]) * 0.5 / 255
                    ds = (DoG1[i, j, k + 1] - DoG1[i, j, k - 1]) * 0.5 / 255
                    # gradient matrix del(D)
                    dD = np.matrix([[dx], [dy], [ds]])

                    # Find second derivatives approximated as difference (normalised)
                    dxx = (DoG1[i, j + 1, k] + DoG1[i, j - 1, k] - 2 * DoG1[i, j, k]) * 1.0 / 255
                    dyy = (DoG1[i + 1, j, k] + DoG1[i - 1, j, k] - 2 * DoG1[i, j, k]) * 1.0 / 255
                    dss = (DoG1[i, j, k + 1] + DoG1[i, j, k - 1] - 2 * DoG1[i, j, k]) * 1.0 / 255
                    dxy = (DoG1[i + 1, j + 1, k] - DoG1[i + 1, j - 1, k] - DoG1[i - 1, j + 1, k] +
                           DoG1[i - 1, j - 1, k]) * 0.25 / 255
                    dxs = (DoG1[i, j + 1, k + 1] - DoG1[i, j - 1, k + 1] - DoG1[i, j + 1, k - 1] +
                           DoG1[i, j - 1, k - 1]) * 0.25 / 255
                    dys = (DoG1[i + 1, j, k + 1] - DoG1[i - 1, j, k + 1] - DoG1[i + 1, j, k - 1] +
                           DoG1[i - 1, j, k - 1]) * 0.25 / 255

                    # Hessian matrix - H
                    H = np.matrix([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])

                    # Solve for x hat by setting d(D)/dx = 0 and calculate D(x hat) : D is DoG
                    x_hat = np.linalg.lstsq(H, dD, rcond=None)[0]
                    D_x_hat = DoG1[i, j, k] + 0.5 * np.dot(dD.transpose(), x_hat)

                    alpha = ((RADIUS_OF_CURVATURE + 1) ** 2)
                    trace_H_sq = (dxx + dyy) ** 2
                    det_H = dxx * dyy - (dxy ** 2)

                    if (trace_H_sq * RADIUS_OF_CURVATURE < alpha * det_H) and (np.abs(x_hat[0]) < 0.5) and \
                            (np.abs(x_hat[1]) < 0.5) and (np.abs(x_hat[2]) < 0.5) and (np.abs(D_x_hat) > 0.03):
                        temp_keypoint = KEYPOINT(i=i, j=j, octave=1, DoG=k, x=j+x_hat[0], y=i+x_hat[1])
                        keypoint1.append(temp_keypoint)

    (M, N) = (DoG2.shape[0], DoG2.shape[1])
    keypoint2 = []

    for k in range(1, 3):
        for i in range(1, M - 1):
            for j in range(1, N - 1):

                if np.abs(DoG2[i, j, k]) < THRESHOLD:
                    continue

                values = np.concatenate((DoG2[i - 1:i + 2, j - 1:j + 2, k - 1],
                            np.concatenate((DoG2[i - 1:i + 2, j - 1:j + 2, k], DoG2[i - 1:i + 2, j - 1:j + 2, k + 1]))))
                max_value = np.max(values)
                min_value = np.min(values)

                if (DoG2[i, j, k] == max_value) or (DoG2[i, j, k] == min_value):
                    # Find first derivatives approximated as difference (normalized)
                    dx = (DoG2[i, j + 1, k] - DoG2[i, j - 1, k]) * 0.5 / 255
                    dy = (DoG2[i + 1, j, k] - DoG2[i - 1, j, k]) * 0.5 / 255
                    ds = (DoG2[i, j, k + 1] - DoG2[i, j, k - 1]) * 0.5 / 255
                    # gradient matrix del(D)
                    dD = np.matrix([[dx], [dy], [ds]])

                    # Find second derivatives approximated as difference (normalised)
                    dxx = (DoG2[i, j + 1, k] + DoG2[i, j - 1, k] - 2 * DoG2[i, j, k]) * 1.0 / 255
                    dyy = (DoG2[i + 1, j, k] + DoG2[i - 1, j, k] - 2 * DoG2[i, j, k]) * 1.0 / 255
                    dss = (DoG2[i, j, k + 1] + DoG2[i, j, k - 1] - 2 * DoG2[i, j, k]) * 1.0 / 255
                    dxy = (DoG2[i + 1, j + 1, k] - DoG2[i + 1, j - 1, k] - DoG2[i - 1, j + 1, k] +
                           DoG2[i - 1, j - 1, k]) * 0.25 / 255
                    dxs = (DoG2[i, j + 1, k + 1] - DoG2[i, j - 1, k + 1] - DoG2[i, j + 1, k - 1] +
                           DoG2[i, j - 1, k - 1]) * 0.25 / 255
                    dys = (DoG2[i + 1, j, k + 1] - DoG2[i - 1, j, k + 1] - DoG2[i + 1, j, k - 1] +
                           DoG2[i - 1, j, k - 1]) * 0.25 / 255

                    # Hessian matrix - H
                    H = np.matrix([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])

                    x_hat = np.linalg.lstsq(H, dD, rcond=None)[0]
                    D_x_hat = DoG2[i, j, k] + 0.5 * np.dot(dD.transpose(), x_hat)

                    alpha = ((RADIUS_OF_CURVATURE + 1) ** 2)
                    trace_H_sq = (dxx + dyy) ** 2
                    det_H = dxx * dyy - (dxy ** 2)

                    if (trace_H_sq * RADIUS_OF_CURVATURE < alpha * det_H) and (np.abs(x_hat[0]) < 0.5) and (
                            np.abs(x_hat[1]) < 0.5) and (np.abs(x_hat[2]) < 0.5) and (np.abs(D_x_hat) > 0.03):
                        temp_keypoint = KEYPOINT(i=i, j=j, octave=2, DoG=k, x=j+x_hat[0], y=i+x_hat[1])
                        keypoint2.append(temp_keypoint)

    (M, N) = (DoG3.shape[0], DoG3.shape[1])
    keypoint3 = []

    for k in range(1, 3):
        for i in range(1, M - 1):
            for j in range(1, N - 1):

                if np.abs(DoG3[i, j, k]) < THRESHOLD:
                    continue

                values = np.concatenate((DoG3[i - 1:i + 2, j - 1:j + 2, k - 1],
                                         np.concatenate(
                                             (DoG3[i - 1:i + 2, j - 1:j + 2, k], DoG3[i - 1:i + 2, j - 1:j + 2, k + 1]))))
                max_value = np.max(values)
                min_value = np.min(values)

                if (DoG3[i, j, k] == max_value) or (DoG3[i, j, k] == min_value):
                    # Find first derivatives approximated as difference (normalized)
                    dx = (DoG3[i, j + 1, k] - DoG3[i, j - 1, k]) * 0.5 / 255
                    dy = (DoG3[i + 1, j, k] - DoG3[i - 1, j, k]) * 0.5 / 255
                    ds = (DoG3[i, j, k + 1] - DoG3[i, j, k - 1]) * 0.5 / 255
                    # gradient matrix del(D)
                    dD = np.matrix([[dx], [dy], [ds]])

                    # Find second derivatives approximated as difference (normalised)
                    dxx = (DoG3[i, j + 1, k] + DoG3[i, j - 1, k] - 2 * DoG3[i, j, k]) * 1.0 / 255
                    dyy = (DoG3[i + 1, j, k] + DoG3[i - 1, j, k] - 2 * DoG3[i, j, k]) * 1.0 / 255
                    dss = (DoG3[i, j, k + 1] + DoG3[i, j, k - 1] - 2 * DoG3[i, j, k]) * 1.0 / 255
                    dxy = (DoG3[i + 1, j + 1, k] - DoG3[i + 1, j - 1, k] - DoG3[i - 1, j + 1, k] +
                           DoG3[i - 1, j - 1, k]) * 0.25 / 255
                    dxs = (DoG3[i, j + 1, k + 1] - DoG3[i, j - 1, k + 1] - DoG3[i, j + 1, k - 1] +
                           DoG3[i, j - 1, k - 1]) * 0.25 / 255
                    dys = (DoG3[i + 1, j, k + 1] - DoG3[i - 1, j, k + 1] - DoG3[i + 1, j, k - 1] +
                           DoG3[i - 1, j, k - 1]) * 0.25 / 255

                    # Hessian matrix - H
                    H = np.matrix([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])

                    x_hat = np.linalg.lstsq(H, dD, rcond=None)[0]
                    D_x_hat = DoG3[i, j, k] + 0.5 * np.dot(dD.transpose(), x_hat)

                    alpha = ((RADIUS_OF_CURVATURE + 1) ** 2)
                    trace_H_sq = (dxx + dyy) ** 2
                    det_H = dxx * dyy - (dxy ** 2)

                    if (trace_H_sq * RADIUS_OF_CURVATURE < alpha * det_H) and (np.abs(x_hat[0]) < 0.5) and (
                            np.abs(x_hat[1]) < 0.5) and (
                            np.abs(x_hat[2]) < 0.5) and (np.abs(D_x_hat) > 0.03):
                        temp_keypoint = KEYPOINT(i=i, j=j, octave=3, DoG=k, x=j+x_hat[0], y=i+x_hat[1])
                        keypoint3.append(temp_keypoint)

    (M, N) = (DoG4.shape[0], DoG4.shape[1])
    keypoint4 = []

    for k in range(1, 3):
        for i in range(1, M - 1):
            for j in range(1, N - 1):

                if np.abs(DoG4[i, j, k]) < THRESHOLD:
                    continue

                values = np.concatenate((DoG4[i - 1:i + 2, j - 1:j + 2, k - 1],
                                         np.concatenate(
                                             (DoG4[i - 1:i + 2, j - 1:j + 2, k], DoG4[i - 1:i + 2, j - 1:j + 2, k + 1]))))
                max_value = np.max(values)
                min_value = np.min(values)

                if (DoG4[i, j, k] == max_value) or (DoG4[i, j, k] == min_value):
                    # Find first derivatives approximated as difference (normalized)
                    dx = (DoG4[i, j + 1, k] - DoG4[i, j - 1, k]) * 0.5 / 255
                    dy = (DoG4[i + 1, j, k] - DoG4[i - 1, j, k]) * 0.5 / 255
                    ds = (DoG4[i, j, k + 1] - DoG4[i, j, k - 1]) * 0.5 / 255
                    # gradient matrix del(D)
                    dD = np.matrix([[dx], [dy], [ds]])

                    # Find second derivatives approximated as difference (normalised)
                    dxx = (DoG4[i, j + 1, k] + DoG4[i, j - 1, k] - 2 * DoG4[i, j, k]) * 1.0 / 255
                    dyy = (DoG4[i + 1, j, k] + DoG4[i - 1, j, k] - 2 * DoG4[i, j, k]) * 1.0 / 255
                    dss = (DoG4[i, j, k + 1] + DoG4[i, j, k - 1] - 2 * DoG4[i, j, k]) * 1.0 / 255
                    dxy = (DoG4[i + 1, j + 1, k] - DoG4[i + 1, j - 1, k] - DoG4[i - 1, j + 1, k] +
                           DoG4[i - 1, j - 1, k]) * 0.25 / 255
                    dxs = (DoG4[i, j + 1, k + 1] - DoG4[i, j - 1, k + 1] - DoG4[i, j + 1, k - 1] +
                           DoG4[i, j - 1, k - 1]) * 0.25 / 255
                    dys = (DoG4[i + 1, j, k + 1] - DoG4[i - 1, j, k + 1] - DoG4[i + 1, j, k - 1] +
                           DoG4[i - 1, j, k - 1]) * 0.25 / 255

                    # Hessian matrix - H
                    H = np.matrix([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])

                    x_hat = np.linalg.lstsq(H, dD, rcond=None)[0]
                    D_x_hat = DoG4[i, j, k] + 0.5 * np.dot(dD.transpose(), x_hat)

                    alpha = ((RADIUS_OF_CURVATURE + 1) ** 2)
                    trace_H_sq = (dxx + dyy) ** 2
                    det_H = dxx * dyy - (dxy ** 2)

                    if (trace_H_sq * RADIUS_OF_CURVATURE < alpha * det_H) and (np.abs(x_hat[0]) < 0.5) and (
                            np.abs(x_hat[1]) < 0.5) and (
                            np.abs(x_hat[2]) < 0.5) and (np.abs(D_x_hat) > 0.03):
                        temp_keypoint = KEYPOINT(i=i, j=j, octave=4, DoG=k, x=j+x_hat[0], y=i+x_hat[1])
                        keypoint4.append(temp_keypoint)

    return keypoint1, keypoint2, keypoint3, keypoint4


def compute_magnitude_angle(scale1, scale2, scale3, scale4):
    # scale 1
    (M, N) = (scale1.shape[0], scale1.shape[1])
    scale_x_1p = np.zeros_like(scale1[:, :, :OCTAVE_DIMENSION - 3])
    scale_x_1n = np.zeros_like(scale1[:, :, :OCTAVE_DIMENSION - 3])
    scale_y_1p = np.zeros_like(scale1[:, :, :OCTAVE_DIMENSION - 3])
    scale_y_1n = np.zeros_like(scale1[:, :, :OCTAVE_DIMENSION - 3])

    scale_x_1n[:, 1:, :] = scale1[:, :N-1, 1:OCTAVE_DIMENSION-2]
    scale_x_1p[:, :N-1, :] = scale1[:, 1:, 1:OCTAVE_DIMENSION-2]
    scale_y_1n[1:, :, :] = scale1[:M-1, :, 1:OCTAVE_DIMENSION-2]
    scale_y_1p[:M-1, :, :] = scale1[1:, :, 1:OCTAVE_DIMENSION-2]

    magnitude1 = np.sqrt(np.square(scale_x_1p - scale_x_1n) + np.square(scale_y_1p - scale_y_1n))
    theta1 = np.arctan2(scale_y_1p - scale_y_1n, scale_x_1p - scale_x_1n) * 180 / np.pi
    theta1 = np.mod(theta1 + 360, 360 * np.ones_like(theta1))

    # scale 2
    (M, N) = (scale2.shape[0], scale2.shape[1])
    scale_x_1p = np.zeros_like(scale2[:, :, :OCTAVE_DIMENSION - 3])
    scale_x_1n = np.zeros_like(scale2[:, :, :OCTAVE_DIMENSION - 3])
    scale_y_1p = np.zeros_like(scale2[:, :, :OCTAVE_DIMENSION - 3])
    scale_y_1n = np.zeros_like(scale2[:, :, :OCTAVE_DIMENSION - 3])

    scale_x_1n[:, 1:, :] = scale2[:, :N - 1, 1:OCTAVE_DIMENSION - 2]
    scale_x_1p[:, :N - 1, :] = scale2[:, 1:, 1:OCTAVE_DIMENSION - 2]
    scale_y_1n[1:, :, :] = scale2[:M - 1, :, 1:OCTAVE_DIMENSION - 2]
    scale_y_1p[:M - 1, :, :] = scale2[1:, :, 1:OCTAVE_DIMENSION - 2]

    magnitude2 = np.sqrt(np.square(scale_x_1p - scale_x_1n) + np.square(scale_y_1p - scale_y_1n))
    theta2 = np.arctan2(scale_y_1p - scale_y_1n, scale_x_1p - scale_x_1n) * 180 / np.pi
    theta2 = np.mod(theta2 + 360, 360 * np.ones_like(theta2))

    # scale 3
    (M, N) = (scale3.shape[0], scale3.shape[1])
    scale_x_1p = np.zeros_like(scale3[:, :, :OCTAVE_DIMENSION - 3])
    scale_x_1n = np.zeros_like(scale3[:, :, :OCTAVE_DIMENSION - 3])
    scale_y_1p = np.zeros_like(scale3[:, :, :OCTAVE_DIMENSION - 3])
    scale_y_1n = np.zeros_like(scale3[:, :, :OCTAVE_DIMENSION - 3])

    scale_x_1n[:, 1:, :] = scale3[:, :N - 1, 1:OCTAVE_DIMENSION - 2]
    scale_x_1p[:, :N - 1, :] = scale3[:, 1:, 1:OCTAVE_DIMENSION - 2]
    scale_y_1n[1:, :, :] = scale3[:M - 1, :, 1:OCTAVE_DIMENSION - 2]
    scale_y_1p[:M - 1, :, :] = scale3[1:, :, 1:OCTAVE_DIMENSION - 2]

    magnitude3 = np.sqrt(np.square(scale_x_1p - scale_x_1n) + np.square(scale_y_1p - scale_y_1n))
    theta3 = np.arctan2(scale_y_1p - scale_y_1n, scale_x_1p - scale_x_1n) * 180 / np.pi
    theta3 = np.mod(theta3 + 360, 360 * np.ones_like(theta3))

    # scale 4
    (M, N) = (scale4.shape[0], scale4.shape[1])
    scale_x_1p = np.zeros_like(scale4[:, :, :OCTAVE_DIMENSION - 3])
    scale_x_1n = np.zeros_like(scale4[:, :, :OCTAVE_DIMENSION - 3])
    scale_y_1p = np.zeros_like(scale4[:, :, :OCTAVE_DIMENSION - 3])
    scale_y_1n = np.zeros_like(scale4[:, :, :OCTAVE_DIMENSION - 3])

    scale_x_1n[:, 1:, :] = scale4[:, :N - 1, 1:OCTAVE_DIMENSION - 2]
    scale_x_1p[:, :N - 1, :] = scale4[:, 1:, 1:OCTAVE_DIMENSION - 2]
    scale_y_1n[1:, :, :] = scale4[:M - 1, :, 1:OCTAVE_DIMENSION - 2]
    scale_y_1p[:M - 1, :, :] = scale4[1:, :, 1:OCTAVE_DIMENSION - 2]

    magnitude4 = np.sqrt(np.square(scale_x_1p - scale_x_1n) + np.square(scale_y_1p - scale_y_1n))
    theta4 = np.arctan2(scale_y_1p - scale_y_1n, scale_x_1p - scale_x_1n) * 180 / np.pi
    theta4 = np.mod(theta4 + 360, 360 * np.ones_like(theta4))

    return magnitude1, theta1, magnitude2, theta2, magnitude3, theta3, magnitude4, theta4


def compute_histogram(magnitude, theta):
    histogram = np.zeros(36)

    for i in range(0, theta.shape[0]):
        for j in range(0, theta.shape[1]):
            histogram[np.uint8(theta[i, j])] = histogram[np.uint8(theta[i, j])] + magnitude[i, j]

    return histogram


def compute_descriptor(scale1, M1, T1, M2, T2, M3, T3, M4, T4, keypoints):
    magnitude = np.zeros((scale1.shape[0], scale1.shape[1], 8))
    orientation = np.zeros((scale1.shape[0], scale1.shape[1], 8))

    for i in range(0, 2):
        magnitude[:, :, i] = (M1[:, :, i]).astype(float)
        orientation[:, :, i] = (T1[:, :, i]).astype(int)

    for i in range(0, 2):
        magnitude[:, :, i + 2] = misc.imresize(M2[:, :, i], (scale1.shape[0], scale1.shape[1]), "bilinear").astype(
            int)
        orientation[:, :, i + 2] = misc.imresize(T2[:, :, i], (scale1.shape[0], scale1.shape[1]), "bilinear").astype(
            int)

    for i in range(0, 2):
        magnitude[:, :, i + 4] = misc.imresize(M3[:, :, i], (scale1.shape[0], scale1.shape[1]), "bilinear").astype(
            int)
        orientation[:, :, i + 4] = misc.imresize(T3[:, :, i], (scale1.shape[0], scale1.shape[1]), "bilinear").astype(
            int)

    for i in range(0, 2):
        magnitude[:, :, i + 6] = misc.imresize(M4[:, :, i], (scale1.shape[0], scale1.shape[1]), "bilinear").astype(
            int)
        orientation[:, :, i + 6] = misc.imresize(T4[:, :, i], (scale1.shape[0], scale1.shape[1]), "bilinear").astype(
            int)

    descriptors = np.zeros((len(keypoints), 128))
    k = np.sqrt(2)
    kvectotal = np.uint8(np.array([SIGMA, SIGMA * k, SIGMA * (k ** 2), SIGMA * (k ** 3), SIGMA * (k ** 4), SIGMA * (k ** 5),
                          SIGMA * (k ** 6), SIGMA * (k ** 7)]) * 1000)

    for i in range(0, len(keypoints)):
        x0 = np.int(keypoints[i].x)
        y0 = np.int(keypoints[i].y)
        value = np.uint8(keypoints[i].sigma * 1000)
        scale_idx = np.int8(np.argwhere(kvectotal == value))[0][0]

        gaussian_window = gaussian_blur(magnitude[x0-16:x0+16, y0-16:y0+16, scale_idx], keypoints[i].sigma)

        if type(gaussian_window.size) == int and gaussian_window.size < 1024:
            continue

        for x in range(-8, 8):
            for y in range(-8, 8):
                theta = keypoints[i].angle * np.pi / 180.0

                xrot = np.int(np.round((np.cos(theta) * x) - (np.sin(theta) * y)))
                yrot = np.int(np.round((np.sin(theta) * x) + (np.cos(theta) * y)))

                x_ = np.int8(x0 + xrot)
                y_ = np.int8(y0 + yrot)

                weight = gaussian_window[xrot+8, yrot+8]

                angle = orientation[x_, y_, scale_idx] - keypoints[i].angle
                angle = np.int8(angle/10)

                if angle < 0:
                    angle = 36 + angle

                bin_idx = np.clip(np.floor((8.0 / 36) * angle), 0, 7).astype(int)
                bin_idx = bin_idx[0]
                descriptors[i, 32 * int((x + 8) / 4) + 8 * int((y + 8) / 4) + bin_idx] += weight

        descriptors[i, :] = descriptors[i, :] / norm(descriptors[i, :])
        descriptors[i, :] = np.clip(descriptors[i, :], 0, 0.2)

        descriptors[i, :] = descriptors[i, :] / norm(descriptors[i, :])

    return descriptors
    # return np.uint8(descriptors)


def compute_final_keypoints_descriptors(keyset1, keyset2, keyset3, keyset4, scale1, scale2, scale3, scale4):
    keypoint_list = []
    descriptor_list = []

    # Compute magnitude and orientation of all points in scale space
    (M1, T_1, M2, T_2, M3, T_3, M4, T_4) = compute_magnitude_angle(scale1, scale2, scale3, scale4)

    # Covert 0 to 360 degrees into 36 bins
    T1 = np.floor(T_1 / 10)
    T2 = np.floor(T_2 / 10)
    T3 = np.floor(T_3 / 10)
    T4 = np.floor(T_4 / 10)

    new_keypoints1 = []

    # Scale 1
    for key in keyset1:
        (x, y) = key.j, key.i
        k = np.uint8(key.DoG - 1)
        minx = np.uint8(min(8, x))
        miny = np.uint8(min(8, y))

        theta_neighbourhood = T1[x - minx: x + minx + 1, y - miny: y + miny + 1, k]
        magnitude_neighbourhood = M1[x - minx: x + minx + 1, y - miny: y + miny + 1, k]
        magnitude_neighbourhood = gaussian_blur(magnitude_neighbourhood, 1.5 * SIGMA * (np.sqrt(2))**(k+1))

        histogram = compute_histogram(magnitude_neighbourhood, theta_neighbourhood)
        histogram_sorted = histogram.copy()
        histogram_sorted.sort()

        key.sigma = SIGMA * (np.sqrt(2)) ** (k + 1)

        if histogram_sorted[-2] > 0.8 * histogram_sorted[-1]:
            key.angle = np.argwhere(histogram == histogram_sorted[-1])[0] * 10
            new_keypoints1.append(key)

            for h in range(2, 36):
                if histogram_sorted[-h] > 0.8 * histogram_sorted[-1]:
                    key.angle = np.argwhere(histogram == histogram_sorted[-h])[0] * 10
                    new_keypoints1.append(key)
                else:
                    break
        else:
            key.angle = np.argwhere(histogram == histogram_sorted[-1])[0] * 10
            new_keypoints1.append(key)

    # Scale 2
    new_keypoints2 = []
    for key in keyset2:
        (x, y) = key.j, key.i
        k = np.uint8(key.DoG - 1)
        minx = np.uint8(min(8, x))
        miny = np.uint8(min(8, y))

        theta_neighbourhood = T2[x - minx: x + minx + 1, y - miny: y + miny + 1, k]
        magnitude_neighbourhood = M2[x - minx: x + minx + 1, y - miny: y + miny + 1, k]
        magnitude_neighbourhood = gaussian_blur(magnitude_neighbourhood, 1.5 * 2 * SIGMA * (np.sqrt(2))**(k+1))

        histogram = compute_histogram(magnitude_neighbourhood, theta_neighbourhood)
        histogram_sorted = histogram.copy()
        histogram_sorted.sort()

        key.sigma = 2 * SIGMA * (np.sqrt(2)) ** (k + 1)

        if histogram_sorted[-2] > 0.8 * histogram_sorted[-1]:
            key.angle = np.argwhere(histogram == histogram_sorted[-1])[0] * 10
            new_keypoints2.append(key)

            for h in range(2, 36):
                if histogram_sorted[-h] > 0.8 * histogram_sorted[-1]:
                    key.angle = np.argwhere(histogram == histogram_sorted[-h])[0] * 10
                    new_keypoints2.append(key)
                else:
                    break
        else:
            key.angle = np.argwhere(histogram == histogram_sorted[-1])[0] * 10
            new_keypoints2.append(key)

    # Scale 3
    new_keypoints3 = []
    for key in keyset3:
        (x, y) = key.j, key.i
        k = np.uint8(key.DoG - 1)
        minx = np.uint8(min(8, x))
        miny = np.uint8(min(8, y))

        theta_neighbourhood = T3[x - minx: x + minx + 1, y - miny: y + miny + 1, k]
        magnitude_neighbourhood = M3[x - minx: x + minx + 1, y - miny: y + miny + 1, k]
        magnitude_neighbourhood = gaussian_blur(magnitude_neighbourhood, 1.5 * 4 * SIGMA * (np.sqrt(2))**(k+1))

        histogram = compute_histogram(magnitude_neighbourhood, theta_neighbourhood)
        histogram_sorted = histogram.copy()
        histogram_sorted.sort()

        key.sigma = 4 * SIGMA * (np.sqrt(2)) ** (k + 1)

        if histogram_sorted[-2] > 0.8 * histogram_sorted[-1]:
            key.angle = np.argwhere(histogram == histogram_sorted[-1])[0] * 10
            new_keypoints3.append(key)

            for h in range(2, 36):
                if histogram_sorted[-h] > 0.8 * histogram_sorted[-1]:
                    key.angle = np.argwhere(histogram == histogram_sorted[-h])[0] * 10
                    new_keypoints3.append(key)
                else:
                    break
        else:
            key.angle = np.argwhere(histogram == histogram_sorted[-1])[0] * 10
            new_keypoints3.append(key)

    # Scale 4
    new_keypoints4 = []
    for key in keyset4:
        (x, y) = key.j, key.i
        k = np.uint8(key.DoG - 1)
        minx = np.uint8(min(8, x))
        miny = np.uint8(min(8, y))

        theta_neighbourhood = T4[x - minx: x + minx + 1, y - miny: y + miny + 1, k]
        magnitude_neighbourhood = M4[x - minx: x + minx + 1, y - miny: y + miny + 1, k]
        magnitude_neighbourhood = gaussian_blur(magnitude_neighbourhood, 1.5 * 8 * SIGMA * (np.sqrt(2))**(k+1))

        histogram = compute_histogram(magnitude_neighbourhood, theta_neighbourhood)
        histogram_sorted = histogram.copy()
        histogram_sorted.sort()

        key.sigma = 8 * SIGMA * (np.sqrt(2)) ** (k + 1)

        if histogram_sorted[-2] > 0.8 * histogram_sorted[-1]:
            key.angle = np.argwhere(histogram == histogram_sorted[-1])[0] * 10
            new_keypoints4.append(key)

            for h in range(2, 36):
                if histogram_sorted[-h] > 0.8 * histogram_sorted[-1]:
                    key.angle = np.argwhere(histogram == histogram_sorted[-h])[0] * 10
                    new_keypoints4.append(key)
                else:
                    break
        else:
            key.angle = np.argwhere(histogram == histogram_sorted[-1])[0] * 10
            new_keypoints4.append(key)

    new_keypoints = normalize_keypoints(new_keypoints1, new_keypoints2, new_keypoints3, new_keypoints4)
    # Remove duplicate keys
    keypoint_list = list(set(new_keypoints))

    for k in keypoint_list:
        if k.i < 16 or k.j < 16 or k.i > (scale1.shape[0] - 17) or k.j > (scale1.shape[1] - 17):
            keypoint_list.remove(k)

    descriptor_list = compute_descriptor(scale1, M1, T_1, M2, T_2, M3, T_3, M4, T_4, keypoint_list)

    return keypoint_list, descriptor_list


def convert_KEYPOINT_to_Keypoint(source):
    dest = []

    for k in source:
        key = cv2.KeyPoint(k.x, k.y, k.DoG)
        dest.append(key)

    return dest


def normalize_keypoints(keyset1, keyset2, keyset3, keyset4):
    keyset = []

    for k in keyset1:
        keyset.append(k)

    for k in keyset2:
        k.i = 2 * k.i
        k.j = 2 * k.j
        k.x = 2 * k.x
        k.y = 2 * k.y
        keyset.append(k)

    for k in keyset3:
        k.i = 4 * k.i
        k.j = 4 * k.j
        k.x = 4 * k.x
        k.y = 4 * k.y
        keyset.append(k)

    for k in keyset4:
        k.i = 8 * k.i
        k.j = 8 * k.j
        k.x = 8 * k.x
        k.y = 8 * k.y
        keyset.append(k)

    return keyset


def sift(image):
    # Generate four scale scale-space
    (L1, L2, L3, L4) = generate_scale_space(image)
    # Get the Difference of Gaussian - an approximation to Laplace of Gaussian
    (D1, D2, D3, D4) = generate_DoG(L1, L2, L3, L4)
    # Obtain initial keypoints
    (K1, K2, K3, K4) = determine_initial_keypoints(D1, D2, D3, D4)
    # Obtain final keypoints and descriptors from initial keypoints
    (keypoints, descriptors) = compute_final_keypoints_descriptors(K1, K2, K3, K4, L1, L2, L3, L4)

    # return keypoints, descriptors
    return keypoints, descriptors
    # return K1, K2, K3, K4
