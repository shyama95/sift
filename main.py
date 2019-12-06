# ---------------------------------------------------------------#
# __name__ = "SIFT keypoint detection implementation in python"
# __author__ = "Shyama P"
# __version__ = "1.0"
# __email__ = "shyamap95@gmail.com"
# __status__ = "Development"
# ---------------------------------------------------------------#
import argparse
# OpenCV2 library is used for reading/ writing of images and color space
# conversion only
import cv2
# Matplotlib library is used for displaying output image
import matplotlib.pyplot as plt
# Has implementation of SIFT keypoint detection
from sift_implementation import *


def main(input_image):    
    # Read input image
    image = cv2.imread(input_image)
    # Get keypoints
    (keypoints, descriptors) = sift(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    # Convert detected keypoints from custom class format to cv2 keypoint datatype
    keypoints = convert_KEYPOINT_to_Keypoint(keypoints)
    # Draw kepoints using on the input image
    output_image = cv2.drawKeypoints(image, keypoints, cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    # Convert output image to RGB space
    output_image = cv2.cvtColor(output_image,cv2.COLOR_BGR2RGB)

    # Display output image
    plt.imshow(output_image)
    plt.show()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SIFT keypoint detection implementation in python')
    parser.add_argument("--input", type=str, required=True, help='Input image path')

    args = parser.parse_args()
    input_image = args.input
    
    main(input_image)