# ---------------------------------------------------------------#
# __name__ = "SIFT keypoint detection implementation in python"
# __author__ = "Shyama P"
# __version__ = "1.0"
# __email__ = "shyamap95@gmail.com"
# __status__ = "Development"
# ---------------------------------------------------------------#


# Constants definition
SIGMA = 0.707107  # Initial standard deviation for octave 1
SAMPLE_RATE = 2  # Downsampling rate between octaves
OCTAVE_DIMENSION = 5  # Number of images per octave
RADIUS_OF_CURVATURE = 10  # Radius of curvature for edge point rejection
THRESHOLD = 15
