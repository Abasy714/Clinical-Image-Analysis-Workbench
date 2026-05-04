"""
Histogram computation from scratch without using numpy.histogram or cv2.calcHist.
Supports 8-bit and 16-bit grayscale images.
"""
import numpy as np
from utils.image_utils import validate_grayscale
from utils.error_handler import wrap_errors


@wrap_errors
#takes array  and no. of intensity levels 
def compute_histogram(image: np.ndarray, bins: int = 256) -> np.ndarray:
    #validates that image is 2d
    validate_grayscale(image)
    #creates array of zeros ,acts as counter for intensity vals 
    hist = np.zeros(bins, dtype=np.int64)

    #converts into 1d array 
    flat = image.flatten()

    #loop over pixels in image
    for pixel_value in flat:
        idx = int(pixel_value)
        if 0 <= idx < bins:
            hist[idx] += 1 #increments count for this intensity level 

    return hist #array that holds the count for each intensity value


@wrap_errors
def compute_cdf(histogram: np.ndarray) -> np.ndarray:
    num_levels = len(histogram) #gets number of bins(256)
    cdf = np.zeros(num_levels, dtype=np.float64) #empty array to hold cumulative vals

   
    cdf[0] = histogram[0] #first cdf value is the first histo bin count 
    for i in range(1, num_levels): #cumulative sum
        cdf[i] = cdf[i - 1] + histogram[i]

    total_pixels = cdf[-1]
    if total_pixels > 0: #normalizes cdf
        cdf = cdf / total_pixels

    return cdf