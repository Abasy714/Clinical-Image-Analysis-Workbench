"""
2D convolution implemented entirely from scratch 
This is the core engine used by all spatial filters.
"""

import numpy as np
from utils.error_handler import wrap_errors

PADDING_MODES = ['zero', 'reflect', 'replicate']


def _pad_image(image: np.ndarray, pad_h: int, pad_w: int, mode: str) -> np.ndarray:
    #we choose the padding that we want we have 3 padding modes:
     # zero padding: pads with zeros (black pixels) -- bey fill el border with zeros (black pixels)
      # best used when you want to avoid introducing any new pixel values at the borders, but can create a black border effect
     # reflect padding: pads by reflecting the image across the border -- bey fill el border by reflecting the image across the border (like a mirror)
      # best used when you want to create a seamless border that maintains the local image structure, but can introduce artifacts if the image has strong edges at the borders
     # replicate padding: pads by replicating the edge pixels -- bey fill el border by replicating the edge pixels (like extending the edge pixels outward)
        # best used when you want to maintain the intensity of the edge pixels at the borders, but can create a "stretched" effect if the image has strong edges at the borders
    """
    Pad a 2D image array on all sides.

    Parameters
    ----------
    image   : 2D float64 array (H, W)
    pad_h   : number of rows to add on each of top and bottom
    pad_w   : number of cols to add on each of left and right
    mode    : 'zero' | 'reflect' | 'replicate'

    Returns
    -------
    padded  : 2D float64 array (H + 2*pad_h, W + 2*pad_w)
    """
    if mode == 'zero':
        return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    if mode == 'reflect':
        return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

    if mode == 'replicate':
        return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

    raise ValueError(f"Unknown padding mode '{mode}'. Choose from {PADDING_MODES}.")


@wrap_errors
def convolve2d(image: np.ndarray, kernel: np.ndarray, padding: str = 'zero') -> np.ndarray:
    """
    Apply a 2D convolution kernel to a grayscale image from scratch.

    The kernel is flipped (true convolution, not correlation) before sliding.
    Works for any odd or even kernel size.

    Parameters
    ----------
    image   : 2D numpy array (H, W), any numeric dtype
    kernel  : 2D numpy array (kH, kW), the convolution kernel
    padding : padding strategy — 'zero' (default), 'replicate', or 'reflect'

    Returns
    -------
    output  : 2D float64 array (H, W), same spatial size as input
    """
    #ndim stands for number of dimensions in the array
    if image.ndim != 2: #must be grayscale images only , 3ashan convolution is defined for 2d images bas
        raise ValueError(f"convolve2d expects a 2D image, got shape {image.shape}")  
    if kernel.ndim != 2: #must be 2d kernel only , 3ashan convolution is defined for 2d kernels bas
        raise ValueError(f"convolve2d expects a 2D kernel, got shape {kernel.shape}")

    image = image.astype(np.float64) #hanestakhdem float64 for safety from overflow during convolution, we convert the image to float64 before processing
    kernel = kernel.astype(np.float64)

    # true convolution = correlation with flipped kernel
    kernel_flipped = np.flip(kernel) #ehna hena ben3kes el kernel ashan ne3mel convolution mesh correlation, conolution is a correlation bas flipped 180 degrees

    img_h, img_w = image.shape
    k_h, k_w = kernel_flipped.shape #we get the height and width of the kernel after flipping

    #ben calculate han pad b ad eh fo2 w taht w yemeen w shemal (row and columns)
    pad_h = k_h // 2 #we use this equation ashan ne3mel padding ashan el kernel yb2a sabet ala el image lel center pixels
    pad_w = k_w // 2 

    padded = _pad_image(image, pad_h, pad_w, padding) #apply padding

    # Vectorized sliding window — replaces O(H*W) Python loop
    from numpy.lib.stride_tricks import as_strided # badal ma ne3mel loop ala kol pixel lewahdaha, ben3mel sliding window ala el image kolaha mara wahda using numpy's as_strided
    shape = (img_h, img_w, k_h, k_w)
    strides = ( 
        padded.strides[0],   # dim 0: move to next row of CENTER pixels
        padded.strides[1],   # dim 1: move to next col of CENTER pixels
        padded.strides[0],   # dim 2: move down inside the patch (kernel window finding neighboring pixels)
        padded.strides[1],   # dim 3: move right inside the patch (kernel window finding neighboring pixels)
)
    patches = as_strided(padded, shape=shape, strides=strides)
    output = np.einsum('ijkl,kl->ij', patches, kernel_flipped) #ehna ben3mel dot product ben el patches (sliding windows) we el kernel el ma3kes, we ben7seb el convolution result for each pixel fe el output array
    return output





















#for i in range(img_h):
    #for j in range(img_w):
       # output[i,j] = np.sum(
           # padded[i:i+k_h, j:j+k_w] * kernel_flipped
       # )