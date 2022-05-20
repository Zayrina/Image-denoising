import numpy as np
import cv2
import padasip as pa
from bm3d import bm3d
from skimage.restoration import denoise_nl_means, estimate_sigma


def mean_filter(img, noise_type='Gaussian noise'):
    kernel_size = (3,3)
    if noise_type == 'Gaussian noise': kernel_size = (5,5)
    elif noise_type == 'Speckle noise': kernel_size = (3,3)
    elif noise_type == 'Salt and pepper noise': kernel_size = (9,9)
    return cv2.blur(img, kernel_size)


def non_local_means(img):
    patch_kw = dict(patch_size=5,  # 5x5 patches
                patch_distance=6)  # 13x13 search area
    sigma_est = np.mean(estimate_sigma(img))
    denoise_fast = denoise_nl_means(img, h=0.6 * sigma_est, sigma=sigma_est, fast_mode=True,
                                **patch_kw)
    return denoise_fast
    # return cv2.fastNlMeansDenoising(img)


def median_filter(img, noise_type):
    kernel_size = 3
    if noise_type == 'Gaussian noise': kernel_size = 5
    elif noise_type == 'Speckle noise': kernel_size = 3
    elif noise_type == 'Salt and pepper noise': kernel_size = 3
    return cv2.medianBlur(img, kernel_size)


def bm3d_local(img, sigma=None):
    if sigma == None:
        sigma_est = np.mean(estimate_sigma(img))
    else:
        sigma_est = sigma
   
    print("sigma_est: ", sigma_est)
    # psd = 30 / 255
    return bm3d(img, sigma_est)



