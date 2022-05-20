import argparse
from plot import plot_noise_types, draw_image_plt, draw_image, plot_filtered
import numpy as np
import cv2
from math import log10, sqrt
from add_noise import Gaussian_noise, Salt_Pepper_noise, Speckle_noise
from algorithms import mean_filter, median_filter, non_local_means
from bm3d import bm3d


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'



def PSNR(gt_img, noised_img):
    return 10 * np.log10(255.0 * 255.0 / np.mean(((gt_img - noised_img).ravel()) ** 2))


def add_noise(img, noise_type):
    if noise_type == 'Gaussian noise':
        return Gaussian_noise(img, mean=0., sigma=20.)
    elif noise_type == 'Speckle noise':
        return Speckle_noise(img, mean=0., sigma=2.)
    elif noise_type == 'Salt and pepper noise':
        return Salt_Pepper_noise(img, prob=.15)


def denoise(noised_img, alg_type):
    if alg_type == 'Mean filter':
        return mean_filter(noised_img)
    elif alg_type == 'LMS_adaptive_mean filter':
        return non_local_means(img)
    elif alg_type == 'Median filter':
        return median_filter(noised_img)
    elif alg_type == 'BM3D':
        return bm3d(img)


if __name__=='__main__':
    
    noise_types = {'g': 'Gaussian noise', 's': 'Speckle noise', 's&p': 'Salt and pepper noise'}
    noise_types = {'s&p': 'Salt and pepper noise'}
    denoise_alg_types = {'mn': 'Mean filter', 'mn_lms': 'LMS_adaptive_mean filter', \
                         'med': 'Median filter', 'bm3d': 'BM3D'}
    denoise_alg_types = {'mn': 'Mean filter', 'med': 'Median filter'}
    
    # denoise_alg_types = {'mn': 'Mean filter'}
    noise_parameters = {'Gaussian noise': range(5, 40), 'Speckle noise': np.arange(0.5, 5., 0.2), \
                  'Salt and pepper noise': np.arange(0.1, 0.45, 0.01)}
    filters_parameters = {'Mean filter': [3, 4, 5, 6, 7, 8, 9], 'Median filter': [3, 5, 7, 9]}
    efficiency_coefs = {}
    
    img_name = r'images/Lenna.png'
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

    print(color.BOLD + "\nFinding out the parameters" + color.END)
    for alg_type in denoise_alg_types.values():
        efficiency_coefs[alg_type] = {}
        print(color.DARKCYAN + alg_type + color.END)
        for noise_type in noise_types.values():
            efficiency_coefs[alg_type][noise_type] = {}
            
            print(color.PURPLE + noise_type + color.END)
            for noise_param in noise_parameters[noise_type]:
                if noise_type == 'Gaussian noise':
                    noised_img = Gaussian_noise(img, mean=0., sigma=noise_param)
                elif noise_type == 'Speckle noise':
                    noised_img = Speckle_noise(img, mean=0., sigma=noise_param)
                elif noise_type == 'Salt and pepper noise':
                    noised_img = Salt_Pepper_noise(img, prob=noise_param)

                psnr_noised = PSNR(img, noised_img)

                for alg_param in filters_parameters[alg_type]:
                    if alg_type == 'Mean filter':
                        kernel_size = (alg_param, alg_param) 
                        denoised_img = cv2.blur(noised_img, kernel_size)

                    elif alg_type == 'Median filter':
                        denoised_img = cv2.medianBlur(noised_img, alg_param)
                
                    psnr_denoised = PSNR(img, denoised_img)
                    efficiency_coefs[alg_type][noise_type][psnr_denoised / psnr_noised] = \
                        (noise_param, alg_param)

            best_efficiency = max(list(efficiency_coefs[alg_type][noise_type].keys()))
            print(best_efficiency)
            print(efficiency_coefs[alg_type][noise_type][best_efficiency])
            
            noise_param = efficiency_coefs[alg_type][noise_type][best_efficiency][0]
            alg_param = efficiency_coefs[alg_type][noise_type][best_efficiency][1]
            if noise_type == 'Gaussian noise':
                noised_img = Gaussian_noise(img, mean=0., sigma=noise_param)
            elif noise_type == 'Speckle noise':
                noised_img = Speckle_noise(img, mean=0., sigma=noise_param)
            elif noise_type == 'Salt and pepper noise':
                noised_img = Salt_Pepper_noise(img, prob=noise_param)
            
            if alg_type == 'Mean filter':
                kernel_size = (alg_param, alg_param) 
                denoised_img = cv2.blur(noised_img, kernel_size)

            elif alg_type == 'Median filter':
                denoised_img = cv2.medianBlur(noised_img, alg_param)

            title_noised = noise_type + ', PSNR='"{:.1f}".format(PSNR(img, noised_img)) + 'dB'
            title_denoised = alg_type + ', PSNR='"{:.1f}".format(PSNR(img, denoised_img)) + 'dB'
            plot_filtered(img, noised_img, denoised_img, title_noised, title_denoised)
            



