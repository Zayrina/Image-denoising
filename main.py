import argparse
from plot import plot_noise_types_new, draw_image_plt, draw_image, plot_filtered
import numpy as np
import cv2
from add_noise import Gaussian_noise, Salt_Pepper_noise, Speckle_noise
from algorithms import mean_filter, median_filter, non_local_means, bm3d_local
from bm3d import bm3d
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr


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
        return Gaussian_noise(img, mean=0., sigma=25.)
    elif noise_type == 'Speckle noise':
        return Speckle_noise(img, mean=0., sigma=.8)
    elif noise_type == 'Salt and pepper noise':
        return Salt_Pepper_noise(img, prob=0.15)
    elif noise_type == 'Ground truth':
        return img


def denoise(noised_img, alg_type, noise_type):
    if alg_type == 'Mean filter':
        return mean_filter(noised_img, noise_type)
    elif alg_type == 'Non-Local Means':
        return non_local_means(noised_img)
    elif alg_type == 'Median filter':
        return median_filter(noised_img, noise_type)
    elif alg_type == 'BM3D':
        # if noise_type == 'Speckle noise':
        #     return bm3d_local(noised_img, .8)
        # else:
        #     return bm3d_local(noised_img)
        return bm3d_local(noised_img)
    elif alg_type == 'Ground truth':
        return noised_img



def plot_results(img, alg_type, is_gray=True):

    if alg_type == 'Ground truth': figsize=(9., 2.3)
    else: figsize=(9., 2.08)
    fig = plt.figure(figsize=figsize) 
    
    for noise_num in range(len(noise_types)):

        noise_type = noise_types[noise_num]
        noised_img = add_noise(img, noise_type)
        denoised_img = denoise(noised_img, alg_type, noise_type)
        psnr_denoised = psnr(img, denoised_img)

        sub = plt.subplot2grid((1, len(noise_types)),(0, noise_num))

        if is_gray: sub.imshow(denoised_img, cmap='gray')
        else: sub.imshow(denoised_img)

        if alg_type == 'Ground truth':
            sub.set_title(noise_type, fontsize=13) 
            plt.subplots_adjust(left=0.025,
                    bottom=0.00, 
                    right=0.99, 
                    top=0.81, 
                    wspace=0.02, 
                    hspace=0.01)
        else:
            plt.subplots_adjust(left=0.025,
                    bottom=0.00, 
                    right=0.99, 
                    top=0.9, 
                    wspace=0.02, 
                    hspace=0.01)
        
        if noise_type == 'Ground truth':
            sub.set_xlabel('PSNR = '+ r'$\infty$' +' dB', fontsize=12)
            sub.xaxis.set_label_position('top')
            sub.set_ylabel(alg_type, fontsize=13)
    
        else:
            sub.set_xlabel('PSNR='"{:.1f}".format(psnr_denoised) + 'dB', fontsize=12)
            sub.xaxis.set_label_position('top')
        
        sub.set_xticks([]), sub.set_yticks([])
    
    plt.show()

if __name__=='__main__':
    
    noise_types = ['Ground truth', 'Gaussian noise', 'Speckle noise', 'Salt and pepper noise']
    alg_types = ['Ground truth', 'Mean filter', 'Median filter', 'Non-Local Means', 'BM3D']


    img_name = r'images/Lenna.png'
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

    
    for alg_num in range(len(alg_types)):
        alg_type = alg_types[alg_num]
        plot_results(img, alg_type, is_gray=True)

    # noised_img = Salt_Pepper_noise(img, prob=0.1)
    # noised_img = Speckle_noise(img, mean=0., sigma=.8)
    # denoised_img = bm3d_local(noised_img)
    # # denoised_img = BM3D_matlab(noised_img, 13.0)
    # title_noised = 'noise' + ', PSNR='"{:.1f}".format(PSNR(img, noised_img)) + 'dB'
    # title_denoised = 'BM3D' + ', PSNR='"{:.1f}".format(PSNR(img, denoised_img)) + 'dB'
    # plot_filtered(img, noised_img, denoised_img, title_noised, title_denoised)

    # denoised_img = bm3d_local(noised_img, .8)
    # plot_filtered(img, noised_img, denoised_img, title_noised, title_denoised)


    



