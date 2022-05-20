import numpy as np
import cv2

# from scipy.ndimage import gaussian_filter
# import os
# import cv2


def Gaussian_noise(img, mean=0., sigma=1.):
    noise = np.random.normal(mean, sigma, img.size)
    noise = noise.reshape(img.shape).astype('uint8')
    return img + noise


def Speckle_noise(img, mean=0., sigma=1.):
    noise = np.random.normal(mean, sigma, img.size)
    noise = noise.reshape(img.shape).astype('uint8')
    return img + img * noise


def Salt_Pepper_noise(img, prob=0.1):
    # img = cv2.imread(img_name, 0) # to grayscale
    row, col = img.shape
    noised_img = np.copy(img)

    number_of_pixels = prob * row * col
    sp_ratio = np.random.rand()
    for _ in range(int(number_of_pixels * sp_ratio)):
        y_coord=np.random.randint(0, row - 1)
        x_coord=np.random.randint(0, col - 1)
        noised_img[y_coord][x_coord] = 255
            
    for _ in range(int(number_of_pixels * (1 - sp_ratio))):
        y_coord=np.random.randint(0, row - 1)
        x_coord=np.random.randint(0, col - 1)
        noised_img[y_coord][x_coord] = 0
    
    return noised_img


    





