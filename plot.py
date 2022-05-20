import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg
from add_noise import Gaussian_noise, Salt_Pepper_noise, Speckle_noise

# def draw_colorchecker(stimuli: np.ndarray, shape: Tuple[int, int], show=False):
#     assert len(shape) == 2
#     rows = shape[0]
#     cols = shape[1]

#     if len(stimuli.shape) == 2:
#         carray = np.asarray(stimuli)
#         carray = carray.reshape((rows, cols, 3))
#     elif len(stimuli.shape) == 3:
#         pixels_num = stimuli.shape[0]
#         size = int(sqrt(pixels_num))
#         if not (pixels_num % size == 0):
#             raise ValueError(f'There is no integer size like size**2 == stimuli.shape[0]!')
#         tmp = np.asarray(stimuli)
#         tmp = tmp.reshape((tmp.shape[0], rows, cols, 3))
#         carray = np.zeros((rows*size, cols*size, 3))
#         for i in range(rows):
#             for j in range(cols):
#                 for si in range(size):
#                     for sj in range(size):
#                         carray[i*size + si, j*size + sj] = tmp[si*size + sj, i, j, :]
    
#     carray = carray / carray.max()
#     plt.imshow(carray)
#     if show: plt.show()
#     return carray


# def error_heatmap(nslices: int, tips: list) -> None:
#     """
#     This function builds heatmap to visualize accuracy usings learning and test samples
#     Args:
#         nslices (int): 
#             number of bars in colorchecker
#         tips (list): 
#             list of sensitivities
#     """    
#     a = np.array(tips)
#     tips1 = a.reshape((nslices, -1))
#     value_max = max(tips1, key=lambda item: item[1])[1]
#     value_min = min(tips1, key=lambda item: item[1])[1]
#     sns.set_theme()
#     sns.heatmap(tips1, annot = True, vmin=value_min, vmax=value_max, center= (value_min+value_max)//2, fmt='.3g', cmap= 'coolwarm')



# def draw_multiple_plots(subtitle, cc, cc_shape):
#     fig, axes = plt.subplots(1, 3, sharex=True, 
#                               sharey=True, figsize=(8,5))
#     fig.suptitle(subtitle)
#     cc_size = cc_shape[0] * cc_shape[1]

#     for i, ax in enumerate(axes.flat):
#         cc_part = cc[i * cc_size : (i + 1) * cc_size, :].reshape((cc_shape[0], cc_shape[1], 3))
#         cc_part /= cc_part.max()
#         ax.imshow(cc_part)
#         ax.set_title('1')

#     fig.tight_layout()
#     plt.show()
#     plt.close()


def draw_image_plt(title, image):
    # image_shape = image.shape
    # image = image.reshape((image_shape[0], image_shape[1], 3))
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()
    plt.close()

def draw_image(title, img):
    width = 500
    height = int(img.shape[1] * (width / img.shape[0]))
    img = cv2.resize(img, (width, height)) 
    cv2.imshow(title, img)
    cv2.waitKey(0)


def plot_noise_4_types():
    fig = plt.figure(figsize=(11, 9)) 
    sub1 = plt.subplot2grid((2,6),(0,0), colspan=3)
    sub2 = plt.subplot2grid((2,6),(0,3), colspan=3) 
    sub3 = plt.subplot2grid((2,6),(1,0), colspan=2)
    sub4 = plt.subplot2grid((2,6),(1,2), colspan=2) 
    sub5 = plt.subplot2grid((2,6),(1,4), colspan=2)
    subs = [sub1, sub2, sub3, sub4, sub5]

    for i, sub in enumerate(subs):
        d = d_list[i]
        sub.set_title('d = ' + str(d))
        sub.set_ylabel("f", fontsize=10)
        sub.set_xlabel("N_fev", fontsize=10)
        sub.grid()

        for method in N_all[d].keys():
            sub.scatter(N_all[d][method], f_from_N_all[d][method], label=method)
        sub.legend(fontsize=12)
    
    fig.tight_layout()
    plt.show()


def plot_noise_types(img_name):
    fig = plt.figure(figsize=(6, 6)) 
    sub1 = plt.subplot2grid((2,2),(0,0))
    sub2 = plt.subplot2grid((2,2),(0,1)) 
    sub3 = plt.subplot2grid((2,2),(1,0))
    sub4 = plt.subplot2grid((2,2),(1,1))
    subs = [sub1, sub2, sub3, sub4]
    for sub in subs:
        sub.axis('off') 

    # img = mpimg.imread(img_name)
    img = cv2.imread(img_name, 0)

    sub1.imshow(img, cmap='gray')
    sub1.set_title('Ground truth')
    noised_img = Gaussian_noise(img, mean=0., sigma=25.)
    sub2.imshow(noised_img, cmap='gray')
    sub2.set_title('Gaussian noise')
    noised_img = Speckle_noise(img, mean=0., sigma=1.)
    sub3.imshow(noised_img, cmap='gray')
    sub3.set_title('Speckle noise')
    noised_img = Salt_Pepper_noise(img_name, prob=0.1)
    sub4.imshow(noised_img, cmap='gray')
    sub4.set_title('Salt and pepper noise')
    
    plt.subplots_adjust(left=0.01,
                    bottom=0.01, 
                    right=0.99, 
                    top=0.96, 
                    wspace=0.01, 
                    hspace=0.1)
    # fig.tight_layout()
    plt.show()


def plot_filtered(img, noised_img, denoised_img, noise_type, filter_type, is_gray=True):
    plt.figure(figsize=(9,3))
    
    plt.subplot(131) 
    if is_gray:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.title('Ground truth, PSNR = '+ r'$\infty$' +' dB')
    plt.xticks([]), plt.yticks([])

    plt.subplot(132) 
    if is_gray:
        plt.imshow(noised_img, cmap='gray')
    else:
        plt.imshow(noised_img)
    plt.title(noise_type)
    plt.xticks([]), plt.yticks([])

    plt.subplot(133) 
    if is_gray:
        plt.imshow(denoised_img, cmap='gray')
    else:
        plt.imshow(denoised_img)
    plt.title(filter_type)
    plt.xticks([]), plt.yticks([])

    plt.subplots_adjust(left=0.01,
                bottom=0.01, 
                right=0.99, 
                top=0.9, 
                wspace=0.01, 
                hspace=0.1)

    plt.show()


def plot_noise_types_new(img_name):
    fig = plt.figure(figsize=(10., 2.5))
    
    sub1 = plt.subplot2grid((1,4),(0,0))
    sub2 = plt.subplot2grid((1,4),(0,1)) 
    sub3 = plt.subplot2grid((1,4),(0,2))
    sub4 = plt.subplot2grid((1,4),(0,3))
    subs = [sub1, sub2, sub3, sub4]
    for sub in subs:
        sub.set_xticks([]), sub.set_yticks([])

    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

    sub1.imshow(img, cmap='gray')
    sub1.set_title('Ground truth')
    noised_img = Gaussian_noise(img, mean=0., sigma=45.)
    sub2.imshow(noised_img, cmap='gray')
    sub2.set_title('Gaussian noise')
    noised_img = Speckle_noise(img, mean=0., sigma=1.3)
    sub3.imshow(noised_img, cmap='gray')
    sub3.set_title('Speckle noise')
    noised_img = Salt_Pepper_noise(img, prob=0.4)
    sub4.imshow(noised_img, cmap='gray')
    sub4.set_title('Salt and pepper noise')
    
    # plt.subplots_adjust(left=0.01,
    #                 bottom=0.01, 
    #                 right=0.99, 
    #                 top=0.96, 
    #                 wspace=0.01, 
    #                 hspace=0.3)
    fig.tight_layout()
    plt.show()


def plot_results(img_name, is_gray=True):
    fig = plt.figure(figsize=(10., 2.5)) 

    noise_types = ['Ground truth', 'Gaussian noise', 'Speckle noise', 'Salt and pepper noise']
    alg_types = ['Ground truth', 'Mean filter', 'Median filter', 'Non-Local Means', 'BM3D']
    
    noise_types = ['Ground truth', 'Gaussian noise', 'Speckle noise', 'Salt and pepper noise']
    alg_types = ['Ground truth', 'Mean filter', 'Non-Local Means', 'Median filter']
    
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    fig.suptitle('Ground truth       Gaussian noise       Speckle noise        Salt and pepper noise')

    for alg_num in range(len(alg_types)):
        alg_type = alg_types[alg_num]

        for noise_num in range(len(noise_types)):

            noise_type = noise_types[noise_num]
            noised_img = add_noise(img, noise_type)
            denoised_img = denoise(noised_img, alg_type)
            psnr_denoised = PSNR(img, denoised_img)

            sub = plt.subplot2grid((len(alg_types), len(noise_types)),(alg_num,noise_num))

            if is_gray: sub.imshow(denoised_img, cmap='gray')
            else: sub.imshow(denoised_img)

            if noise_type == 'Ground truth':
                sub.set_title('PSNR = '+ r'$\infty$' +' dB')
                sub.set_title(alg_type, rotation='vertical')
            else:
                sub.set_title('PSNR='"{:.1f}".format(psnr_denoised) + 'dB')
            sub.set_xticks([]), sub.set_yticks([])
            
    fig.tight_layout()
    
    # plt.subplots_adjust(left=0.02,
    #                 bottom=0.01, 
    #                 right=0.99, 
    #                 top=1.96, 
    #                 wspace=0.02, 
    #                 hspace=0.4)
    
    plt.show()



