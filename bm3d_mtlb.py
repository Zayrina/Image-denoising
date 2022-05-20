import numpy as np
import matlab.engine
import matplotlib.pyplot as plt

# from functools import partial
# from OpenDenoising import data
# from OpenDenoising import model

eng = matlab.engine.start_matlab()


def display_results(clean_imgs, noisy_imgs, rest_imgs, name):
    """Display denoising results."""
    fig, axes = plt.subplots(5, 3, figsize=(15, 15))

    plt.suptitle("Denoising results using {}".format(name))

    for i in range(5):
        axes[i, 0].imshow(np.squeeze(clean_imgs[i]), cmap="gray")
        axes[i, 0].axis("off")
        axes[i, 0].set_title("Ground-Truth")

        axes[i, 1].imshow(np.squeeze(noisy_imgs[i]), cmap="gray")
        axes[i, 1].axis("off")
        axes[i, 1].set_title("Noised Image")

        axes[i, 2].imshow(np.squeeze(rest_imgs[i]), cmap="gray")
        axes[i, 2].axis("off")
        axes[i, 2].set_title("Restored Images")


# data.download_BSDS_grayscale(output_dir="./tmp/BSDS500/")


# Validation images generator
# valid_generator = data.DatasetFactory.create(path="./tmp/BSDS500/Valid",
#                                              batch_size=8,
#                                              n_channels=1,
#                                              noise_config={data.utils.gaussian_noise: [25]},
#                                              name="BSDS_Valid")



def BM3D_matlab(z, sigma=25.0, profile="np", channels_first=False):
    """This function wraps MATLAB's BM3D implementation, available on matlab_libs/BM3Dlib. The original code is
    available to the public through the author's page, on http://www.cs.tut.fi/~foi/GCF-BM3D/

    Parameters
    ----------
    z : :class:`numpy.ndarray`
        4D batch of noised images. It has shape: (batch_size, height, width, channels).
    sigma : float
        Level of gaussian noise.
    profile : str
        One between {'np', 'lc', 'high', 'vn', 'vn_old'}. Algorithm's profile.

        Available for grayscale:

        * 'np': Normal profile.
        * 'lc': Fast profile.
        * 'high': High quality profile.
        * 'vn': High noise profile (sigma > 40.0)
        * 'vn_old': old 'vn' profile. Yields inferior results than 'vn'.

        Available for RGB:

        * 'np': Normal profile.
        * 'lc': Fast profile.

    Returns
    -------
    y_est : :class:`numpy.ndarray`
        4D batch of denoised images. It has shape: (batch_size, height, width, channels).
    """
    _z = z.copy()
    rgb = True if _z.shape[-1] == 3 else False
    if rgb:
        assert (profile in ["np", "lc", "high", "vn", "vn_old"]), "Expected profile to be 'np', 'lc', 'high', 'vn' " \
                                                                  "or 'vn_old' but got {}.".format(profile)
    else:
        assert (profile in ["np", "lc"]), "Expected profile to be 'np', 'lc' bug got {}".format(profile)


    # Convert input arrays to matlab
    m_sigma = matlab.double([sigma])
    m_show = matlab.int64([0])

    # Call BM3D function on matlab
    y_est = []
    for i in range(len(_z)):
        m_z = matlab.double(_z[i, :, :, :].tolist())
        if rgb:
            _, y_est = eng.CBM3D(m_z, m_z, m_sigma, profile, m_show, nargout=2)
        else:
            _, y = eng.BM3D(m_z, m_z, m_sigma, profile, m_show, nargout=2)
        y_est.append(np.asarray(y))

    y_est = np.asarray(y_est).reshape(z.shape)
    return y_est