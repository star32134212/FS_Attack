import numpy as np
import math
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.measure import compare_ssim
from scipy.stats import pearsonr
from skimage import color

img1 = im_orig
img2 = im_adv
img1_gray = color.rgb2gray(img1)
img2_gray = color.rgb2gray(img2)

PSNR = peak_signal_noise_ratio(img1, img2)
MSE = mean_squared_error(img1, img2)
SSIM = structural_similarity(img1, img2, multichannel=True)
PCCS = pearsonr(img1_gray.flatten(),img2_gray.flatten())

print('MSE: ', MSE)
print('PSNR: ', PSNR)
print('SSIM: ', SSIM)
print('PCCS: ', PCCS[0])

def psnr(img1,img2):
    mse = np.mean((img1/255. - img2/255.)**2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def gini(array):
    array = np.array(array, dtype=np.float64)
    array = np.abs(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1, array.shape[0] + 1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))