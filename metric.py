from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.measure import compare_ssim
from scipy.stats import pearsonr
from skimage import color
import numpy as np
import argparse


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--img', type=str, default='output/lrp_1_ori_img.npy')
    argparser.add_argument('--adv_img', type=str, default='output/lrp_1_adv_img.npy')
    args = argparser.parse_args()
    
    im_orig = np.load(args.img)
    im_adv = np.load(args.adv_img)
    print(im_orig.max(),im_orig.min(),im_orig.shape,im_orig.shape)
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

if __name__ == "__main__":
    main()