import cv2
import numpy as np
from utils import resize_psf_3d, background_mean
import deconvolution_methods_3d as dm
from utils import resize_psf_2d

if __name__ == '__main__':

    # region 载入三维图像
    psf_path = r"231220/psf_centroid.tif"
    source_path = r"231220/5179_ori-2.tif"
    back_path = r"231220/background"
    back_mean = background_mean(back_path)
    print(back_mean)
    _, psf = cv2.imreadmulti(psf_path, flags=cv2.IMREAD_UNCHANGED)
    psf = np.float64(psf)
    psf /= psf.sum()
    _, img = cv2.imreadmulti(source_path, flags=cv2.IMREAD_UNCHANGED)
    img = np.float64(img)
    print(np.mean(img))
    # img = np.float64(img) - np.mean(img) * 0.8
    # img = np.float64(img) - back_mean
    # img[np.where(img < 0)] = 0
    img /= np.max(img)

    focus = 19
    focus = int(img.shape[0] / 2 + psf.shape[0] / 2-focus-1)

    if psf.shape != img.shape:
        psf = resize_psf_3d(img, psf)
    # endregion

    psf_flip = np.flip(psf, axis=[0])

    # region deconvolution
    path = "231220/5179-2_result/centroid/5179-2"



    dm.wiener_3d(img, psf_flip, snr=50, pad=7, path=path)
    # dm.rl_3d(img, psf_flip, pad=0, niter=200, eps=1e-5, path=path)
    # dm.nearest_neighbor_3d(img, psf_flip, focus=focus, c=0.9, snr=25, path=path)
    # endregion


