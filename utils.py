import numpy as np
import glob
import cv2
import os

PAD_ZERO: int = 0
PAD_REFLECT: int = 1


def background_mean(back_path):
    """
    Read the back data
    Calculate the average and maximum value
    :return: list of background
    """
    back_fn_list = glob.glob(os.path.join(back_path, "*.tif"))
    back_list = []
    for back_fn in back_fn_list:
        back_list.append(cv2.imread(back_fn, cv2.IMREAD_UNCHANGED))
    back_mean = np.mean(back_list)
    print("Successfully import", len(back_list), "background, mean:", back_mean)
    return back_mean


def convolution_3d(img: np.ndarray, psf: np.ndarray):
    """
    3D convolution of clear images with PSF
    :param img: Clear images
    :param psf: PSF
    :return: A blurred 3D image after convolution
    """
    otf = np.fft.fftn(psf)
    otf[np.where(np.abs(otf) < 1e-4)] = 0
    img_fft = np.fft.fftn(img)
    img_blur_fft = img_fft * otf
    img_blur = np.real(np.fft.fftshift(np.fft.ifftn(img_blur_fft)))
    return img_blur


def generate_3d_gaussian_kernel(size, sigma):
    kernel = np.fromfunction(lambda x, y, z: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
        -((x - size[0] // 2) ** 2 + (y - size[1] // 2) ** 2 + (z - size[2] // 2) ** 2) / (2 * sigma ** 2)),
                             (size[0], size[1], size[2]))
    kernel /= np.sum(kernel)
    return kernel


# 边缘模糊函数(用于减轻维纳滤波的振铃效应)
def blur_edge_2d(img, d=4):
    h, w = img.shape[:2]
    img_pad = cv2.copyMakeBorder(img, d, d, d, d, cv2.BORDER_WRAP)
    img_blur = cv2.GaussianBlur(img_pad, (2 * d + 1, 2 * d + 1), -1)[d:-d, d:-d]  # 虽然高斯滤波后生成的图像是扩展后的图像，但是只取扩展的边缘以内的部分
    y, x = np.indices((h, w))
    dist = np.dstack([x, w - x - 1, y, h - y - 1]).min(-1)
    w = np.minimum(np.float32(dist) / d, 1.0)
    return img * w + img_blur * (1 - w)


def blur_edge_3d(img, d=15):
    a, b, c = img.shape[:3]
    img_pad = np.pad(img, d, "reflect")
    kernel = generate_3d_gaussian_kernel(img_pad.shape, 1)

    img_blur = convolution_3d(img_pad, kernel)[d:-d, d:-d, d:-d]
    z, y, x = np.indices((a, b, c))
    dist = np.stack([x, c - x - 1, y, b - y - 1, z, a - z - 1], axis=3).min(-1)
    w = np.minimum(np.float32(dist) / d, 1.0)

    return img * w + img_blur * (1 - w)


def resize_psf_2d(img: np.ndarray, psf: np.ndarray):
    """
    Resize a 2D PSF image to the target image size
    :param img: Reference image array
    :param psf: Point Spread Function array to resize
    :return: the psf array padded to get the same shape as image
    """
    kernel = np.zeros(img.shape)
    x_start = int(img.shape[0] / 2 - psf.shape[0] / 2)
    y_start = int(img.shape[1] / 2 - psf.shape[1] / 2)
    kernel[x_start:x_start + psf.shape[0], y_start:y_start + psf.shape[1]] = psf
    return kernel


def resize_psf_3d(img: np.ndarray, psf: np.ndarray):
    """
    Resize a 3D PSF image to the target image size
    :param img: Reference image array
    :param psf: Point Spread Function array to resize
    :return: the psf array padded to get the same shape as image
    """
    kernel = np.zeros(img.shape)
    x_start = int(img.shape[0] / 2 - psf.shape[0] / 2)
    y_start = int(img.shape[1] / 2 - psf.shape[1] / 2)
    z_start = int(img.shape[2] / 2 - psf.shape[2] / 2)
    kernel[x_start:x_start + psf.shape[0], y_start:y_start + psf.shape[1], z_start:z_start + psf.shape[2]] = psf
    return kernel


def pad_3d(img: np.ndarray, psf: np.ndarray, pad, flag=PAD_REFLECT):
    """
    Pad a 3D image and it PSF for deconvolution
    :param img: Image array
    :param psf: Point Spread Function array
    :param pad: The number of edge-filled lines
    :param flag: Padding mode
    :return: image, psf, padding: padded versions of the image and the PSF, plus the padding tuple
    """
    padding = pad
    if isinstance(pad, tuple) and len(pad) != img.ndim:
        raise Exception("Padding must be the same dimension as image")
    if isinstance(pad, int):
        if pad == 0:
            return img, psf, (0, 0, 0)
        padding = (pad, pad, pad)

    if padding[0] > 0 and padding[1] > 0 and padding[2] > 0:
        p3d = np.array([padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]])
        if flag == PAD_REFLECT:
            img_pad = np.pad(img, p3d.reshape((3, 2)), "reflect")
        else:
            img_pad = np.pad(img, p3d.reshape((3, 2)), "constant")
        psf_pad = np.pad(psf, p3d.reshape((3, 2)), "constant")
    else:
        img_pad = img
        psf_pad = psf
    return img_pad, psf_pad, padding


def unpad_3d(img: np.ndarray, padding: tuple) -> np.ndarray:
    """
    Remove the padding of an image
    :param img: 3D image to unpad
    :param padding: Padding in each dimension
    :return: The unpadded image
    """
    return img[padding[0]:-padding[0],
           padding[1]:-padding[1],
           padding[2]:-padding[2]]

