import cv2
import time
import utils
from utils import convolution_3d, blur_edge_3d, blur_edge_2d
import numpy as np

BLUR_ONLY: int = 0
BLUR_GAUSS: int = 1
BLUR_POISSON: int = 2
BLUR_GP: int = 3

PAD_ZERO = utils.PAD_ZERO
PAD_REFLECT = utils.PAD_REFLECT

INV_DIRECT: int = 0
INV_CUT: int = 1


def blur_3d(img: np.ndarray, psf: np.ndarray, noise_flag: int, mean=0, sigma=0.001, pad=0, pad_flag=PAD_ZERO):
    """
    Image blur with edge extension (PSF + noise)
    :param img: Clear images
    :param psf: PSF
    :param noise_flag: Types of noise
        BLUR_ONLY - Noiseless
        BLUR_GAUSS - Gaussian noise
        BLUR_POISSON - Poisson noise
    :param mean: Gaussian noise mean
    :param sigma: Standard deviation of Gaussian noise
    :param pad: The number of edge-filled lines
    :param pad_flag: Padding mode
        PAD_ZERO - Zero padding
        PAD_REFLECT - Mirror padding
    :return: Blurred image
    """
    img_pad, psf_pad, padding = utils.pad_3d(img, psf, pad, pad_flag)
    img_blur = convolution_3d(img_pad, psf_pad)

    if noise_flag == BLUR_GAUSS:
        noise = np.random.normal(mean, sigma, img_pad.shape)
        img_blur = img_blur + noise
        img_blur = np.clip(img_blur, a_min=0, a_max=1)

    elif noise_flag == BLUR_POISSON:
        vals = len(np.unique(img_blur))
        vals = 2 ** np.ceil(np.log2(vals))
        img_blur = np.random.poisson(img_blur * vals) / np.float32(vals)

    elif noise_flag == BLUR_GP:
        vals = len(np.unique(img_blur))
        vals = 2 ** np.ceil(np.log2(vals))
        img_blur = np.random.poisson(img_blur * vals) / np.float32(vals)
        noise = np.random.normal(mean, sigma, img_pad.shape)
        img_blur = img_blur + noise
        img_blur = np.clip(img_blur, a_min=0, a_max=1)

    if img_pad.shape != img.shape:
        return utils.unpad_3d(img_blur, padding)
    return img_blur


def inverse_3d(img: np.ndarray, psf: np.ndarray, flag: int, eps=10e-2):
    """
    Deconvolution with inverse filter
    :param img: Blurred image
    :param psf: PSF
    :param flag: Types of deconvolution
        INV_DIRECT - naive inverse filter
        INV_CUT - inverse-cutoff filter
    :param eps: Cut-off threshold
    :return: filtered image
    """
    otf = np.fft.fftn(psf)
    img_fft = np.fft.fftn(img)
    inv_filter = 1 / otf
    if flag == INV_CUT:
        inv_filter[np.where(otf < eps)] = 0
    result_fft = img_fft * inv_filter
    result = np.real(np.fft.fftshift(np.fft.ifftn(result_fft)))
    result[np.where(result < 0)] = 0
    return result


# def wiener1_3d(img: np.ndarray, psf: np.ndarray, snr=100, pad=0):
#     """
#     Wiener deconvolution with edge blur
#     :param img: Blurred image
#     :param psf: PSF
#     :param snr: Signal-to-noise ratio parameters, set artificially
#     :param pad: The number of edge-filled lines
#     :return: Filtered image
#     """
#     img_pad = blur_edge_3d(img, d=pad)
#
#     otf = np.fft.fftn(psf)
#     img_fft = np.fft.fftn(img_pad)
#     otf2 = np.abs(otf) ** 2
#     wiener_filter = otf.conjugate() / (otf2 + 1 / snr ** 2)
#     result_fft = img_fft * wiener_filter
#     result = np.real(np.fft.fftshift(np.fft.ifftn(result_fft)))
#     result[np.where(result < 0)] = 0
#
#     return result

def wiener_3d(img: np.ndarray, psf: np.ndarray, path=None, snr=100, pad=0):
    """
    Wiener deconvolution with edge extension
    :param img: Blurred image
    :param psf: PSF
    :param path: save path
    :param snr: Signal-to-noise ratio parameters, set artificially
    :param pad: The number of edge-filled lines
    :return: Filtered image
    """
    print("start Wiener deconvolution...")
    start_time = time.time()
    img_pad, psf_pad, padding = utils.pad_3d(img, psf, pad)

    otf = np.fft.fftn(psf_pad)
    img_fft = np.fft.fftn(img_pad)
    otf2 = np.abs(otf) ** 2
    wiener_filter = otf.conjugate() / (otf2 + 1 / snr ** 2)
    result_fft = img_fft * wiener_filter
    result = np.real(np.fft.fftshift(np.fft.ifftn(result_fft)))
    result[np.where(result < 0)] = 0

    if img_pad.shape != img.shape:
        result = utils.unpad_3d(result, padding)

    end_time = time.time()
    execution_time = end_time - start_time
    print("Wiener deconvolution took", execution_time, "secs!")

    if path is not None:
        save_tiff_3d(path + "_wiener.tif", result)
    return result


def rl_3d(img, psf, path, pad=0, niter=50, eps=1e-2):
    """
    Three-dimensional Richardson-Lucy algorithm
    :param img: Blurred image
    :param psf: PSF
    :param path: save path
    :param pad: The number of edge-filled lines
    :param niter: Maximum number of iterations
    :param eps: Stop-iter threshold
    :return:
    """
    print("start Richardson-Lucy deconvolution...")
    start_time0 = time.time()
    img_pad, psf_pad, padding = utils.pad_3d(img, psf / np.sum(psf), pad)

    otf = np.fft.fftn(psf_pad)
    otf_mirror = np.fft.fftn(np.flip(psf_pad, axis=[0, 1, 2]))
    result = img_pad.copy()

    for iters in range(niter):
        start_time = time.time()
        result0 = result.copy()
        result_fft = np.fft.fftn(result)
        tmp_fft = result_fft * otf
        tmp = np.real(np.fft.fftshift(np.fft.ifftn(tmp_fft)))
        tmp = img_pad / tmp
        tmp_fft = np.fft.fftn(tmp)
        tmp_fft = tmp_fft * otf_mirror
        tmp = np.real(np.fft.fftshift(np.fft.ifftn(tmp_fft)))
        result = result * tmp
        result[np.where(result < 0)] = 0
        delta = np.linalg.norm((result - result0).reshape(-1), ord=1) / np.linalg.norm(result0.reshape(-1), ord=1)
        end_time = time.time()
        execution_time = end_time - start_time
        print("iter:", iters + 1, ", delta:", delta, ", took", execution_time, "secs")

        if path is not None:
            if (iters + 1) % 10 == 0:
                if img_pad.shape != img.shape:
                    img_save = utils.unpad_3d(result, padding)
                    save_tiff_3d(path + "_RL" + str(iters + 1) + ".tif", img_save)
                else:
                    save_tiff_3d(path + "_RL" + str(iters + 1) + ".tif", result)

        if delta < eps or iters == niter - 1:
            print("Richardson-Lucy complete at", iters + 1, "th iter, ", end='')
            break

    if img_pad.shape != img.shape:
        result = utils.unpad_3d(result, padding)

    end_time = time.time()
    execution_time = end_time - start_time0
    print("took", execution_time, "secs!")
    return result


def rl_lp_3d(img, psf, pad=0, niter=50, eps=1e-2):
    """
    Three-dimensional Richardson-Lucy algorithm
    :param img: Blurred image
    :param psf: PSF
    :param pad: The number of edge-filled lines
    :param niter: Maximum number of iterations
    :param eps: Stop-iter threshold
    :return:
    """
    print("start Richardson-Lucy with Large photon flux...")
    img_pad, psf_pad, padding = utils.pad_3d(img, psf / np.sum(psf), pad)

    psf_roll = np.fft.fftshift(psf_pad)
    otf = np.fft.fftn(psf_roll)
    otf_mirror = np.fft.fftn(np.flip(psf_roll, axis=[0, 1, 2]))
    img_fft = np.fft.fftn(img_pad)
    result = img_pad.copy()

    for iters in range(niter):
        result0 = result.copy()
        result_fft = np.fft.fftn(result)
        tmp_fft = result_fft * otf
        tmp_fft = img_fft - tmp_fft
        tmp_fft = tmp_fft * otf_mirror
        tmp = np.real(np.fft.ifftn(tmp_fft))
        result = result + tmp
        result[np.where(result < 0)] = 0
        delta = np.linalg.norm((result - result0).reshape(-1), ord=1) / np.linalg.norm(result0.reshape(-1), ord=1)
        print("iter:", iters + 1, "delta:", delta)
        if delta < eps or iters == niter - 1:
            print("Richardson-Lucy with Large photon flux complete at", iters + 1, "th iter!")
            break

    if img_pad.shape != img.shape:
        return utils.unpad_3d(result, padding)
    return result


def nearest_neighbor_3d(img, psf, focus, c=0.9, path=None, snr=100):
    print("start nearest neighbor deconvolution...")
    start_time = time.time()
    psf_list = [psf[focus - 1], psf[focus], psf[focus + 1]]

    psf = np.float64(psf_list)
    for i in range(len(psf)):
        psf[i] /= psf[i].sum()

    obj_list = []
    obj_w_list = []
    for i in range(1, len(img) - 1):
        obj, obj_w = nearest_neighbor([img[i - 1], img[i], img[i + 1]], psf, c, snr)
        obj_list.append(obj)
        obj_w_list.append(obj_w)

    obj_array = np.array(obj_list)
    obj_w_array = np.array(obj_w_list)

    obj_array = (obj_array - obj_array.min()) / (obj_array.max() - obj_array.min())
    obj_w_array = (obj_w_array - obj_w_array.min()) / (obj_w_array.max() - obj_w_array.min())

    end_time = time.time()
    execution_time = end_time - start_time
    print("Nearest neighbor deconvolution took", execution_time, "secs!")

    if path is not None:
        save_tiff_3d(path + "_nearest_neighbor.tif", obj_array)
        save_tiff_3d(path + "_nearest_neighbor_wiener.tif", obj_w_array)
    return obj_list, obj_w_list


def nearest_neighbor(img_list, psf_list, c, snr):
    blur_sum = np.complex128(np.zeros(img_list[1].shape))
    # 上下两层分别模糊
    for i in [0, 2]:
        psf_f = np.fft.fft2(np.fft.fftshift(psf_list[i]))
        img_f = np.fft.fft2(img_list[i])
        img_blur_f = psf_f * img_f
        blur_sum += img_blur_f

    img_f = np.fft.fft2(img_list[1])
    obj_f = img_f - c * blur_sum / 2
    obj = np.real(np.fft.ifft2(obj_f))
    obj[np.where(obj < 0)] = 0

    # 对中间维纳滤波
    epsilon = 1 / float(snr)
    psf_f = np.fft.fft2(np.fft.fftshift(psf_list[1]))
    psf_f2 = np.abs(psf_f) ** 2
    filter_w = psf_f.conjugate() / (psf_f2 + epsilon ** 2)

    obj1 = blur_edge_2d(obj)
    obj_f = np.fft.fft2(obj1)
    obj_w_f = obj_f * filter_w
    obj_w = np.real(np.fft.ifft2(obj_w_f))
    obj_w[np.where(obj_w < 0)] = 0

    return obj, obj_w


def save_tiff_3d(filename, img):
    img = np.uint16((img - img.min()) * 65535 / (img.max() - img.min()))
    stat = cv2.imwritemulti(filename, tuple(img),
                            (int(cv2.IMWRITE_TIFF_RESUNIT), 1, int(cv2.IMWRITE_TIFF_COMPRESSION), 1))
    if stat:
        print("Successfully save", filename, "!")
        return

# def background_estimation(img, level=7, iters=3):
#     """
#     小波变换背景估计
#     :param img:
#     :param level:
#     :param iters:
#     :return:
#     """
#
#     shape = img.shape
#     background = np.zeros(shape)
#     for frames in range(shape[0]):
#         initial = img[frames]
#         res = initial
#         for ii in range(iters):
#
#     [m, n] = wavedec2(res, dlevel, wavename); %将一层进行小波变换，m是各层分解系数，n为各层分解系数长度（大小）
#     vec = zeros(size(m));
#     vec(1: n(1) * n(1) * 1) = m(1: n(1) * n(1) * 1);
#     Biter = waverec2(vec, n, wavename);
#     if th > 0
#         eps = sqrt(abs(res)) / 2; %与原图的这个值比较
#         ind = initial > (Biter + eps);
#         res(ind) = Biter(ind) + eps(ind);
#         [m, n] = wavedec2(res, dlevel, wavename);
#         vec = zeros(size(m));
#         vec(1: n(1) * n(1) * 1) = m(1: n(1) * n(1) * 1);
#         Biter = waverec2(vec, n, wavename);
#     end
#
#
# end
# Background(:,:, frames) = Biter;
# progressbar(frames / size(imgs, 3));
# end
# Background = Background(1:x, 1: y,:); %又回到原来的大小
