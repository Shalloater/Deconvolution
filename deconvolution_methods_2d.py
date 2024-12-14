import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import utils


def blur_2d(img, psf, path=None, show=0):
    psf = np.float32(psf)
    img = np.float32(img)
    psf = psf / np.sum(psf)
    img = img / np.max(img)

    otf = np.fft.fft2(np.fft.fftshift(psf))
    img_f = np.fft.fft2(img)
    img_blur_f = img_f * otf
    img_blur = np.real(np.fft.ifft2(img_blur_f))
    img_blur[np.where(img_blur < 0)] = 0
    img_blur = (img_blur - img_blur.min()) * 256 / (img_blur.max() - img_blur.min())
    img_blur = np.int16(img_blur)

    # 加一个掩膜

    if show:
        plt.figure()
        plt.subplot(231), plt.imshow(img, cmap='gray'), plt.axis("off"), plt.title("Original")
        plt.subplot(232), plt.imshow(psf, cmap='gray'), plt.axis("off"), plt.title("PSF")
        plt.subplot(233), plt.imshow(img_blur, cmap='gray'), plt.axis("off"), plt.title("Blur")
        plt.subplot(234), plt.imshow(np.log(np.abs(np.fft.fftshift(img_f)) + 1e-5)), plt.axis("off"), plt.title("F_ori")
        plt.subplot(235), plt.imshow(np.log(np.abs(np.fft.fftshift(otf)) + 1e-5)), plt.axis("off"), plt.title("OTF")
        plt.subplot(236), plt.imshow(np.log(np.abs(np.fft.fftshift(img_blur_f)) + 1e-5)), plt.axis("off"), plt.title(
            "F_blur")
        plt.show()

    if path is not None:
        cv2.imwrite(path + "_blur.png", img_blur)

    return img_blur


def rl_2d(img, psf, path=None, pad=0, niter=50, eps=1e-2):
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
    img_blur = utils.blur_edge_2d(img, pad)

    otf = np.fft.fftn(psf)
    otf_mirror = np.fft.fftn(np.flip(psf, axis=[0, 1]))
    result = img.copy()

    for iters in range(niter):
        start_time = time.time()
        result0 = result.copy()
        result_fft = np.fft.fftn(result)
        tmp_fft = result_fft * otf
        tmp = np.real(np.fft.fftshift(np.fft.ifftn(tmp_fft)))
        tmp = img / tmp
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


def rl_2d_accelerate(img, psf, path=None, niter=50, show=0):
    # initialization
    eps = 1e-6
    psf = np.float32(psf)
    img = np.float32(img)
    psf = psf / np.sum(psf)
    img = img / np.max(img)

    shape = img.shape
    pad = np.int16(np.floor(min(shape) / 6))
    img_pad = np.pad(img, pad, "edge")
    psf_pad = np.zeros(img_pad.shape)
    psf_pad[0:psf.shape[0], 0: psf.shape[1]] = psf
    # index = np.where(kernel == np.max(kernel))
    # x, y = index[0][0], index[1][0]
    psf_pad = np.fft.fftshift(psf_pad)

    xk = img_pad
    yk = np.zeros(img_pad.shape)
    vk = np.zeros(img_pad.shape)

    otf = np.fft.fft2(psf_pad)

    rl_iter = lambda estimate, data, kernel_f: np.fft.fft2(
        data / np.max(np.fft.ifft2(kernel_f * np.fft.fft2(estimate))))
    # 伤心
    for i in range(niter):
        yk_update = yk
        yk = xk * np.real(np.fft.ifft2(np.conj(otf) * rl_iter(xk, img_pad, otf))) / np.real(
            np.fft.ifft2(otf * np.fft.fft2(np.ones(img_pad.shape))))
        yk[np.where(yk < 1e-6)] = 1e-6
        vk_update = vk
        vk = xk - yk
        vk[np.where(vk < 1e-6)] = 1e-6
        if i == 0:
            alpha = 0
        else:
            alpha = np.sum(vk * vk_update) / (np.sum(vk_update * vk_update) + eps)
            alpha = max(min(alpha, 1), 1e-6)
        xk = yk + alpha * (yk - yk_update)
        xk[np.where(xk < 1e-6)] = 1e-6
        xk[np.where(xk == np.nan)] = 1e-6
        print("Iter:", i + 1)

    xk[np.where(xk < 0)] = 0
    img_decon = xk[pad:- pad, pad: -pad]

    if show:
        plt.figure()
        plt.subplot(111), plt.imshow(img_decon, cmap='gray'), plt.axis("off"), plt.title("Original")
        # plt.subplot(232), plt.imshow(psf, cmap='gray'), plt.axis("off"), plt.title("PSF")
        # plt.subplot(233), plt.imshow(img_blur, cmap='gray'), plt.axis("off"), plt.title("Blur")
        # plt.subplot(234), plt.imshow(np.log(np.abs(np.fft.fftshift(img_f)) + 1e-5)), plt.axis("off"), plt.title("F_ori")
        # plt.subplot(235), plt.imshow(np.log(np.abs(np.fft.fftshift(otf)) + 1e-5)), plt.axis("off"), plt.title("OTF")
        # plt.subplot(236), plt.imshow(np.log(np.abs(np.fft.fftshift(img_blur_f)) + 1e-5)), plt.axis("off"), plt.title(
        #     "F_blur")
        plt.show()

    return img_decon


if __name__ == '__main__':
    source_path = r"240112/source/stride2.tif"
    kernel_path = r"240112/source/PSF_virtual1_128.tif"
    source = cv2.imread(source_path, cv2.IMREAD_UNCHANGED)
    kernel = cv2.imread(kernel_path, cv2.IMREAD_UNCHANGED)
    result_path = "240112/result/stride2"
    source_blur = blur_2d(source, kernel,result_path, show=1)
    # rl_2d(source_blur, kernel, result_path, niter=10000, show=1)
