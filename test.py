import utils
import torch
import numpy as np
from scipy import signal

if __name__ == '__main__':
    # a=np.outer(signal.windows.gaussian(100, 0.05), signal.windows.gaussian(100, 0.05),)
    # b = np.outer(a,signal.windows.gaussian(100, 0.05))
    # print(b.shape)

    # 示例使用
    # size = (111,1025,1025)
    # sigma = 0.2
    # kernel = utils.generate_3d_gaussian_kernel(size, sigma)
    # print(kernel.shape)

    # 生成一个大小为5x5x5的三维高斯模糊核，标准差为1.0
    kernel = utils.generate_3d_gaussian_kernel((111, 1024, 1024), 1)
    print(kernel.shape)
