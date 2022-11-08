from conv_1d_kernels import CConvKernel
import numpy as np


class CConvKernelMovingAverage(CConvKernel):

    def kernel_mask(self):
        self._mask = np.ones(shape=self.kernel_size) / self.kernel_size
