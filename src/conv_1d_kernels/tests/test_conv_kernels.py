import unittest
from conv_1d_kernels import CConvKernelMovingAverage, CConvKernelTriangle


class TestConvKernels(unittest.TestCase):

    def mask_size(self, filter):
        for kernel_size in [3, 5, 9, 11]:
            filter.kernel_size = kernel_size
            self.assertTrue(filter.mask.size == kernel_size)


class TestConvKernelTriangle(TestConvKernels):

        def setUp(self):
            self.filter = CConvKernelTriangle()

        def test_kernel_size(self):
            self.mask_size(self.filter)


class TestConvKernelMovingAverage(TestConvKernels):

    def setUp(self):
        self.filter = CConvKernelMovingAverage()

    def test_kernel_size(self):
        self.mask_size(self.filter)
