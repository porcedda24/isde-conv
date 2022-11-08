import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from conv_1d_kernels import CConvKernelMovingAverage

data = pd.read_csv('../data/mnist_data.csv')
data = np.array(data)

labels = data[:, 0]
data = data[:, 1:] / 255

print(data.shape, labels.shape, np.unique(labels))

mask_size = 7
mask = np.ones(shape=(mask_size,)) * 1 / mask_size
print(mask)

k = (mask_size - 1) // 2
print(k)

x = data[0, :]  # extract the first sample
plt.imshow(x.reshape(28, 28))
plt.show()

# do the convolution
xp = x.copy()
for i in range(k, x.size - k):
    v = x[i - k:i + k + 1]
    xp[i] = np.dot(v, mask)

plt.subplot(1, 2, 1)
plt.imshow(x.reshape(28, 28))
plt.subplot(1, 2, 2)
plt.imshow(xp.reshape(28, 28))
plt.show()

# this has to call kernel_mask to create mask = [1 1 1 1 1]
filter = CConvKernelMovingAverage(kernel_size=5)
print(filter.mask)

# this has to call kernel_mask to create mask = [1 1 1]
filter.kernel_size = 7
print(filter.mask)

xp = filter.kernel(x)

plt.subplot(1, 2, 1)
plt.imshow(x.reshape(28, 28))
plt.subplot(1, 2, 2)
plt.imshow(xp.reshape(28, 28))
plt.show()