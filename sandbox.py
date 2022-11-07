import numpy as np
import matplotlib.pyplot as plt

# plt.tight_layout()
path = '/home/jperez/data/HKH/processed_L07_2005/train/tiff_140_slice_1.npy'
arr = np.load(path)[:, :, 8]
plt.figure(figsize=(10, 15))
print(arr.shape, arr.dtype, arr.min(), arr.max())
plt.imshow(arr, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.savefig('fig.png')