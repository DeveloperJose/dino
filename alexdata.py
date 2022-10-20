import pathlib
import multiprocessing

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset


class AlexData(Dataset):
    def __init__(self, path, transforms):
        self.transforms = transforms

        self.path = pathlib.Path(path)
        # self.img_files = list(self.path.glob('*.npy'))
        self.img_files = list(self.path.glob('*.png'))

    def __getitem__(self, i):
        im_path = self.img_files[i]

        # arr = np.load(im_path).T
        # im = Image.fromarray(arr)
        
        im = Image.open(im_path).convert('RGB')

        if self.transforms:
            im = self.transforms(im)
        return im, torch.Tensor()

    def __len__(self):
        return len(self.img_files)

DIM = 64
SKIP = 64

if __name__ == '__main__':
    path = pathlib.Path('/home/jperez/data/alexbrain/train/ff_images')
    output_path = pathlib.Path('/home/jperez/data/alexbrain/train/ff_images_64')

    def process(filename):
        im = np.array(Image.open(filename))
        v = np.lib.stride_tricks.sliding_window_view(im, (DIM, DIM, 3))[::SKIP, ::SKIP][:, :, 0].reshape((-1, DIM, DIM, 3))

        for idx in range(v.shape[0]):
            batch_im = v[idx].T
            np.save(output_path / f'{filename.stem}_slice_{idx}.npy', batch_im)

    image_paths = list(path.glob('*.png'))
    pbar = tqdm(total=len(image_paths), desc=f'Processing dataset')
    with multiprocessing.Pool(32) as pool:
        for _ in pool.map(process, image_paths):
            pbar.update(1)
    
    # for filepath in path.glob('*.png'):
    #     im = Image.open(filepath)
    #     arr = np.array(im, dtype=np.float32).T
    #     np.save(filepath.with_suffix('.npy'), arr)
    #     print(filepath, arr.shape)