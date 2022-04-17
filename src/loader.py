import os
import torch
import pathlib
import numpy as np
import glob as glob
from PIL import Image
from torch.utils.data import DataLoader

class ImageProcessor():
    """
    Function to preprocess the images from the custom 
    dataset. It includes a series of transformations:
    - At __init__ we convert the image to the desired [colorspace].
    - Crop function crops the image to the desired [proportion].
    - Scale scales the images to desired size [n]x[n].
    - Compress compresses the image to a given resolution.
    - Normalize performs the normalization of the channels.
    """
    def __init__(self):
        self.image = None

    def read(self, image_path, colorspace='RGB'):
        self._image = Image.open(image_path).convert(colorspace) 
        self.image = self._image.copy()
    
    def crop(self, image, proportion = 2 ** 6):
        nx, ny = image.size
        n = min(nx, ny)
        left = top = n / proportion
        right = bottom = (proportion - 1) * n / proportion
        return image.crop((left, top, right, bottom))

    def scale(self, image, n = 256):
        return image.resize((n, n), Image.ANTIALIAS)
        
    def compress(self, image, quality=0):
        image.save('/tmp/aux.jpg', format='JPEG', quality=1)
        return Image.open('/tmp/aux.jpg')

    def normalize(self, image):
        return np.array(image).astype('float') / 255.0

    def process(self, image_path):
        self.read(image_path=image_path)
        self._image = self.crop(self._image)
        self.image = self.crop(self.image)
        self.image = self.scale(self.image)
        self.image = self.compress(self.image)
        self.image = self.normalize(self.image)
        
        self._image = self.normalize(self._image)


class ImageDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        image_root: str,
        categories: list = [],
        split: str = 'train',
        rgb: bool = True,
        image_extension: str = "JPEG",
    ):

        self._image_data_path = image_root
        self.categories = categories
        self._colorspace = 'RGB' if rgb else 'L'
        self.processor = ImageProcessor()

        print(f'IMAGE DATA LOCATED AT: {self._image_data_path}')

        self.image_extension = image_extension
        self._MAX_LIMIT = 10

        self._index = 0
        self._indices = []
        for category in self.categories:
            print(category)
            directory = f'{self._image_data_path}/{category}'
            file_count = sum(len(files) for _, _, files in os.walk(directory))
            if self.split == 'train':
                irange = range(0, int(0.8 * file_count))
            elif self.split == 'test':
                irange = range(file_count - int(0.8 * file_count), file_count)
            for i in irange:
                if self._index >= self._MAX_LIMIT:
                    break
                self._index += 1
                self._indices.append(f'{directory}/{i + 1}')

    def __len__(self):
        return self._index

    def __getitem__(self, index):
        img_in = np.load(f'{self._indices[index]}_in', mmap_mode='r+').astype('float64')
        img_out = np.load(f'{self._indices[index]}_out', mmap_mode='r+').astype('float64')
        return (img_in, img_out)

if __name__ == '__main__':
    MY_DATA_FOLDER = '../data'
    print('Preparing data...')
    mappings = []
    with open(f'{MY_DATA_FOLDER}/mappings.txt') as f:
        for line in f:
            (key, i, img) = line.split()
            mappings.append(img)

    dataset = ImageDataset(
        image_root=MY_DATA_FOLDER, 
        categories=mappings,
        split='train', 
        rgb=True,
        image_extension='JPEG'
    )
    print('Dataset prepared.')

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    print('Data loaded ++')
