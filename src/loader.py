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
        folder: str = 'train',
        rgb: bool = True,
        image_extension: str = "JPEG",
    ):

        self._image_data_path = pathlib.Path(image_root) / folder
        self._colorspace = 'RGB' if rgb else 'L'
        self.processor = ImageProcessor()

        print(f'IMAGE DATA LOCATED AT: {self._image_data_path}')

        self.image_extension = image_extension
        self._MAX_LIMIT = 10000

        self._index = 0
        for img in glob.glob(f'{self._image_data_path}/*.{self.image_extension}'):
            self._index += 1
        #        if folder == 'train' or (folder == 'test' and test_i > self._MAX_LIMIT):
        #            self._indices.append((key, re.search(r'(?<=_)\d+', img).group()))
        #            self._index += 1
        #        if self._index == self._MAX_LIMIT:
        #            break
            if self._index >= self._MAX_LIMIT:
                break

    def __len__(self):
        return self._index

    def __getitem__(self, index):
        #key = self._indices[index][0]
        #indexer = self._indices[index][1]

        img_path = glob.glob(f'{self._image_data_path}/*.{self.image_extension}')[0]
        
        self.processor.process(img_path).astype('float64')
        
        return (self.processor._image, self.processor.image)