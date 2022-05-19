import os
import torch
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

class breakNestedLoops(Exception): pass

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
        self._image = self.scale(self._image)
        self._image = self.normalize(self._image)

        self.image = self.crop(self.image)
        self.image = self.scale(self.image)
        self.image = self.compress(self.image)
        self.image = self.normalize(self.image)

class ImageDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        image_root: str,
        categories: list = [],
        split: str = 'train',
        rgb: bool = True,
        image_extension: str = "JPEG",
        max_limit: int = 10000,
        percentage_train: float = 0.8,
        pixel_shuffle = True
    ):

        self._image_data_path = image_root
        self.categories = categories
        self.split = split
        self._colorspace = 'RGB' if rgb else 'L'
        self.processor = ImageProcessor()
        self.pixel_shuffle = pixel_shuffle

        print(f'IMAGE DATA LOCATED AT: {self._image_data_path}')

        self.image_extension = image_extension
        self._MAX_LIMIT = max_limit
        
        print(f'MAX LIMIT SET TO: {self._MAX_LIMIT}')

        self._index = 0
        self._indices = []
        try:
            for category in self.categories:
                directory = f'{self._image_data_path}/{category}'
                indexes = set()
                for _, _, files in os.walk(directory):
                    for fileName in files:
                        indexes.add(fileName.split('_')[0])
                    self._index += len(indexes)
                    if self._index >= self._MAX_LIMIT:
                        raise breakNestedLoops()
                self._indices += [f'{directory}/{i}' for i in sorted(list(indexes))]
        except breakNestedLoops as e: 
            self._indices += [f'{directory}/{i}' for i in sorted(list(indexes))]
                              
        random.shuffle(self._indices)
        L = len(self._indices)
        if self.split == 'train':
            self._indices = self._indices[0:int(percentage_train * L)]
        elif self.split == 'test':
            self._indices = self._indices[int(percentage_train * L):L]
        else: raise Exception('Unknown split. Use train or test.')
        self._index = len(self._indices)
        print(f'SAMPLES IN DATALOADER: {self._index}')
        if self.pixel_shuffle:
            self.zero_channel = torch.zeros(1, 256, 256, requires_grad=False)
        

    def __len__(self):
        return self._index

    def __getitem__(self, index):
        img_in = torch.from_numpy(np.load(f'{self._indices[index]}_in.npy', mmap_mode='r+', allow_pickle=True).astype('float32').transpose((2,0,1)))
        if self.pixel_shuffle:
            img_in = torch.cat((img_in ,self.zero_channel), 0)
        img_out = torch.from_numpy(np.load(f'{self._indices[index]}_out.npy', mmap_mode='r+', allow_pickle=True).astype('float32').transpose((2,0,1)))
        return (img_in, img_out)

if __name__ == '__main__':
    MY_DATA_FOLDER = f"{os.environ.get('DATA_PATH')}/data"
    print('Preparing data...')
    mappings = []
    with open(os.path.join(os.environ.get('ROOT_PATH'), os.environ.get('CAT_FILE'))) as f:
        for line in f:
            (key, i, img) = line.split()
            mappings.append(img)

    dataset = ImageDataset(
        image_root=MY_DATA_FOLDER, 
        categories=mappings,
        split='train', 
        rgb=True,
        image_extension='JPEG',
        max_limit=10
    )
    print('Dataset prepared.')

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print('Data loaded ++')
    # for i, batch in enumerate(dataloader):
    #     print(i, batch)
    print(len(dataloader.dataset))
