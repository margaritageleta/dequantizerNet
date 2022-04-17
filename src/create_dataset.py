from downloader import ImageDownloader
import os
from git import refresh
from loader import ImageProcessor
from py import process
import numpy as np
from tqdm import tqdm

N_PHOTOS = 10
PREPROCESSED_DIR = "../preprocessed_data"
DATA_DIR = "../data"


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
print("Reading categories list...")
categories = []
with open(f'../categories.txt') as f:
    for line in f:
        (key, i, img) = line.split()
        categories.append(img)
print("Categories list read")
downloader = ImageDownloader('../config.cfg')
processor = ImageProcessor()
with tqdm(categories) as t:
    for category in categories:
        t.set_description(f"Downloading images with tag <<{category}>>", refresh=True)
        downloader.download(PREPROCESSED_DIR, category=category, n_photos=N_PHOTOS)
        preprocessed_folder = f"{PREPROCESSED_DIR}/{category}/"
        data_folder = f"{DATA_DIR}/{category}/"
        create_folder(data_folder)
        t.set_description(f"Processing images with tag <<{category}>>", refresh=True)
        for id in range(1, N_PHOTOS+1):
            processor.process(f"{preprocessed_folder}{id}.jpg")
            np.save(f"{data_folder}{id}_in", processor._image, allow_pickle=True)
            np.save(f"{data_folder}{id}_out", processor.image, allow_pickle=True)





