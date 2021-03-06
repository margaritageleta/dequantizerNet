from downloader import ImageDownloader
import os
from loader import ImageProcessor
import numpy as np
from tqdm import tqdm
import shutil
import time
import torch

N_PHOTOS = int(os.environ.get('N_PHOTOS'))
PREPROCESSED_DIR = os.path.join(os.environ.get('DATA_PATH'), f'preprocessed_data')
DATA_DIR = os.path.join(os.environ.get('DATA_PATH'), f'data')

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def ack_download(file, n):
    with open(file, "w") as f:
        f.write(str(n))

def get_last_cat_downloaded(file):
    n = 0
    with open(file,"r") as f:
        n = f.readlines()
    return int(n[0])
    
print("Reading categories list...")
categories = []
with open(os.path.join(os.environ.get('ROOT_PATH'), os.environ.get('CAT_FILE'))) as f:
    for line in f:
        (key, i, img) = line.split()
        categories.append(img)

print("Categories list read")
downloader = ImageDownloader(os.path.join(os.environ.get('ROOT_PATH'), os.environ.get('CONFIG_FILE')))
processor = ImageProcessor(n=96)
last_cat = get_last_cat_downloaded(os.path.join(os.environ.get('ROOT_PATH'), os.environ.get('DWD_CAT_FILE')))
with tqdm(categories[last_cat:]) as t:
    for i, category in enumerate(t):
        t.set_description(f"Downloading images with tag <<{category}>>")
        errors = 0
        done = False
        while not done and errors < 10:
            try:
                n_downloaded = downloader.download(PREPROCESSED_DIR, category=category, n_photos=3*N_PHOTOS)
                preprocessed_folder = f"{PREPROCESSED_DIR}/{category}/"
                data_folder = f"{DATA_DIR}/{category}/"
                create_folder(data_folder)
                t.set_description(f"Processing images with tag <<{category}>>", refresh=True)
                id = 1
                for idppr in range(1, n_downloaded + 1):
                  if id > N_PHOTOS:
                    break
                  try:
                    processor.process(f"{preprocessed_folder}{idppr}.jpg")
                    np.save(f"{data_folder}{id}_in", processor.image, allow_pickle=True)
                    np.save(f"{data_folder}{id}_out", processor._image, allow_pickle=True)
                    torch.from_numpy(np.load(f"{data_folder}{id}_in.npy", mmap_mode='r+', allow_pickle=True).astype('float32').transpose((2,0,1)))
                    torch.from_numpy(np.load(f"{data_folder}{id}_out.npy", mmap_mode='r+', allow_pickle=True).astype('float32').transpose((2,0,1)))
                    id += 1
                  except Exception as e: 
                    print(e)
                    try:
                      os.remove(f"{data_folder}{id}_in.npy")
                    except:
                      pass

                    try:
                      os.remove(f"{data_folder}{id}_out.npy")
                    except:
                      pass
                shutil.rmtree(preprocessed_folder)
                done = True
                ack_download(os.path.join(os.environ.get('ROOT_PATH'), os.environ.get('DWD_CAT_FILE')), last_cat + i)
            except:
                errors += 1
                time.sleep(20)


      








