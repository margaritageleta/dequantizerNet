import flickr_api
import configparser
import os

class ImageDownloader:
    def __init__(self, configFile):
        self.configFile = configFile
        self.config = configparser.RawConfigParser()
        self.config.read(configFile)
        flickr_api.set_keys(api_key = self.config.get('FLICKR','key'), api_secret = self.config.get('FLICKR','secret'))
    
    def download(self, outDir, category = None, n_photos = 100):
        if category:
            self._download_by_search(outDir, category, n_photos)
        else:
            self._download_recent(outDir, n_photos)
        
    def _download_by_search(self, outDir, category, n_photos):
        photos = flickr_api.Walker(flickr_api.Photo.search, tags=category)
        
        folder = f'{outDir}/{category}/'
        self._create_folder(folder)
        self._save_photos(photos, folder, 0, n_photos)

    def _download_recent(self, outDir, n_photos):
        photos = flickr_api.Walker(flickr_api.Photo.getRecent)
        lastId = int(self.config.get('FLICKR', 'lastId'))
        
        folder = f'{outDir}/recent/'
        self._create_folder(folder)
        
        self._save_photos(photos, folder, lastId, n_photos)
        
        self.config.set('FLICKR', 'lastId', lastId + n_photos)
        
        with open(self.configFile, 'w') as fp:
            self.config.write(fp)
    
    def _create_folder(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        
    def _save_photos(self, photos, folder, lastId, n_photos):
        count = 0
        for photo in photos:
            if count == n_photos:
                break
            try:
                photo.save(filename=f'{folder}/{lastId+count+1}', size_label='Medium 640')
                count += 1
            except:
                continue