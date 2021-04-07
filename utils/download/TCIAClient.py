from pathlib import Path
from zipfile import ZipFile
import requests

class Client:
    def __init__(self, url):
        if url[-1] != '/':
            url = url + '/'
        self.base_url = url
        
    def get_json(self, get, params={}):
        params['format'] = 'json'
        url = self.base_url + get

        return requests.get(url, params=params).json()
        
    def get_image(self,
                  series_instance_uid,
                  local_path,
                  unzip,
                  remove_zip):
        
        url = self.base_url + 'getImage'
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True)
        
        params = {'SeriesInstanceUID': series_instance_uid}
        with requests.get(url, params=params, stream=True) as req:
            req.raise_for_status()
            with open(local_path, 'wb') as ff:
                for chunk in req.iter_content(chunk_size=8192): 
                    ff.write(chunk)
        if unzip:
            with ZipFile(local_path, 'r') as zz:
                try:
                    zz.extractall(local_path.parent)
                except:
                    print(f'unzipping for {local_path} failed')
            if remove_zip:
                local_path.unlink()
        return True
