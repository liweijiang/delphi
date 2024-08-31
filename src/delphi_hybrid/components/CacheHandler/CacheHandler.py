"""
File: CacheHandler.py
Note: The parent class of all CacheHandlers
"""

import os
import sys
import abc
import json
from tqdm import tqdm

sys.path.append(os.getcwd())
from scripts.utils.utils import *

class CacheHandler():
    __metaclass__ = abc.ABCMeta

    def __init__(self, cache_type, cache_dir="cache", filename=None):
        self.cache_type = cache_type
        self.filename = filename
        self.cache_dir = cache_dir
        self._load_cache()
        print(f"{cache_type} cache size: {len(self.cache)}")

    def __name__(self):
        return self.cache_type + "CacheHandler"

    def _get_cache_path(self):
        if self.filename == None:
            path = data_base_path + f"{self.cache_dir}/{self.cache_type}.json"
        else:
            path = data_base_path + f"{self.cache_dir}/{self.filename}.json"
        return path

    def _load_cache(self):
        cache_file_path = self._get_cache_path()
        cache_dir_path = "/".join(cache_file_path.split("/")[:-1])
        if not os.path.exists(cache_file_path):
            ensure_dir(cache_dir_path)
            with open(cache_file_path, 'w') as f:
                json.dump({}, f)
        print(f"Loading {self.cache_type} cache from ...")
        print(cache_file_path)
        self.cache = read_json(cache_file_path)
        print(f"...done loading {self.cache_type} cache!")

    def _save_cache(self):
        cache_file_path = self._get_cache_path()
        with open(cache_file_path, 'w') as f:
            json.dump(self.cache, f)
        print("Saved cache to", cache_file_path)

    def _add_instance(self, cache_key, instance, is_save):
        self.cache[cache_key] = instance
        if is_save:
            self._save_cache()

    @abc.abstractmethod
    def _generate_instance(self, event):
        pass

    def fetch_instance(self, event):
        if event not in self.cache or (event in self.cache and self.cache[event] == None):
            return None
        else:
            return self.cache[event]

    def get_instance(self, event, is_save=False):
        if event not in self.cache or (event in self.cache and self.cache[event] == None):
            instance = self._generate_instance(event)
            self._add_instance(event, instance, is_save)
        else:
            # print("[Note] Instance exists in cache!")
            instance = self.cache[event]
        return instance

    def update_instance(self, event, is_save=True):
        """
        Regenerate instance (no matter if it's already in the cache or not) and update the cache
        """
        instance = self._generate_instance(event)
        self._add_instance(event, instance, is_save=is_save)
        return instance

    def save_instance(self, event):
        return self.get_instance(event, is_save=True)

    # def save_all_instance(self, events):
    #     for event in tqdm(events):
    #         self.save_instance(event)

    def save_all_instance(self, events, save_interval=1):
        for i, event in enumerate(tqdm(events)):
            if type(event) != type(""):
                continue
            self.get_instance(event)
            if i % save_interval == 0:
                self._save_cache()
        self._save_cache()

    def regenerate_all_instance(self, events, save_interval=1):
        for i, event in enumerate(tqdm(events)):
            if type(event) != type(""):
                continue
            self.update_instance(event, is_save=False)
            if i % save_interval == 0:
                self._save_cache()
        self._save_cache()

if __name__ == "__main__":
    delphi_cache_handler = CacheHandler("delphi_scores")
    delphi_cache = delphi_cache_handler.cache
