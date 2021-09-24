import logging
import os
import urllib.request
import zipfile

log = logging.getLogger(__name__)

_CACHE_DIR = "./datasets"


def get_cache_path(*paths, cache_dir=None):
    if not cache_dir:
        cache_dir = _CACHE_DIR

    return os.path.join(cache_dir, *paths)


def download_file(url, file_path, unzip=False, cache_dir=None):
    file_path = get_cache_path(file_path, cache_dir=cache_dir)
    if not os.path.exists(file_path):
        log.warning(f"Downloading {url} to {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        urllib.request.urlretrieve(url, file_path)
        if unzip:
            with zipfile.ZipFile(file_path) as zip_ref:
                zip_ref.extractall(os.path.dirname(file_path))

    return file_path
