import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image


def dicom_file_to_ary(path):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def process_directory(directory_path, resize_to=(256, 256)):
    parent_directory = str(directory_path).split('/')[-1]
    processed_dir = f'test_images_processed_{resize_to[0]}/{parent_directory}'
    make_dir(processed_dir)
    for image_path in directory_path.iterdir():
        processed_ary = dicom_file_to_ary(image_path)
        im = Image.fromarray(processed_ary).resize(resize_to)
        im.save(f'{processed_dir}/{image_path.stem}.png')


def preprocess_images(path_directory):
    directories = list(Path(path_directory).iterdir())
    with mp.Pool(64) as p:
        p.map(process_directory, directories)
