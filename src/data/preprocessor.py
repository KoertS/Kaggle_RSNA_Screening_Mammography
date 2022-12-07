import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image


class Preprocessor:
    def __init__(self, input_directory, output_directory, resize_to=(256, 256), workers=64):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.resize_to = resize_to
        self.workers = workers

    def preprocess(self):
        directories_patients = list(Path(self.input_directory).iterdir())
        with mp.Pool(processes=self.workers) as p:
            p.map(self.process_directory_patient, directories_patients)

    def process_directory_patient(self, directory_patient):
        parent_directory = os.path.split(directory_patient)[-1]
        processed_dir = f'{self.output_directory}/{parent_directory}'
        make_dir(processed_dir)
        for image_path in directory_patient.iterdir():
            processed_ary = dicom_file_to_ary(image_path)
            im = Image.fromarray(processed_ary).resize(self.resize_to)
            im.save(f'{processed_dir}/{image_path.stem}.png')


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
