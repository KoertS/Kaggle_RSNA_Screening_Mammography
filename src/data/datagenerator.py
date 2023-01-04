import os

import numpy as np
import pandas as pd
import tensorflow as tf

from .extract_roi import extract_roi
from .preprocessor import Preprocessor


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, path_images, batch_size=32, shuffle=True, crop_roi=False):
        self.dataframe = dataframe.copy()
        if 'prediction_id' not in dataframe:
            self.dataframe['prediction_id'] = dataframe["patient_id"].astype(str) + '_' + dataframe[
                "laterality"].astype(str)
            self.labels = self.dataframe.groupby('prediction_id')['cancer'].max()
            self.inference = False
        else:
            self.inference = True
        self.prediction_ids = self.dataframe['prediction_id'].unique()
        self.path_images = path_images
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        self.crop_roi = crop_roi

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(len(self.prediction_ids) / self.batch_size)

    def __getitem__(self, index):
        """Generate one batch of data"""
        batch_indexes = self.prediction_ids[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(batch_indexes)
        return X, y

    def __get_input(self, path):
        image = tf.keras.preprocessing.image.load_img(path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        if self.crop_roi:
            image_arr = extract_roi(image_arr)
        return image_arr

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)

    def __data_generation(self, batch_indexes):
        paths = self.get_paths_images(batch_indexes)
        X = np.asarray([self.__get_input(path) for path in paths])
        if self.inference:
            return X, None
        else:
            y = np.array([self.labels[batch_indexes]])
            return X, y

    def get_paths_images(self, batch_indexes):
        batch = self.dataframe[self.dataframe['prediction_id'].isin(batch_indexes)]
        rows_batch = self.get_rows(batch)
        return self.path_images + rows_batch["patient_id"].astype(str) + "/" + rows_batch["image_id"].astype(
            str) + ".png"

    def get_rows(self, batch):
        """Select only 1 MLO view picture per breast"""
        only_MLO_view_images = batch[batch['view'] == 'MLO']
        only_one_per_prediction_id = only_MLO_view_images.groupby('prediction_id')[['patient_id', 'image_id']].max()
        return only_one_per_prediction_id


def get_train_val_generator(config, environment):
    df = get_train_dataframe(config['data'], environment)
    df_train, df_val = split_dataframe(df, config['hyperparams']['train_size'])
    path_images = config['data']['dir_processed'][environment]['train'] + config['data']['train_images_dir'][
        environment]
    train_gen = DataGenerator(dataframe=df_train, path_images=path_images,
                              batch_size=config['hyperparams']['batch_size'],
                              crop_roi=config['hyperparams']['crop_roi'])
    val_gen = DataGenerator(dataframe=df_val, path_images=path_images,
                            batch_size=config['hyperparams']['batch_size'], crop_roi=config['hyperparams']['crop_roi'])
    return train_gen, val_gen


def split_dataframe(df, ratio, shuffle=True):
    patient_ids = df["patient_id"].unique()
    if shuffle:
        np.random.shuffle(patient_ids)
    train_size = int(len(patient_ids) * ratio)
    train_ids = patient_ids[:train_size]
    val_ids = patient_ids[train_size:]
    df_train = df[df['patient_id'].isin(train_ids)]
    df_val = df[df['patient_id'].isin(val_ids)]
    return df_train, df_val


def get_train_dataframe(config_data, environment):
    path = config_data['dir_raw'][environment] + config_data['train_dataframe']
    return pd.read_csv(path)


def get_test_dataframe(config_data, environment):
    path = config_data['dir_raw'][environment] + config_data['test_dataframe']
    return pd.read_csv(path)


def get_test_generator(test_dataframe, config_data, environment):
    preprocessed_dir = config_data['dir_processed'][environment]['test'] + config_data['test_images_dir']
    if not os.path.isdir(preprocessed_dir):
        input_directory = config_data['dir_raw'][environment] + config_data['test_images_dir']
        print(f'{preprocessed_dir} not found')
        print(f'Preprocessing: {input_directory} to {preprocessed_dir}')
        preprocessor = Preprocessor(input_dir=input_directory,
                                    preprocessed_dir=preprocessed_dir,
                                    workers=32)
        preprocessor.preprocess()
    print(f'Creating datagenerator with images from: {preprocessed_dir}')
    return DataGenerator(dataframe=test_dataframe, batch_size=1, path_images=preprocessed_dir, shuffle=False)
