import numpy as np
import tensorflow as tf


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, path_images, batch_size=32, shuffle=True):
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
