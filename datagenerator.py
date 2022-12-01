import numpy as np
import tensorflow as tf


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, path="train_images/", batch_size=32, shuffle=True):
        self.df = df
        self.path = path
        self.n = len(self.df)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(self.n / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        batch = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(batch)
        return X, y

    def __get_input(self, path):
        image = tf.keras.preprocessing.image.load_img(path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        return image_arr

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __data_generation(self, df_batch):
        paths = self.path + df_batch["patient_id"].astype(str) + "/" + df_batch["image_id"].astype(str) + ".png"
        X = np.asarray([self.__get_input(path) for path in paths])
        y = df_batch['cancer']
        return X, y