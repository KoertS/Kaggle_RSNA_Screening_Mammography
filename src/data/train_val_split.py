import numpy as np
import pandas as pd

from . import datagenerator

np.random.seed(0)


def get_train_val_generator(path_dataframe, path_images, train_size, batch_size):
    df = pd.read_csv(path_dataframe)

    patient_ids = df["patient_id"].unique()
    np.random.shuffle(patient_ids)
    train_size = int(len(patient_ids) * train_size)
    train_ids = patient_ids[:train_size]
    val_ids = patient_ids[train_size:]

    df_train = df[df['patient_id'].isin(train_ids)]
    df_val = df[df['patient_id'].isin(val_ids)]

    train_gen = datagenerator.DataGenerator(dataframe=df_train, path_images=path_images, batch_size=batch_size)
    val_gen = datagenerator.DataGenerator(dataframe=df_val, path_images=path_images, batch_size=batch_size)
    return train_gen, val_gen
