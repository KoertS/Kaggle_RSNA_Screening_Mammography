# import numpy as np
# import pandas as pd
#
# from . import datagenerator
#
# np.random.seed(0)
#
#
# def get_train_val_generator(config, environment):
#     path_dataframe = get_path_dataframe(config['data'], environment)
#     df = pd.read_csv(path_dataframe)
#     df_train, df_val = split_dataframe(df, config['hyperparams']['train_size'])
#     path_images = get_path_images(config['data'], environment)
#     train_gen = datagenerator.DataGenerator(dataframe=df_train, path_images=path_images,
#                                             batch_size=config['hyperparams']['batch_size'])
#     val_gen = datagenerator.DataGenerator(dataframe=df_val, path_images=path_images,
#                                           batch_size=config['hyperparams']['batch_size'])
#     return train_gen, val_gen
#
#
# def split_dataframe(df, ratio, shuffle=True):
#     patient_ids = df["patient_id"].unique()
#     if shuffle:
#         np.random.shuffle(patient_ids)
#     train_size = int(len(patient_ids) * ratio)
#     train_ids = patient_ids[:train_size]
#     val_ids = patient_ids[train_size:]
#     df_train = df[df['patient_id'].isin(train_ids)]
#     df_val = df[df['patient_id'].isin(val_ids)]
#     return df_train, df_val
#
#
# def get_path_dataframe(config_data, environment):
#     return config_data['dir_raw'][environment] + config_data['train_dataframe']
#
#
# def get_path_images(config_data, environment):
#     return config_data['dir_processed'][environment] + config_data['train_images_dir']
