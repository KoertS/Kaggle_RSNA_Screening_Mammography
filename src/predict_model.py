import argparse

import pandas as pd
import tensorflow as tf
import yaml

from data.datagenerator import DataGenerator
from models.model import pfbeta_tf

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, nargs='?', default='../config/config.yaml', help='Path to the config file')
parser.add_argument('--model', type=str, nargs='?', default='test_model.h5', help='Filename of model h5 file')

args = parser.parse_args()

with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

custom_metric = {"pfbeta_tf": pfbeta_tf}
model = tf.keras.models.load_model(config['data']['path_model'] + args.model, custom_objects=custom_metric)

df_test = pd.read_csv(config['data']['test_frame'])
patient_ids = df_test["patient_id"].unique()
prediction_ids = df_test["prediction_id"].unique()

test_gen = DataGenerator(dataframe=df_test, batch_size=1, path_images=config['data']['test_images_processed_dir'])

y_pred = model.predict(test_gen).T[0]
df_submission = pd.DataFrame({'prediction_id': prediction_ids, 'cancer': y_pred})
print(df_submission.head())
df_submission.to_csv(config['data']['path_submission'], index=False)
