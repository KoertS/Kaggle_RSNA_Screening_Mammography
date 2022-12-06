import pandas as pd
import tensorflow as tf

from datagenerator import DataGenerator
from model import pfbeta_tf

image_dir = 'test_images_processed/'

custom_metric = {"pfbeta_tf": pfbeta_tf}
path_model = 'models/test_model.h5'
model = tf.keras.models.load_model(path_model, custom_objects=custom_metric)

filename = 'input/test.csv'
df_test = pd.read_csv(filename)
patient_ids = df_test["patient_id"].unique()
prediction_ids = df_test["prediction_id"].unique()

test_gen = DataGenerator(df_test, batch_size=1, path=image_dir)
y_pred = model.predict(test_gen).T[0]
df_submission = pd.DataFrame({'prediction_id': prediction_ids, 'cancer': y_pred})
print(df_submission.head())
df_submission.to_csv('submission.csv', index=False)
