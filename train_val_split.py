import numpy as np
import pandas as pd

from datagenerator import DataGenerator

np.random.seed(0)
df = pd.read_csv('input/train.csv')

patient_ids = df["patient_id"].unique()
np.random.shuffle(patient_ids)
train_size = int(len(patient_ids) * 0.8)
train_ids = patient_ids[:train_size]
val_ids = patient_ids[train_size:]

df_train = df[df['patient_id'].isin(train_ids)]
df_val = df[df['patient_id'].isin(val_ids)]

train_gen = DataGenerator(df_train, batch_size=8)
val_gen = DataGenerator(df_val, batch_size=8)

for X, y in train_gen:
    print(X.shape)
    print(y.shape)
    print(y[0])
    exit()
