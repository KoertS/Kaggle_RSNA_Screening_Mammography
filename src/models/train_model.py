import os

import wandb
import yaml

from src.data.train_val_split import get_train_val_generator
from src.models.model import build_test_model, pfbeta_tf

with open('../../config/config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

wandb.init(project=os.environ.get("WANDB_PROJECT_NAME"))
wandb.config.update(config['hyperparams'])

model = build_test_model()
train_gen, val_gen = get_train_val_generator(path_dataframe=config['data']['train_dataframe'],
                                             path_images=config['data']['train_images_processed_dir'],
                                             train_size=config['hyperparams']['train_size'],
                                             batch_size=config['hyperparams']['batch_size'])
metrics = [pfbeta_tf]
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=metrics)
print(model.summary())
history = model.fit(train_gen, validation_data=val_gen, epochs=config['hyperparams']['epochs'])
model.save()
