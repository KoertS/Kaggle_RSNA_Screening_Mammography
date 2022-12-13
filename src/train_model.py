import os

import wandb
import yaml

from data.train_val_split import get_train_val_generator
from models.model import build_test_model, pfbeta_tf

with open('../../config/config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

wandb.login(key=os.environ.get("WANDB_API_KEY"))
wandb.init(project=os.environ.get("WANDB_PROJECT_NAME"))
wandb.config.update(config['hyperparams'])
wandb_callback = wandb.keras.WandbCallback(log_weights=True)

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
history = model.fit(train_gen, validation_data=val_gen, epochs=config['hyperparams']['epochs'],
                    callbacks=[wandb_callback])

path_models = config['data']['path_model']
model_name = wandb.run.name.replace('-', '_')
model_path = f'{path_models}/{model_name}'
model.save(model_path)
