import argparse
import os

import wandb
import yaml

from data.datagenerator import get_train_val_generator
from models.model import create_model, save_model

parser = argparse.ArgumentParser()
parser.add_argument("--kaggle", nargs="?", default='local', const="kaggle", help="Use Kaggle data paths")
parser.add_argument('--config', type=str, nargs='?', default='../config/config.yaml', help='Path to the config file')
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

wandb.login(key=os.environ.get("WANDB_API_KEY"))
wandb.init(project="Kaggle_RSNA_Screening_Mammography")
wandb.config.update(config['hyperparams'])
wandb_callback = wandb.keras.WandbCallback(log_weights=True)

model = create_model(config['hyperparams'])
print(f'Training model: {model.name}')

train_gen, val_gen = get_train_val_generator(config=config, environment=args.kaggle)
history = model.fit(train_gen, validation_data=val_gen, epochs=config['hyperparams']['epochs'],
                    callbacks=[wandb_callback])
save_model(model, name=wandb.run.name.replace('-', '_'), dir_models=config['data']['dir_models'][args.kaggle])
