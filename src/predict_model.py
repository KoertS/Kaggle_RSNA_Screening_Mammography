import argparse

import pandas as pd
import yaml

from data.datagenerator import get_test_generator, get_test_dataframe
from models.model import load_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, nargs='?', default='../config/config.yaml',
                        help='Path to the config file')
    parser.add_argument("--kaggle", nargs="?", default='local', const="kaggle", help="Use Kaggle data paths")
    parser.add_argument('--model', type=str, nargs='?', default='test_model.h5', help='Filename of model h5 file')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = load_model(name=args.model, dir_models=config['data']['dir_models'])
    df_test = get_test_dataframe(config_data=config['data'], environment=args.kaggle)
    prediction_ids = df_test["prediction_id"].unique()

    test_gen = get_test_generator(test_dataframe=df_test, config_data=config['data'], environment=args.kaggle)

    y_pred = model.predict(test_gen).T[0]
    df_submission = pd.DataFrame({'prediction_id': prediction_ids, 'cancer': y_pred})
    print(df_submission.head())
    df_submission.to_csv(config['data']['path_submission'], index=False)
