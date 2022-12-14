import os
from pathlib import Path

import efficientnet.tfkeras as efn
import tensorflow as tf


def build_test_model(inp_dim=(256, 256, 3)):
    inputs = tf.keras.layers.Input(inp_dim)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', strides=4)(inputs)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name='test_model')
    return model


def build_efficient_net(model_name="EfficientNetB4",
                        inp_dim=(256, 256, 3),
                        include_top=False):
    base = getattr(efn, model_name)(input_shape=inp_dim,
                                    weights='imagenet',
                                    include_top=include_top)
    inp = base.inputs
    x = base.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(32, activation='silu')(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inp, outputs=x, name=model_name)
    return model


def pfbeta_tf(labels, preds, beta=1):
    preds = tf.clip_by_value(preds, 0, 1)
    y_true_count = tf.reduce_sum(labels)
    ctp = tf.reduce_sum(preds[labels == 1])
    cfp = tf.reduce_sum(preds[labels == 0])
    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if c_precision > 0 and c_recall > 0:
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0.0


def create_model(hyperparams):
    input_dim = (*(hyperparams['input_size'],) * 2, 3)
    if hyperparams['model'] == 'EfficientNet':
        model = build_efficient_net(inp_dim=input_dim)
    else:
        model = build_test_model(inp_dim=input_dim)
    metrics = [pfbeta_tf, tf.keras.metrics.AUC()]
    optimizer = get_optimizer(hyperparams)
    loss_function = get_loss_function(hyperparams)
    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=metrics)
    return model


def get_optimizer(hyperparams):
    optimizer = hyperparams['optimizer']
    if optimizer == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=hyperparams['learning_rate'])
    else:
        raise ValueError(f'Unrecognized optimizer: {optimizer}')


def get_loss_function(hyperparams):
    loss_function = hyperparams['loss']
    if loss_function == 'binary_crossentropy':
        return tf.keras.losses.BinaryCrossentropy()
    else:
        raise ValueError(f'Unrecognized loss function: {loss_function}')


def save_model(model, dir_models, name, ):
    model_path = dir_models + name
    print(f'Saving model to: {model_path}')
    output_file = Path(model_path)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    model.save(model_path)


def load_model(path):
    name = os.path.basename(path)
    custom_metric = {"pfbeta_tf": pfbeta_tf}
    model = tf.keras.models.load_model(path, custom_objects=custom_metric)
    print(f'Loading model: {name}')
    return model
