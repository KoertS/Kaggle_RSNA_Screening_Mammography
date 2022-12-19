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
                        inp_dim=(256, 256),
                        include_top=False):
    base = getattr(efn, model_name)(input_shape=(*inp_dim, 3),
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
    if hyperparams['model'] == 'EfficientNet':
        model = build_efficient_net()
    else:
        model = build_test_model()

    metrics = [pfbeta_tf]
    model.compile(optimizer=hyperparams['optimizer'],
                  loss=hyperparams['loss'],
                  metrics=metrics)
    return model


def save_model(model, dir_models, name, ):
    model_path = dir_models + name
    print(f'Saving model to: {model_path}')
    model.save(model_path)


def load_model(dir_models, name):
    model_path = dir_models + name
    custom_metric = {"pfbeta_tf": pfbeta_tf}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_metric)
    print(f'Loading model from: {model_path}')
    return model
