import efficientnet.tfkeras as efn
import tensorflow as tf


def build_test_model(inp_dim=(256, 256, 3)):
    inputs = tf.keras.layers.Input(inp_dim)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', strides=4)(inputs)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model


def build_efficient_net(model_name="EfficientNetB4",
                        inp_dim=(256, 256),
                        include_top=False):
    base = getattr(efn, model_name)(input_shape=(*inp_dim, 3),
                                    weights='imagenet',
                                    include_top=include_top)
    inp = base.inputs
    x = base.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)  # use GAP to get pooling result form conv outputs
    x = tf.keras.layers.Dense(32, activation='silu')(x)  # use activation to apply non-linearity
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)  # use sigmoid to convert predictions to [0-1]
    model = tf.keras.Model(inputs=inp, outputs=x)
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
