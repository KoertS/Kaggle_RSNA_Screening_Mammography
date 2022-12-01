import tensorflow as tf


def create_model(inp_dim=(256, 256, 3)):
    inputs = tf.keras.layers.Input(inp_dim)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', strides=4)(inputs)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model
