import tensorflow as tf
from tensorflow.keras import layers, models, backend as K


def build_base_network(input_shape):
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=input_shape),
        layers.Dropout(0.1),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(128, activation='relu')
    ])
    return model


def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))


def contrastive_loss(y_true, y_pred, margin=1):
    y_true = tf.cast(y_true, tf.float32)
    square_pred = K.square(y_pred)
    margin_square = K.square(K.max(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def build_siamese_model(input_shape):
    base_network = build_base_network(input_shape)

    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    # Riga aggiornata con output_shape esplicito:
    distance = layers.Lambda(euclidean_distance, output_shape=lambda shapes: (shapes[0][0], 1))(
        [processed_a, processed_b])

    model = models.Model(inputs=[input_a, input_b], outputs=distance)
    return model