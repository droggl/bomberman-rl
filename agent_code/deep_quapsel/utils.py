from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, Flatten
import tensorflow as tf

import agent_code.deep_quapsel.dql_params as params


def create_dql_model():
    model = Sequential(
        [
            Conv2D(32, (4,4), input_shape=params.FEATURE_SHAPE),
            Activation('relu'),

            Conv2D(64, (3,3)),
            Activation('relu'),

            Flatten(),
            Dense(128, activation='relu'),
            Dense(6, activation='linear') # output layer with action index as output
        ]
    )

    model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
    model.trainable = True
    return model
