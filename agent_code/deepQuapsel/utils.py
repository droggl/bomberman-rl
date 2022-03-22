import os
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
import tensorflow as tf

import agent_code.deepQuapsel.dql_params as params


class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._train_dir = self.log_dir
        self._train_step = self.step
        self._log_write_dir = self.log_dir

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        pass
        # self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        pass
        # self._write_logs(stats, self.step)


def create_dql_model():
    model = Sequential(
        [
            Conv2D(32, (4,4), input_shape=params.FEATURE_SHAPE),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.2), # dropout 20%

            Conv2D(32, (4,4)),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.2),

            Flatten(),  # converts 3D feature maps to 1D feature vectors
            Dense(64),
            Dense(6, activation='linear') # output layer with action index as output
        ]
    )

    model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])
    return model
