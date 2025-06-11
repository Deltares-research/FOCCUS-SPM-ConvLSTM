# -*- coding: utf-8 -*-
"""
Module for defining model acrhitecture and warmup, for SPM Prediction Pipeline. Adapted from Senyang Li's work (unpublished, contact for information).

Author: Beau van Koert, Edits by L. Beyaard
Date: June 2025
"""

# Import libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np


class WarmUpCosineDecayScheduler(keras.callbacks.Callback):
    """
    Custom callback for Keras models to adjust the learning rate using a cosine decay schedule with warmup, worked best.

    Attributes:
        T (int): The period of the half cosine cycle.
        learning_rate_base (float): The base learning rate before any adjustments.
        total_steps (int): The total number of training steps.
        global_step_init (int): The initial global step， for a counter indicating the current step.
        warmup_learning_rate (float): The starting learning rate for the warmup phase.
        warmup_steps (int): The number of steps to linearly increase the learning rate during warmup.
        hold_base_rate_steps (int): The number of steps to hold the base learning rate constant before decay starts.
        verbose (int): Verbosity mode, 0 or 1.
    """
    def __init__(self, T, learning_rate_base, total_steps, global_step_init=0, warmup_learning_rate=0.0, warmup_steps=0, hold_base_rate_steps=0, verbose=0):
        super(WarmUpCosineDecayScheduler, self).__init__()
        self.T = T
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.verbose = verbose
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        """
        Updates the global step count at the end of each batch.
        """
        self.global_step += 1
        lr = self.model.optimizer.learning_rate.numpy()                         # Updated: Replaced K.get_value
        self.learning_rates.append(lr)                                          # Should work similar to K.get_value

    def on_batch_begin(self, batch, logs=None):
        """
        Adjusts the learning rate at the beginning of each batch based on the cosine decay schedule with warmup.
        """
        lr = self._cosine_decay_with_warmup()
        self.model.optimizer.learning_rate.assign(lr)                           # Updated: Replace K.set_value
        if self.verbose > 0:
            print(f'\nBatch {self.global_step + 1:05d}: setting learning rate to {lr}.')

    def _cosine_decay_with_warmup(self):
        """
        Calculates the learning rate based on cosine decay with warmup.

        Returns:
            float: The calculated learning rate.
        """
        # Check epoch duration of warmup
        if self.total_steps < self.warmup_steps:
            raise ValueError('total_steps must be larger or equal to warmup_steps.')
        
        # General learning rate
        learning_rate = 0.5 * self.learning_rate_base * (1 + np.cos(np.pi * self.T * (self.global_step - self.warmup_steps - self.hold_base_rate_steps) / float(self.total_steps - self.warmup_steps - self.hold_base_rate_steps)))
        
        # Learning rate for hold base
        if self.hold_base_rate_steps > 0:
            learning_rate = np.where(self.global_step > self.warmup_steps + self.hold_base_rate_steps, learning_rate, self.learning_rate_base)
        
        # Learning rate for warmup
        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError('learning_rate_base must be larger or equal to warmup_learning_rate.')
            
            slope = (self.learning_rate_base - self.warmup_learning_rate) / self.warmup_steps
            warmup_rate = slope * self.global_step + self.warmup_learning_rate
            learning_rate = np.where(self.global_step < self.warmup_steps, warmup_rate, learning_rate)
        
        # Exit scheme
        return np.where(self.global_step > self.total_steps, 0.0, learning_rate)

class LossHistory(keras.callbacks.Callback):
    """
    Custom callback to record training and validation loss history.
    """
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs=None):
        self.val_losses.append(logs.get('val_loss'))

class ReflectionPadding2D(keras.layers.Layer):
    """
    Custom layer for 2D reflection padding: essential to deal with the boundary in the CNN model
    """
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = list(padding)
        self.input_spec = [keras.layers.InputSpec(ndim=5)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        s = list(s)
        s[-2] += 2 * self.padding[0]
        s[-1] += 2 * self.padding[1]
        return tuple(s)

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [0, 0], [0, 0], [h_pad, h_pad], [w_pad, w_pad]], 'REFLECT')

def getModel(input_shape, activation, dropout, num_conv_layers, n_lstm_conv):
    """
    Defines and returns the model architecture, based on model architecture
    settings in config file.

    Parameters:
        input_shape (tuple): Shape of the input data.
        activation (str): Activation function to use.
        dropout (float): Dropout rate (0 means no dropout layers).
        num_conv_layers (int): Total number of ConvLSTM layers (first + intermediate + final).
        n_lstm_conv (int): Number of filters in the first ConvLSTM layer.

    Returns:
        keras.models.Model: Compiled Keras model.
    """
    KERNEL_SIZE_1 = 3                   # Kernel in first ConvLSTM layer
    PADDING_1 = int(KERNEL_SIZE_1 / 2)
    KERNEL_SIZE_2 = 5                   # Kernel in other ConvLSTM layers
    PADDING_2 = int(KERNEL_SIZE_2 / 2)
    SEQ_LEN = input_shape[0]            # LSTM sequence length
    N_LATS = input_shape[2]             # number of latitudes
    N_LONS = input_shape[3]             # number of longitudes
    N_LSTM_CONV = n_lstm_conv           # From config
    N_LSTM_CONV2 = N_LSTM_CONV * 2      # Double the first layer's filters
    DROPOUT_RATE = dropout if dropout > 0 else 0  # Use dropout only if > 0

    # Validate num_conv_layers
    if num_conv_layers < 2:
        raise ValueError("For this model setup, num_conv_layers must be at least 2 (first layer + final layer)")

    model = keras.Sequential()                  # Define model type
    model.add(keras.Input(shape=input_shape))   # Input shape

    # First ConvLSTM layer (always included)
    model.add(ReflectionPadding2D(padding=(PADDING_1, PADDING_1)))
    model.add(keras.layers.ConvLSTM2D(
        filters=N_LSTM_CONV,
        kernel_size=KERNEL_SIZE_1,
        data_format="channels_first",
        activation="tanh",
        return_sequences=True,
        padding="valid",
        recurrent_activation="hard_sigmoid"
    ))
    if DROPOUT_RATE > 0:                        # Add dropout layer if dropout rate > 0
        model.add(keras.layers.Dropout(DROPOUT_RATE))

    # Additional ConvLSTM layers based on num_conv_layers
    for i in range(num_conv_layers - 2):        # -2 because first and final layers are added separately
        model.add(ReflectionPadding2D(padding=(PADDING_2, PADDING_2)))
        model.add(keras.layers.ConvLSTM2D(
            filters=N_LSTM_CONV2,
            kernel_size=KERNEL_SIZE_2,
            data_format="channels_first",
            activation="tanh",                  # Bounds the output to [-1, 1]
            return_sequences=True,              # Hidden states can be transfered to the next layer, allowing the model to learn temporal dependencies across the entire input sequence.
            padding="valid",
            recurrent_activation="hard_sigmoid"
        ))
        if DROPOUT_RATE > 0:                    # Add dropout layer if dropout rate > 0
            model.add(keras.layers.Dropout(DROPOUT_RATE))

    # Final ConvLSTM layer (output layer)
    model.add(ReflectionPadding2D(padding=(PADDING_2, PADDING_2)))
    model.add(keras.layers.ConvLSTM2D(
        filters=SEQ_LEN,
        kernel_size=KERNEL_SIZE_2,
        data_format="channels_first",
        activation=None,                # Makes sure the output is not limited to [-1, 1], but to the full (raw) SPM range
        return_sequences=False,         # Ensures the final ConvLSTM collapses the temporal processing into a single output per sample
        padding="valid",
        recurrent_activation="hard_sigmoid"
    ))
    if DROPOUT_RATE > 0:
        model.add(keras.layers.Dropout(DROPOUT_RATE))

    # Activation and reshape layer to match the target output shape
    model.add(keras.layers.Activation(activation))  
    model.add(keras.layers.Reshape((SEQ_LEN, N_LATS, N_LONS)))

    return model
