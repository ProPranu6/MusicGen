import pypianoroll as ppr 
import numpy as np
import os
from random import random
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Flatten, Input, Bidirectional, TimeDistributed, Activation, Concatenate, Embedding, MaxPooling1D, CategoryEncoding, Conv1D, Dropout, AveragePooling1D
from tensorflow.keras.utils import pad_sequences, timeseries_dataset_from_array, split_dataset, to_categorical, plot_model
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from keras_nlp.layers import TransformerEncoder, TransformerDecoder


import calendar

import time
import shutil
from sklearn.model_selection import train_test_split
from processing.monophony import PITCHES, INSTRUMENTS, TIME_VOCAB


def recurrent_encoder_decoder(pitches=PITCHES, instruments=INSTRUMENTS, time_vocabs=len(TIME_VOCAB)):
    """
    (Multi-instrument-nochord) -> (Embedding of size 30) -> (Contextual Average embedding) -> (Encoder-Decoder) -> (Multi-instrument softmax outputs)
    """
    Xinp = Input((None, instruments))
    Xpromptinp = Input((None, instruments))
    note_to_vec = Sequential([Embedding(pitches+time_vocabs, 30), 
                              AveragePooling1D(INSTRUMENTS),
                              Flatten()
    ])

    X = TimeDistributed(note_to_vec)(Xinp)
    X = Bidirectional(LSTM(256, return_sequences=True))(X)
    X = Bidirectional(LSTM(256, return_sequences=True))(X)
   
    _, *internal_state = LSTM(512, return_state=True)(X)

    Y = TimeDistributed(note_to_vec)(Xpromptinp)
    Y = LSTM(512, return_sequences=True)(Y, initial_state=internal_state)
    Y = TimeDistributed(Dense(512, 'relu'))(Y)
    

    Out = []
    for instrument in range(instruments):
        Out += [TimeDistributed(Dense(pitches+time_vocabs+1), name=f'instrument_{instrument+1}')(Y)]
    

    losses = [tf.keras.losses.SparseCategoricalCrossentropy(ignore_class=pitches+time_vocabs, from_logits=True)]*instruments
    if instruments == 1:
        Out = Out[0]
        losses = losses[0]
    
    In = [Xinp, Xpromptinp]
    model = Model(In, Out)
    model.compile(Adam(1e-3), loss=losses, metrics=['accuracy'])
    return model

def recurrent(pitches=PITCHES, instruments=INSTRUMENTS, time_vocabs=len(TIME_VOCAB)):
    Xinp = Input((None, instruments))
    X = LSTM(200, return_sequences=True)(Xinp)
    X = LSTM(100, return_sequences=True)(X)
    X = TimeDistributed(Dense(50, 'relu'))(X)
    
    Out = []
    for instrument in range(instruments):
        Out += [TimeDistributed(Dense(pitches+time_vocabs+1), name=f'instrument_{instrument+1}')(X)]
    
    losses = [tf.keras.losses.SparseCategoricalCrossentropy(ignore_class=pitches+time_vocabs, from_logits=True)]*instruments
    if instruments == 1:
        Out = Out[0]
        losses = losses[0]
    
    In = Xinp 
    model = Model(In, Out)
    model.compile(Adam(1e-3), loss=losses, metrics=['accuracy'])
    return model


def transformer_encoder_decoder(music_dim=PITCHES+len(TIME_VOCAB), instruments=INSTRUMENTS):

    """
    (Multi-instrument-nochord) -> (Embedding of size 100) -> (Contextual Average embedding) -> (Encoder-Decoder Transformer) -> (Multi-instrument softmax outputs)
    """

    Xinp = Input((None, instruments))
    Xpromptinp = Input((None, instruments))

    note_to_vec = Sequential([Embedding(music_dim, 100), 
                              AveragePooling1D(INSTRUMENTS),
                              Flatten()
    ], name='note_to_vec')

    X = TimeDistributed(note_to_vec)(Xinp)
    X = TransformerEncoder(intermediate_dim=400, num_heads=8)(X)
    X = TransformerEncoder(intermediate_dim=400, num_heads=8)(X)

    Y = TimeDistributed(note_to_vec)(Xpromptinp)
    Y = TransformerDecoder(intermediate_dim=400, num_heads=8)(Y)
    Y = TransformerDecoder(intermediate_dim=400, num_heads=8)(Y, X) 
    Y = TransformerDecoder(intermediate_dim=400, num_heads=8)(Y) 

    Out = []
    for instrument in range(instruments):
        Out += [TimeDistributed(Dense(music_dim), name=f'instrument_{instrument+1}')(Y)]
    

    losses = [tf.keras.losses.SparseCategoricalCrossentropy(ignore_class=music_dim-1, from_logits=True)]*instruments
    if instruments == 1:
        Out = Out[0]
        losses = losses[0]
    
    In = [Xinp, Xpromptinp]
    model = Model(In, Out)
    model.compile(Adam(1e-3), loss=losses, metrics=['accuracy'])
    return model


def transformer_cnn_encoder_decoder(music_dim=PITCHES+len(TIME_VOCAB), instruments=INSTRUMENTS):
    """
    (Multi-instrument-nochord) -> (Embedding of size 30) -> (Contextual Convoluted[CNN] embedding) -> (Encoder-Decoder Transformer) -> (Multi-instrument softmax outputs)
    """
    assert instruments == 5, "can only be modeled for multi-instrument(5) music generation"

    Xinp = Input((None, instruments))
    Xpromptinp = Input((None, instruments))

    note_to_vec = Sequential([Embedding(music_dim, 100),
                              Conv1D(50, 2, padding='same'),    
                              Conv1D(10, 2, padding='same'), 
                              AveragePooling1D(instruments),
                              Flatten()
    ], name='note_to_vec')

    X = TimeDistributed(note_to_vec)(Xinp)
    X = TransformerEncoder(intermediate_dim=400, num_heads=8)(X)
    X = TransformerEncoder(intermediate_dim=400, num_heads=8)(X)

    Y = TimeDistributed(note_to_vec)(Xpromptinp)
    Y = TransformerDecoder(intermediate_dim=400, num_heads=8)(Y)
    Y = TransformerDecoder(intermediate_dim=400, num_heads=8)(Y, X) 
    Y = TransformerDecoder(intermediate_dim=400, num_heads=8)(Y) 

    Out = []
    for instrument in range(instruments):
        Out += [TimeDistributed(Dense(music_dim), name=f'instrument_{instrument+1}')(Y)]
    

    losses = [tf.keras.losses.SparseCategoricalCrossentropy(ignore_class=music_dim-1, from_logits=True)]*instruments
    if instruments == 1:
        Out = Out[0]
        losses = losses[0]
    
    In = [Xinp, Xpromptinp]
    model = Model(In, Out)
    model.compile(Adam(1e-3), loss=losses, metrics=['accuracy'])
    return model



import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses, metrics, Model

class VAE(Model):
    def __init__(self, seq_len, latent_dim, instruments):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(seq_len, 128, instruments)),
            layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            layers.Flatten(),
            layers.Dense(latent_dim + latent_dim),
        ])

        self.decoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(units=7*7*32, activation=tf.nn.relu),
            layers.Reshape(target_shape=(7, 7, 32)),
            layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding='SAME', activation='relu'),
            layers.Conv2DTranspose(filters=32, kernel_size=3, strides=(2, 2), padding='SAME', activation='relu'),
            layers.Conv2DTranspose(filters=instruments, kernel_size=3, strides=(1, 1), padding='SAME'),
        ])

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z):
        return self.decoder(z)

    def call(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mean, logvar

# Define the loss function
def vae_loss(x_original, x_reconstructed, mean, logvar):
    reconstruction_loss = tf.reduce_mean(losses.mean_squared_error(x_original, x_reconstructed))
    kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mean) - tf.exp(logvar))
    return reconstruction_loss + kl_loss

def variational_autoencoder(seq_len=32, music_dim=PITCHES+len(TIME_VOCAB), instruments=INSTRUMENTS):
    model = VAE(seq_len=seq_len, latent_dim=int(0.3*music_dim), instruments=instruments)
    model.compile(optimizer='adam', loss=vae_loss)
    return model

