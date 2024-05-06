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


def transformer_encoder_decoder(music_dim=PITCHES+len(TIME_VOCAB), instruments=INSTRUMENTS, ignore_last_class=True):

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
    

    losses = [tf.keras.losses.SparseCategoricalCrossentropy(ignore_class=music_dim-1 if ignore_last_class else music_dim+10, from_logits=True)]*instruments
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
from tensorflow.python.framework import ops

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
       
        self.kl_loss_tracker = tf.metrics.Mean(name="kl_loss")

        self.decoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(units=8*32*128, activation=tf.nn.relu),  # Adjust units to match desired shape
            layers.Reshape(target_shape=(8, 32, 128)),  # Adjust target shape
            layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding='SAME', activation='relu'),
            layers.Conv2DTranspose(filters=32, kernel_size=3, strides=(2, 2), padding='SAME', activation='relu'),
            layers.Conv2DTranspose(filters=instruments, kernel_size=3, strides=(1, 1), padding='SAME'),
        ])
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z):
        return self.decoder(z)

    def call(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mean, logvar
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        x, y = data[0], data[1]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)
           
            reconstruction_loss = ops.mean(
                ops.sum(
                    tf.keras.losses.mean_sqaured_error(x, reconstruction),
                    axis=(1, 2),
                )
            )

          
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    

# Define the loss function
def vae_loss(x_original, preds):
    x_reconstructed, mean, logvar = preds
    reconstruction_loss = tf.reduce_mean(losses.mean_squared_error(x_original, x_reconstructed))
    kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mean) - tf.exp(logvar))
    return reconstruction_loss + kl_loss

def variational_autoencoder(seq_len=32, latent_dim=20, instruments=INSTRUMENTS):
    #Xinp = Input((seq_len, 128, instruments))
    model = VAE(seq_len=seq_len, latent_dim=latent_dim, instruments=instruments)
    
    #model = Model(Xinp, X)
    model.compile(optimizer='adam', loss=lambda x, y: 6)
    return model

def transformer_decoder(latent_dim):
    Zinp = Input((None, latent_dim*2))
    Z = TransformerDecoder(intermediate_dim=400, num_heads=8)(Zinp)
    Z = TransformerDecoder(intermediate_dim=400, num_heads=8)(Z) 
    Z = TransformerDecoder(intermediate_dim=400, num_heads=8)(Z) 
    Z = TimeDistributed(Dense(latent_dim*2, activation='linear'))(Z)
    
    model = Model(Zinp, Z)
    model.compile(Adam(1e-3), loss='mse', metrics=['mae'])
    return model

def transformer_decoder_variational_autoencoder(vae_model, transformer_decoder_model, instruments=INSTRUMENTS):
    seq_len = vae_model.input.shape[1]
    Xinp = Input((None, seq_len, instruments))
    X1, X2 = TimeDistributed(vae_model.encode)(Xinp)
    X = TimeDistributed(Concatenate())([X1, X2])
    X = transformer_decoder_model(X)
    X = TimeDistributed(vae_model.reparameterize)(X[:X.shape[1]//2], X[X.shape[1]//2 :])
    X = TimeDistributed(vae_model.decode)(X)

    model = Model(Xinp, X)  #meant for inference only
    return model




    