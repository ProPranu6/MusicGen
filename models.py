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
    Xinp = Input((None, instruments))
    Xpromptinp = Input((None, instruments))
    note_to_vec = Sequential([Embedding(pitches+instruments+time_vocabs, 30), 
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


def transformer_encoder_decoder(pitches=PITCHES, instruments=INSTRUMENTS, time_vocabs=len(TIME_VOCAB), encoder_len=100, decoder_len=200):

    Xinp = Input((encoder_len, instruments))
    Xpromptinp = Input((decoder_len, instruments))

    note_to_vec = Sequential([Embedding(pitches+instruments+time_vocabs, 100), 
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
        Out += [TimeDistributed(Dense(pitches+time_vocabs+1), name=f'instrument_{instrument+1}')(Y)]
    

    losses = [tf.keras.losses.SparseCategoricalCrossentropy(ignore_class=pitches+time_vocabs, from_logits=True)]*instruments
    if instruments == 1:
        Out = Out[0]
        losses = losses[0]
    
    In = [Xinp, Xpromptinp]
    model = Model(In, Out)
    model.compile(Adam(1e-3), loss=losses, metrics=['accuracy'])
    return model





