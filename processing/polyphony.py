import numpy as np
from music21 import stream, note, chord, tempo, pitch
from collections import defaultdict
import pypianoroll as ppr 
import matplotlib.pyplot as plt
import pypianoroll as ppr 
import numpy as np
import os
from random import random
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Flatten, Input, Bidirectional, TimeDistributed, Activation, Concatenate, Embedding, MaxPooling1D, CategoryEncoding, Conv1D, Dropout, AveragePooling1D
from tensorflow.keras.utils import pad_sequences, timeseries_dataset_from_array, split_dataset, to_categorical
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from keras_nlp.layers import TransformerEncoder, TransformerDecoder


import calendar

import time
import shutil
from sklearn.model_selection import train_test_split

INSTRUMENTS = 1

def get_pitch_vocab(oov_num=128):
    # Create a list to store all pitches corresponding to MIDI note numbers
    all_pitches = defaultdict(lambda : oov_num)

    # Iterate through MIDI note numbers (0 to 127) and create corresponding pitches
    for note_number in range(oov_num):
        p = pitch.Pitch()
        p.midi = note_number
        all_pitches[str(p)] = note_number #.append(str(p))
    
    return all_pitches

def attach_chords_vocab(sampled_chords, pitch_vocab):
    sampled_chords = list(set(sampled_chords))
    oov_num = pitch_vocab['last'] + len(sampled_chords)
    all_chords = defaultdict(lambda : oov_num)


    # Iterate through MIDI note numbers (0 to 127) and create corresponding pitches
    for chord_number in range(pitch_vocab["last"], oov_num):
        all_chords[sampled_chords[chord_number-pitch_vocab['last']]] = chord_number #.append(str(p))

    all_chords.update(pitch_vocab)
    del all_chords['last']
    return all_chords


def pitch_to_vocab_id(pitches, vocab):
    return np.vectorize(lambda x: vocab[x])(pitches)

def reverse_vocab(vocab):
    last_id = vocab['last']
    del vocab['last']
    rvocab = {v:k for k,v in vocab.items()}
    rvocab.update({last_id:'R'})
    return rvocab


def vocab_id_to_pitch(vocab_ids, rvocab):
    return np.vectorize(lambda x: rvocab[x])(vocab_ids)


def store_batched_dataset(dataset, dataset_name, dataset_type='train'):

    for batch_id, batch in enumerate(dataset):
        inputs, outputs = batch 
        try:
            np.save(f'lpd_5_batched/{dataset_type}_inputs/{dataset_name}_{batch_id}.npy', inputs)
            np.save(f'lpd_5_batched/{dataset_type}_outputs/{dataset_name}_{batch_id}.npy', outputs)
        except Exception as E:
            print(E)



def make_dataset(track, resolution, batch_size, prune_rest_note_percent, encoder_decoder, input_sequence_len, output_sequence_len):
    
    # Argmax results, one pitch at a time step for an instrument, ignores chords
    track = np.expand_dims(parse_track(track, resolution), axis=-1)

    try:
        if encoder_decoder:
            input_track = track[:-output_sequence_len]
            output_track = track[input_sequence_len:]
        else:
            input_track = track[:-1]
            output_track = track[1:]
        

        input_dataset = timeseries_dataset_from_array(input_track, None, sequence_length=input_sequence_len, sequence_stride=1, batch_size=batch_size)
        output_dataset = timeseries_dataset_from_array(output_track, None, sequence_length=output_sequence_len, sequence_stride=1, batch_size=batch_size)

        
        dataset = zip(input_dataset, output_dataset)
            
    except Exception as E:
        dataset = None
        pass
    
    return dataset 


def sample_dataset(dir, nsamples,  train_size=0.8, val_size=0.2, input_sequence_len=2400, output_sequence_len=None, resolution=24, prune_rest_note_percent=0.3, batch_size=64, encoder_decoder=False):

    samples = os.listdir(dir)[:nsamples]

    train_samples, test_samples = train_test_split(samples, train_size=train_size, shuffle=True)
    train_samples, val_samples = train_test_split(train_samples, test_size=val_size, shuffle=False)
    
    try:
        shutil.rmtree('lpd_5_batched/train_inputs')
        shutil.rmtree('lpd_5_batched/train_outputs')
    except:
        pass

    try:
        shutil.rmtree('lpd_5_batched/val_inputs')
        shutil.rmtree('lpd_5_batched/val_outputs')
    except:
        pass 
    
    try:
        shutil.rmtree('lpd_5_batched/test_inputs')
        shutil.rmtree('lpd_5_batched/test_outputs')
    except:
        pass

    os.makedirs(f'lpd_5_batched/train_inputs')
    os.makedirs(f'lpd_5_batched/train_outputs')

    os.makedirs(f'lpd_5_batched/val_inputs')
    os.makedirs(f'lpd_5_batched/val_outputs')

    os.makedirs(f'lpd_5_batched/test_inputs')
    os.makedirs(f'lpd_5_batched/test_outputs')


    #gmt = time.gmtime()
    #ts = calendar.timegm(gmt)
    for dataset_type in ['train', 'val', 'test']:
        if dataset_type == 'train':
            samples = train_samples
        elif dataset_type == 'val':
            samples = val_samples
        elif dataset_type == 'test':
            samples = test_samples
        

        for trackid in tqdm(range(len(samples)),desc=f'Preparing {dataset_type} dataset...'):
            track = ppr.load(os.path.join(dir, samples[trackid]))
            dataset = make_dataset(track, resolution, batch_size, prune_rest_note_percent, encoder_decoder, input_sequence_len, output_sequence_len)
            if dataset:
                store_batched_dataset(dataset, dataset_name=f'Pr{os.path.basename(dir)}-Tr{trackid}', dataset_type=dataset_type)
           
        
    return ('lpd_5_batched/train_inputs/', 'lpd_5_batched/train_outputs/'), ('lpd_5_batched/val_inputs/', 'lpd_5_batched/val_outputs/'),  ('lpd_5_batched/test_inputs/', 'lpd_5_batched/test_outputs/')



def sample_track(dir, nsamples, input_sequence_len, resolution):
    samples = np.array(os.listdir(dir))
    track_ids = np.random.choice(len(samples), replace=False, size=(nsamples,))
    samples = samples[track_ids]

    tracks = []
    for trackid in tqdm(range(0, nsamples, 1), desc='Sampling tracks...'):
        track = ppr.load(os.path.join(dir, samples[trackid]))
        track = np.expand_dims(parse_track(track, resolution), axis=-1)

        input_track = track[:input_sequence_len]
        output_track = track[input_sequence_len:]

        tracks += [(input_track, output_track)]
        
    return tracks

def parse_track(track, resolution):

    global vocab, rvocab

    chord_info = set()
    parsed_track = []
    
    track = track.binarize().set_resolution(resolution).pad_to_same()
    ppr.write('temp_save.mid', track)

    # Load the MIDI file using music21
    file = 'temp_save.mid'
    midi = converter.parse(file)
    parts = midi.parts[1:2] if INSTRUMENTS == 1 else midi.parts


    # Iterate over each track in the MIDI file
    for part in parts:
     
        instrument_track = []
        # Traverse the MIDI elements and extract notes and chords
        notes_to_parse = part.recurse()
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                instrument_track.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                chordd = '.'.join(str(n) for n in element.pitches)
                chord_info.add(chordd)
                instrument_track.append(chordd)
            elif isinstance(element, note.Rest):
                instrument_track.append('R')

        parsed_track.append(instrument_track)


    vocab = attach_chords_vocab(chord_info, get_pitch_vocab())
    rvocab = reverse_vocab(vocab)
    return pitch_to_vocab_id(parsed_track, vocab)


format_targets = lambda y: tf.unstack(tf.experimental.numpy.moveaxis(y, (0, 1, 2), (1, 2, 0)))[0] if INSTRUMENTS == 1 else tuple(tf.unstack(tf.experimental.numpy.moveaxis(y, (0, 1, 2), (1, 2, 0))))

def load_music_batches(input_dir, output_dir, encoder_decoder=True):

    while 1:
        for inp_batch, output_batch in zip(os.listdir(input_dir), os.listdir(output_dir)):
            
            try:
                inputs, targets = np.load(os.path.join(input_dir, inp_batch)), np.load(os.path.join(output_dir, output_batch))
                
                if encoder_decoder:
                    prompt_inputs = np.concatenate([inputs[:, -2:-1], targets[:, :-1]], axis=1)#none, 2400, 5
                    yield [inputs, prompt_inputs], format_targets(targets)
                else:
                    yield inputs, format_targets(targets)
            except Exception as E:
                continue


import subprocess


from random import random, randint


def top_p_sampling(probabilities, p):
    sorted_indices = np.argsort(probabilities)[::-1]  # Sort probabilities in descending order
    cumulative_probs = np.cumsum(probabilities[sorted_indices])  # Compute cumulative probabilities
    if np.any(cumulative_probs <= p):
        selected_indices = sorted_indices[cumulative_probs <= p]  # Select indices where cumulative probability <= p
    else:
        # If none of the cumulative probabilities exceed p, select the maximum probability
        selected_indices = np.array([np.argmax(probabilities)])
    return selected_indices




def compose_music(music_model, cue=None, topn=6, top_p=None, print_gen=False, encoder_decoder=False, slide_cue_after=100):  #cue-shape : (cue_len, 5)
    

    
    #cue = np.concatenate([np.zeros(cue.shape[:-1] + (1,)), cue], axis=-1)
    #cue[np.any(cue, axis=-1)==False, 0] = 1 
    # Argmax results, one pitch at a time step for an instrument, ignores chords
    #cue = cue.argmax(axis=-1) #(800, 5)

    cue = np.expand_dims(cue, axis=0) #(1, time_steps, instruments)

    composition = [cue[:, -1]]      #List[(1, 5,)]
    gen = 1

    start_pcomp = 0
    while True:
 
        if print_gen:
            print("Generation : ", gen)
        gen += 1


        composition_arr = np.concatenate(composition)
        if composition_arr.shape[0]%(slide_cue_after) == 0:
            start_pcomp += slide_cue_after-1
            cue = np.expand_dims(composition_arr[-cue.shape[1]:], axis=0)
            pcomp = pcomp[:, -2:-1, :]
            
        pcomp = np.expand_dims(composition_arr[start_pcomp:start_pcomp+slide_cue_after],axis=0)

      
        if encoder_decoder:
            pred = np.concatenate(music_model( [cue, pcomp] ))  #(5, 1, 129)
        else:
            pred = np.concatenate(music_model(pcomp))  #(5, 1, 129)

        if INSTRUMENTS == 1:
            pred = np.expand_dims(pred, axis=0)


  
        
        preds = []
        for instrument in range(INSTRUMENTS):
           
            probs = np.exp(pred[instrument, -1])
            probs = probs/np.sum(probs)

            if top_p:
                selected_indices = top_p_sampling(probs, top_p)
                new_probs = np.zeros(probs.shape)
                new_probs[selected_indices] = probs[selected_indices]
            
            else:
                exclude_pred = np.argsort(probs)[:-topn]
                probs[exclude_pred] = 0.
                new_probs = probs
            
            new_probs = new_probs/np.sum(new_probs)
            preds += [np.random.choice(vocab['last'], (1,), p=new_probs)]

        preds = np.array(preds)
        currcomp = preds.T #(1, 5)

        composition += [currcomp]
    
        yield np.concatenate(composition)
    

def get_avg_tempo(dir='lpd_5/lpd_5_full/0', topn=1000):
    samples = os.listdir(dir)[:topn]
    tempo = 0.
    count = 0
    for sample in samples:
        with np.load(os.path.join(dir, sample)) as data:
            tempo += np.sum(data['tempo'])
            count += data['tempo'].shape[0]
    return tempo/count

import numpy as np

from music21 import stream, note, midi

from music21 import stream, note, instrument

def pitches_to_midi(pitches_list, instrument_names, resolution, output_file='output.mid'):
    # Create a Score object to hold all instruments
    score = stream.Score()

    # Create a Part object for each instrument
    for i, (pitches, instr_name) in enumerate(zip(pitches_list, instrument_names)):
        # Create an Instrument object for the instrument
        instr = instrument.Instrument()
        instr.partName = instr_name  # Set instrument name
        # Optionally, you can set other attributes such as instrument family, MIDI program number, etc.
        # For example:
        # instr.instrumentName = 'Acoustic Grand Piano'  # Set instrument name (General MIDI)
        # instr.midiProgram = 1  # Set MIDI program number (General MIDI)

        # Create a Stream object for the instrument
        stream_obj = stream.Part()
        stream_obj.append(instr)  # Append the Instrument object to the Stream

        # Add Note objects for each pitch to the Stream
        for pitch in pitches:
            n = note.Note(pitch)
            stream_obj.append(n)

        # Append the Stream object to the Score object
        score.insert(i, stream_obj)

    # Save the Score object as a MIDI file
    score.write('midi', fp=output_file, midiScale=resolution)
    print(f"MIDI file saved as: {output_file}")



def make_midi(composition, tempo=120, resolution=8, output_file=None):


    tracks = []
    tempo = np.full(composition.shape[0], tempo)  #get the tempo

    
    track_data = {0 : ['Drums', 0], 1: ['Piano', 0], 2: ['Guitar', 24], 3:['Bass', 32], 4:['Strings', 48]} #{"is_drum": false, "program": 0, "name": "Piano"}, "0": {"is_drum": true, "program": 0, "name": "Drums"}, "3": {"is_drum": false, "program": 32, "name": "Bass"}, "2": {"is_drum": false, "program": 24, "name": "Guitar"}, "4": {"is_drum": false, "program": 48, "name": "Strings"}, "beat_resolution": 24}'
    if INSTRUMENTS == 1:
        track_data = {0:track_data[1]}
        
    # Create a Track object for each track in the multitrack representation
    piano_rolls = []
    instrument_names = []

    for i, track_name_program in track_data.items():
        
        track_name, program = track_name_program


        piano_roll = vocab_id_to_pitch(composition[:, i], rvocab)#CategoryEncoding(PITCHES, output_mode='one_hot')(composition[:, i]).numpy()[:, 1:]

        piano_rolls += [piano_roll]
        instrument_names += [track_name]

    pitches_to_midi(piano_rolls, instrument_names, output_file, resolution)
    return output_file


from collections import Counter as C
from copy import deepcopy
def get_class_weights(source_loader, steps=200, encoder_decoder=True):


    default = {k:0 for k in range(PITCHES)}
    class_weights = [default for _ in range(INSTRUMENTS)]
    for _ in range(steps):
        x, _ = next(source_loader)  #(batch_size, time_steps, 5)
        
        if encoder_decoder:
            x = x[0]

       
        for i in range(INSTRUMENTS):
            pcwd = class_weights[i]
            ncwd = C(x[:, :, i].ravel().tolist())
            for k in pcwd.keys():
                pcwd[k] += ncwd[k]

            class_weights[i] = pcwd

            #class_weights[i][0] += np.sum(1-(x[:, :, i]))
            #class_weights[i][1] += np.sum(x[:, :, i])
    


    for j in range(INSTRUMENTS):
        cwd = class_weights[j]
        total = sum(cwd.values())
        keys = deepcopy(list(cwd.keys()))
        for k in keys: 
            cwd[k] = 1 - (cwd[k])/total

        class_weights[j] = cwd
    
    if INSTRUMENTS == 1:
        return class_weights[0]
    else:
        return class_weights







