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
PITCHES = 129
TIME_VOCAB = ['d1', 'd8', 'd16', 'd24', 'd32']

import shutil

def _group_subdirs_contents(from_dir=None, to_dir=None):
   

    global TOTAL_TRACKS_COUNT

    # Get a list of all subdirectories
    subdirs = os.listdir(from_dir)
    
    TOTAL_TRACKS_COUNT = 0
    # Iterate over the subdirectories
    for subdir in subdirs:

        # Get a list of all files in the subdirectory
        if os.path.isdir(os.path.join(from_dir, subdir)):
            files = os.listdir(os.path.join(from_dir, subdir))

            # Iterate over the files
            for file in files:
                # Move the file to the main directory
                shutil.move(os.path.join(from_dir, subdir, file), to_dir)
                TOTAL_TRACKS_COUNT += 1

    print("Total Tracks: ", TOTAL_TRACKS_COUNT)
    return to_dir

def resample(empties_boolean, prune_percent=0.7):
    resampled_inclusion_beats = [ ]

    for isempty in empties_boolean:
        include = True
        if isempty :
            if random() < prune_percent:  #70% pruning of empty beats
                include = False
    
        resampled_inclusion_beats += [include]

    return resampled_inclusion_beats

def decimal_multiples(num, base=8, max_mul=len(TIME_VOCAB)-1):
    mul_count = {base*k : 0 for k in range(1, max_mul+1)}
    mul_count.update({1:0})
    
    while num:
        if max_mul == 0:
            div = 1
        else:
            div = (max_mul*base)

        
        mul_count[div] = num//div
            
        num = num%(div)
        max_mul -= 1
    return mul_count



def tokenize_time(pitch_tokenized_track, time_vocab=TIME_VOCAB): #'d8'-1bt, 'd16'-2bt, 'd24'-3bt, 'd32'-4bt

    #a a a b b c c c  - a{3}   b{2}     c{3}
    #d d d d d d d d  - d{4}   <stop>   <stop>

    stop_token_id = PITCHES + len(time_vocab)
    instruments_rep = []
    max_len = 0 

    for instrument in range(pitch_tokenized_track.shape[1]):
        time_roll = pitch_tokenized_track[:, instrument]
        prev = time_roll[0]
        start = 1
        count = 1
        instrument_rep = []
        while start < time_roll.shape[0]:
            curr = time_roll[start]
            if curr != prev:
                instrument_rep += [prev]
                mul_count = decimal_multiples(count-count%8, 8, max_mul=len(time_vocab)-1)
                for time_token_num, token_count in mul_count.items():
                    token_id = time_token_num//8 + PITCHES 
                    instrument_rep += [token_id]*token_count
                prev = curr 
                count = 0
            count += 1
            start += 1
        instrument_rep += [prev]
        mul_count = decimal_multiples(count-count%8, 8, max_mul=len(time_vocab)-1)
        for time_token_num, token_count in mul_count.items():
            token_id = time_token_num//8 + PITCHES 
            instrument_rep += [token_id]*token_count
        
        if len(instrument_rep) >= max_len:
            max_len = len(instrument_rep)

        instruments_rep += [instrument_rep]
    
    time_tokenized_track = np.full((max_len, pitch_tokenized_track.shape[1]), stop_token_id)
    for rollid, roll in enumerate(instruments_rep):
        time_tokenized_track[:len(roll), rollid] = roll 
    
    return time_tokenized_track

def detokenize_time(time_tokenized_track, time_vocab=TIME_VOCAB, cutoff_len=None):

    stop_token_id = PITCHES + len(time_vocab)
    instruments_rep = []

    for instrument in range(time_tokenized_track.shape[1]):
        time_roll = time_tokenized_track[:, instrument]
        prev = time_roll[0]
        prev = prev if prev in range(PITCHES) else 0   #default start token
        start = 1
        count = 1
        instrument_rep = []
        while start <time_roll.shape[0]:
            curr = time_roll[start]
            if curr in range(PITCHES, stop_token_id):
                count += int(time_vocab[curr-PITCHES].replace('d', ''))    
            else:
                instrument_rep += [prev]*count 
                prev = curr
                count = 1

                if not(cutoff_len) and curr == stop_token_id:
                    break
                elif curr==stop_token_id:
                    instrument_rep += [prev]
                else:
                    pass 


            start += 1
        
        if curr != stop_token_id:
            instrument_rep += [prev]*count 

        if cutoff_len:
            instrument_rep = instrument_rep[:cutoff_len]

        instruments_rep += [instrument_rep]

    return np.array(instruments_rep).T
        

def store_batched_dataset(dataset, dataset_name, dataset_type='train'):

    for batch_id, batch in enumerate(dataset):
        inputs, outputs = batch 
        try:
            np.save(f'lpd_5_batched/{dataset_type}_inputs/{dataset_name}_{batch_id}.npy', inputs)
            np.save(f'lpd_5_batched/{dataset_type}_outputs/{dataset_name}_{batch_id}.npy', outputs)
        except Exception as E:
            print(E)

def make_dataset(track, resolution, batch_size, prune_rest_note_percent, encoder_decoder, input_sequence_len, output_sequence_len):
    
    track = track.binarize().set_resolution(resolution).stack()
            
    if INSTRUMENTS == 1:
        track = track[1:2] # take only guitar for one instrument


    # Move axis of tracks
    track = np.moveaxis(track, (0, 1, 2), (1, 0, 2)) #(time_steps, 5, 128)

    # Concatenate extra dimension at 0 for empty   
    track = np.concatenate([np.zeros(track.shape[:-1] + (1,)), track], axis=-1)
    track[np.any(track, axis=-1)==False, 0] = 1 


    # Argmax results, one pitch at a time step for an instrument, ignores chords
    track = track.argmax(axis=-1)

    #print("Shape before resampling : ", d.shape)
    
    # Resample from empty beats
    empty_beats = (np.sum(track, axis=1) == 0)
    inclusion_beats = resample(empty_beats, prune_percent=prune_rest_note_percent)
    
    track= track[inclusion_beats]

    if track.shape[0]:
        track = tokenize_time(track)

    #print("Shape after resampling : ", d.shape)
    
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

def sample_dataset(dir, nsamples, train_size=0.8, val_size=0.2, input_sequence_len=2400, output_sequence_len=None, resolution=24, prune_rest_note_percent=0.3, batch_size=64, encoder_decoder=False):


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
        track = ppr.load(os.path.join(dir, samples[trackid])).binarize().set_resolution(resolution).stack()
            
        if INSTRUMENTS == 1:
            track = track[1:2] # take only guitar for one instrument

        # Move axis of tracks
        track = np.moveaxis(track, (0, 1, 2), (1, 0, 2)) #(time_setps, 5, 128)
        
        # Concatenate extra dimension at 0 for empty   
        track = np.concatenate([np.zeros(track.shape[:-1] + (1,)), track], axis=-1)
        track[np.any(track, axis=-1)==False, 0] = 1 

        # Argmax results, one pitch at a time step for an instrument, ignores chords
        track = track.argmax(axis=-1)

        #if tokenize_tracks:
        #    track = tokenize_time(track)

        input_track = track[:input_sequence_len]
        output_track = track[input_sequence_len:]

        tracks += [(input_track, output_track)]
        
    return tracks


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

def midi_to_wav(midi_path, output_wav_path):
    try:
        # Run Timidity++ command to convert MIDI to WAV
        subprocess.run(["timidity", midi_path, "-Ow", "-o", output_wav_path], check=True)
        print("Conversion completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print("Conversion failed.")


def multitrack_to_midi(multitrack, output_path):
    # Check and convert tracks if necessary
    for i, track in enumerate(multitrack.tracks):
        if not isinstance(track, (ppr.BinaryTrack, ppr.StandardTrack)):
            print(f"Converting track {i} to StandardTrack...")
            multitrack.tracks[i] = track.to_pianoroll().to_track()

    # Write the multitrack to a MIDI file
    multitrack.write(output_path)

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

    #if not cue_tokenized :
    #    cue = tokenize_time(cue)

    cue = np.expand_dims(cue, axis=0)

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
        
        #if output_seq_len:
        #    pcomp = np.concatenate([pcomp, np.zeros((pcomp.shape[0], max(0, output_seq_len - pcomp.shape[1]), pcomp.shape[2]))], axis=1) 

      
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
            preds += [np.random.choice(PITCHES+len(TIME_VOCAB)+1, (1,), p=new_probs)]

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

def make_track(composition, tempo=120, composition_tokenized=True):


    tracks = []
    tempo = np.full(composition.shape[0], tempo)  #get the tempo

    
    track_data = {0 : ['Drums', 0], 1: ['Piano', 0], 2: ['Guitar', 24], 3:['Bass', 32], 4:['Strings', 48]} #{"is_drum": false, "program": 0, "name": "Piano"}, "0": {"is_drum": true, "program": 0, "name": "Drums"}, "3": {"is_drum": false, "program": 32, "name": "Bass"}, "2": {"is_drum": false, "program": 24, "name": "Guitar"}, "4": {"is_drum": false, "program": 48, "name": "Strings"}, "beat_resolution": 24}'
    if INSTRUMENTS == 1:
        track_data = {0:track_data[1]}
        
    
    if composition_tokenized:
        composition = detokenize_time(composition, cutoff_len=composition.shape[0])  #time_tokenized_track length
 

    # Create a Track object for each track in the multitrack representation
    for i, track_name_program in track_data.items():
        
        track_name, program = track_name_program

        #if from_model:
        piano_roll = CategoryEncoding(PITCHES, output_mode='one_hot')(composition[:, i]).numpy()[:, 1:]
 
        #else:
            #piano_roll = CategoryEncoding(PITCHES-1, output_mode='one_hot')(composition.argmax(-1)[:, i]).numpy() 
            #piano_roll[composition.argmax(-1)]
            #piano_roll = composition[:, i]

        # Create a Track object without providing the piano_roll argument
        track = ppr.BinaryTrack(name=track_name)
        
        # Assign piano roll data to the Track object
        track.pianoroll = piano_roll  # Assuming piano_roll is a single-track piano roll
        track.program = program  # Specify the program number if necessary
        
        if track_name == 'Drums':
            track.is_drum = True
        # Append the Track object to the list
        tracks.append(track)

    # Create a Multitrack object and assign the tracks to it
    multitrack = ppr.Multitrack(tracks=tracks, tempo=tempo, resolution=8)

    return multitrack


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






