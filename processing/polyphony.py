import numpy as np
import pypianoroll as ppr 
from collections import defaultdict
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import timeseries_dataset_from_array
import shutil
from sklearn.model_selection import train_test_split
from music21 import stream, note, midi, instrument as INS, converter, chord, pitch
import subprocess
from random import random, randint


INSTRUMENTS = 1
MAX_CHORD_LIMIT = 2000

def get_pitch_vocab(vcard=128):
    """
    Creates a dictionary representing the vocabulary of pitches, following the MIDI format.
    
    Args:
    - vcard (int): Vocabulary cardinality, indicating the number of pitches. Defaults to 128.

    Returns:
    - dict: A dictionary where keys are pitch names and values are their corresponding MIDI note numbers.
    """

    all_pitches = {}

    # Iterate through MIDI note numbers (0 to 127) and create corresponding pitches
    for note_number in range(vcard):
        p = pitch.Pitch()
        p.midi = note_number
        all_pitches[str(p)] = note_number

    return all_pitches

def attach_chords_vocab(sampled_chords, pitch_vocab):
    """
    Creates a music vocabulary where pitch IDs from 0 to 127 represent pitches of notes, and all sampled chords 
    from the training set are allotted IDs beyond that range. Unknown pitches or chords are considered as 'R' (rest note), 
    assigned the last ID.

    Args:
    - sampled_chords (list): List of sampled chords from the training set.
    - pitch_vocab (dict): Dictionary representing the pitch vocabulary.

    Returns:
    - dict: A dictionary where keys are chord names and values are their corresponding IDs.
    """

    sampled_chords = list(set(sampled_chords))
    v_card = len(pitch_vocab) + len(sampled_chords)
    all_chords = defaultdict(lambda : v_card - 1)

    # Assign IDs to sampled chords
    for chord_number in range(len(pitch_vocab), v_card - 1):
        all_chords[sampled_chords[chord_number - len(pitch_vocab)]] = chord_number

    # Include pitch vocabulary and assign 'R' (rest note) the last ID
    all_chords.update(pitch_vocab)
    all_chords['R'] = v_card - 1

    return all_chords

def pitch_to_vocab_id(pitches, vocab):
    """
    Converts pitches to their corresponding IDs according to the given vocabulary.

    Args:
    - pitches (list): List of pitches.
    - vocab (dict): Vocabulary mapping pitch names to IDs.

    Returns:
    - np.array: Array of IDs corresponding to the input pitches.
    """

    return np.vectorize(lambda x: vocab[x])(pitches)

def reverse_vocab(vocab):
    """
    Creates a reverse mapping of the vocabulary, mapping IDs to pitch names.

    Args:
    - vocab (dict): Vocabulary mapping pitch names to IDs.

    Returns:
    - dict: A dictionary where keys are IDs and values are corresponding pitch names.
    """

    last_id = vocab['last']
    del vocab['last']
    rvocab = {v: k for k, v in vocab.items()}
    rvocab.update({last_id: 'R'})
    return rvocab

def vocab_id_to_pitch(vocab_ids, rvocab):
    """
    Converts sequences of pitch IDs to their corresponding pitch names using the reverse vocabulary mapping.

    Args:
    - vocab_ids (np.array): Array of pitch IDs.
    - rvocab (dict): Reverse vocabulary mapping IDs to pitch names.

    Returns:
    - np.array: Array of pitch names corresponding to the input IDs.
    """

    return np.vectorize(lambda x: rvocab[x])(vocab_ids)


def store_batched_dataset(dataset, dataset_name, dataset_type='train'):
    """
    Stores batched datasets as .npy files for batchwise loading during training. 
    Each dataset is identified by the dataset name, which contains the name of the original directory 
    from which the MIDI tracks are extracted, followed by the track ID in the original directory and 
    the batch number associated with that track.
    
    Args:
    - dataset (iterable): Iterable containing batches of input-output pairs.
    - dataset_name (str): Name of the dataset.
    - dataset_type (str): Type of the dataset ('train', 'val', or 'test'). Defaults to 'train'.
    """

    for batch_id, batch in enumerate(dataset):
        inputs, outputs = batch 
        try:
            np.save(f'lpd_5_batched/{dataset_type}_inputs/{dataset_name}_{batch_id}.npy', inputs)
            np.save(f'lpd_5_batched/{dataset_type}_outputs/{dataset_name}_{batch_id}.npy', outputs)
        except Exception as E:
            print(E)


def make_dataset(track, resolution, batch_size, encoder_decoder, input_sequence_len, output_sequence_len):
    """
    Takes a track, resolution as input, parses the track to contain note and chord information, 
    and then makes sequences from them packed by batch size using the arguments given and returns a dataset object.
    
    Args:
    - track (obj): Track object containing MIDI data.
    - resolution (int): Resolution of the MIDI data.
    - batch_size (int): Size of each batch.
    - encoder_decoder (bool): Flag indicating whether the architecture is encoder-decoder.
    - input_sequence_len (int): Length of input sequences.
    - output_sequence_len (int): Length of output sequences.

    Returns:
    - iterable: Dataset containing input-output pairs.
    """

    try:
        # Parse track to get note and chord info
        track = parse_track(track, resolution).T
      

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
    return dataset 


def sample_dataset(dir, nsamples,  train_size=0.8, val_size=0.2, input_sequence_len=2400, output_sequence_len=None, resolution=24, batch_size=64, encoder_decoder=False):
    """
    Takes the main directory containing MIDI tracks, and the number of samples from the original directory to use for 
    creating train, validation, and test datasets, following the given input and output sequence length for the 
    encoder-decoder architecture of a model. Creates the datasets and places them in the train, val, and test 
    directories and returns them.

    Args:
    - dir (str): Main directory containing MIDI tracks.
    - nsamples (int): Number of samples from the original directory to use.
    - train_size (float): Proportion of samples to use for training. Defaults to 0.8.
    - val_size (float): Proportion of samples to use for validation. Defaults to 0.2.
    - input_sequence_len (int): Length of input sequences.
    - output_sequence_len (int): Length of output sequences.
    - resolution (int): Resolution of the MIDI data. Defaults to 24.
    - batch_size (int): Size of each batch. Defaults to 64.
    - encoder_decoder (bool): Flag indicating whether the architecture is encoder-decoder. Defaults to False.

    Returns:
    - tuple: Tuple containing paths to train, val, and test directories.
    """

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


    for dataset_type in ['train', 'val', 'test']:
        if dataset_type == 'train':
            samples = train_samples
            learn_vocab(dir, samples, resolution)
        elif dataset_type == 'val':
            samples = val_samples
        elif dataset_type == 'test':
            samples = test_samples
        

        for trackid in tqdm(range(len(samples)),desc=f'Preparing {dataset_type} dataset...'):
            track = ppr.load(os.path.join(dir, samples[trackid]))
            dataset = make_dataset(track, resolution, batch_size, encoder_decoder, input_sequence_len, output_sequence_len)
            if dataset:
                store_batched_dataset(dataset, dataset_name=f'Pr{os.path.basename(dir)}-Tr{trackid}', dataset_type=dataset_type)
           
        
    return ('lpd_5_batched/train_inputs/', 'lpd_5_batched/train_outputs/'), ('lpd_5_batched/val_inputs/', 'lpd_5_batched/val_outputs/'),  ('lpd_5_batched/test_inputs/', 'lpd_5_batched/test_outputs/')


def sample_track(dir, nsamples, input_sequence_len, resolution):
    """
    Creates a set of input-output tuples of sequences following the input sequence length and resolution 
    for tracks taken from the directory, sampling a total of nsamples.
    
    Args:
    - dir (str): Directory containing MIDI tracks.
    - nsamples (int): Number of samples to take.
    - input_sequence_len (int): Length of input sequences.
    - resolution (int): Resolution of the MIDI data.
    
    Returns:
    - list: List of tuples containing input-output sequences.
    """
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


def learn_vocab(dir, samples, resolution, max_chord_limit=MAX_CHORD_LIMIT):
    """
    Includes chords seen in the tracks sampled from the directory `dir`, with the given resolution. 
    Learns as many as `max_chord_limit` chords.

    Args:
    - dir (str): Directory containing MIDI tracks.
    - samples (list): List of MIDI track filenames.
    - resolution (int): Resolution of the MIDI data.
    - max_chord_limit (int): Maximum number of chords to learn. Defaults to the value of `MAX_CHORD_LIMIT`.

    Returns:
    - None
    """
    global vocab, rvocab

    chord_info = set()
    stop_learning = 0
    for trackid in tqdm(range(len(samples)), desc=f'Learning train vocab...'):
        track = ppr.load(os.path.join(dir, samples[trackid])).binarize().set_resolution(resolution).pad_to_same()
        if INSTRUMENTS == 1:
            track = ppr.Multitrack(tracks=track.tracks[1:2])
        ppr.write('temp_save.mid', track)
        # Load the MIDI file using music21
        file = 'temp_save.mid'
        midi = converter.parse(file)
        parts = midi.parts
        # Iterate over each track in the MIDI file
        for part in parts:
            # Traverse the MIDI elements and extract notes and chords
            notes_to_parse = part.recurse()
            for element in notes_to_parse:
                if isinstance(element, chord.Chord):
                    chordd = '.'.join(str(n.step) for n in element.pitches)
                    chord_info.add(chordd)
                    
                    if len(chord_info) == max_chord_limit:
                        stop_learning = 1
                        print(f"Max chord limit: {MAX_CHORD_LIMIT} reached! Stopping learning now")
                        break
            if stop_learning:
                break
        if stop_learning:
            break

    vocab = attach_chords_vocab(chord_info, get_pitch_vocab())
    rvocab = reverse_vocab(vocab)
    return


def parse_track(track, resolution):
    """
    Converts tracks of pitches represented as pianorolls with the given resolution into sequences of ids 
    which mark the chord or note as learned and stored in the vocabulary (`vocab`), and returns the converted tracks.

    Args:
    - track (obj): Track object containing MIDI data.
    - resolution (int): Resolution of the MIDI data.

    Returns:
    - list: List of lists containing sequences of vocabulary IDs.
    """
    parsed_track = []    
    track = track.binarize().set_resolution(resolution).pad_to_same()
    if INSTRUMENTS == 1:
        track = ppr.Multitrack(tracks=track.tracks[1:2])
    
    ppr.write('temp_save.mid', track)

    # Load the MIDI file using music21
    file = 'temp_save.mid'
    midi = converter.parse(file)
    parts = midi.parts

    # Iterate over each track in the MIDI file
    for part in parts:
        instrument_track = []
        # Traverse the MIDI elements and extract notes and chords
        notes_to_parse = part.recurse()
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                instrument_track.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                chordd = '.'.join(str(n.step) for n in element.pitches)
                instrument_track.append(chordd)
            elif isinstance(element, note.Rest):
                instrument_track.append('R')

        parsed_track.append(instrument_track)

    return pitch_to_vocab_id(parsed_track, vocab)


format_targets = lambda y: tf.unstack(tf.experimental.numpy.moveaxis(y, (0, 1, 2), (1, 2, 0)))[0] if INSTRUMENTS == 1 else tuple(tf.unstack(tf.experimental.numpy.moveaxis(y, (0, 1, 2), (1, 2, 0))))




def load_music_batches(input_dir, output_dir, encoder_decoder=True):
    """
    Loads MIDI files batchwise for encoder-decoder processing.

    Inputs:
        input_dir (str): Directory containing input MIDI batches.
        output_dir (str): Directory containing output MIDI batches.
        encoder_decoder (bool): Flag indicating whether the data is for encoder-decoder processing (default is True).

    Yields:
        Tuple: Input and target data batches formatted for model training.
    """
    while True:
        for inp_batch, output_batch in zip(os.listdir(input_dir), os.listdir(output_dir)):
            try:
                inputs, targets = np.load(os.path.join(input_dir, inp_batch)), np.load(os.path.join(output_dir, output_batch))
                if encoder_decoder:
                    prompt_inputs = np.concatenate([inputs[:, -2:-1], targets[:, :-1]], axis=1)
                    yield [inputs, prompt_inputs], format_targets(targets)
                else:
                    yield inputs, format_targets(targets)
            except Exception as e:
                continue

def top_p_sampling(probabilities, p):
    """
    Selects samples from a probability distribution such that the cumulative probability of the selected samples is less than or equal to a threshold p.

    Inputs:
        probabilities (numpy.ndarray): Probability distribution.
        p (float): Threshold probability.

    Returns:
        numpy.ndarray: Indices of selected samples.
    """
    sorted_indices = np.argsort(probabilities)[::-1]
    cumulative_probs = np.cumsum(probabilities[sorted_indices])
    if np.any(cumulative_probs <= p):
        selected_indices = sorted_indices[cumulative_probs <= p]
    else:
        selected_indices = np.array([np.argmax(probabilities)])
    return selected_indices

def compose_music(music_model, cue=None, topn=6, top_p=None, print_gen=False, encoder_decoder=False, slide_cue_after=100):
    """
    Generates a musical composition represented as sequences of vocab ids sampled from the model predictions.

    Inputs:
        music_model (callable): Model used for music generation.
        cue (numpy.ndarray): Cue for music generation (default is None).
        topn (int): Number of top predictions to consider (default is 6).
        top_p (float): Threshold probability for top-p sampling (default is None).
        print_gen (bool): Flag to print generation information (default is False).
        encoder_decoder (bool): Flag indicating whether the model follows an encoder-decoder architecture (default is False).
        slide_cue_after (int): Number of steps after which to slide the cue (default is 100).

    Yields:
        numpy.ndarray: Musical composition represented as sequences of vocab ids.
    """
    cue = np.expand_dims(cue, axis=0) if cue is not None else None
    composition = [cue[:, -1]] if cue is not None else []
    gen = 1
    start_pcomp = 0
    while True:
        if print_gen:
            print("Generation:", gen)
        gen += 1
        composition_arr = np.concatenate(composition) if composition else np.array([])
        if composition_arr.shape[0] % slide_cue_after == 0:
            start_pcomp += slide_cue_after - 1
            cue = np.expand_dims(composition_arr[-cue.shape[1]:], axis=0)
            pcomp = pcomp[:, -2:-1, :]
        pcomp = np.expand_dims(composition_arr[start_pcomp:start_pcomp+slide_cue_after], axis=0)
        if encoder_decoder:
            pred = np.concatenate(music_model([cue, pcomp]))
        else:
            pred = np.concatenate(music_model(pcomp))
        if INSTRUMENTS == 1:
            pred = np.expand_dims(pred, axis=0)
        preds = []
        for instrument in range(INSTRUMENTS):
            probs = np.exp(pred[instrument, -1])
            probs = probs / np.sum(probs)
            if top_p:
                selected_indices = top_p_sampling(probs, top_p)
                new_probs = np.zeros(probs.shape)
                new_probs[selected_indices] = probs[selected_indices]
            else:
                exclude_pred = np.argsort(probs)[:-topn]
                probs[exclude_pred] = 0.
                new_probs = probs
            new_probs = new_probs / np.sum(new_probs)
            preds += [np.random.choice(vocab['last'], (1,), p=new_probs)]
        preds = np.array(preds)
        currcomp = preds.T
        composition += [currcomp]
        yield np.concatenate(composition)

def get_avg_tempo(dir='lpd_5/lpd_5_full/0', topn=1000):
    """
    Computes the average tempo of MIDI files in the specified directory.

    Inputs:
        dir (str): Directory containing MIDI files (default is 'lpd_5/lpd_5_full/0').
        topn (int): Number of MIDI files to consider for calculating average tempo (default is 1000).

    Returns:
        float: Average tempo.
    """
    samples = os.listdir(dir)[:topn]
    tempo = 0.
    count = 0
    for sample in samples:
        with np.load(os.path.join(dir, sample)) as data:
            tempo += np.sum(data['tempo'])
            count += data['tempo'].shape[0]
    return tempo / count

def pitches_to_midi(pitches_list, instrument_names, resolution, output_file='output.mid'):
    """
    Converts sequences of pitches for notes or chords to MIDI files and saves them to the output file location.

    Inputs:
        pitches_list (list): List of sequences of pitches for notes or chords.
        instrument_names (list): List of instrument names corresponding to the pitches_list.
        resolution (int): MIDI resolution.
        output_file (str): Output file path for the MIDI file (default is 'output.mid').
    """
    score = stream.Score()
    for i, (pitches, instr_name) in enumerate(zip(pitches_list, instrument_names)):
        instr = INS.Instrument()
        instr.partName = instr_name
        stream_obj = stream.Part()
        stream_obj.append(instr)
        for pitch in pitches:
            n = note.Note(pitch)
            stream_obj.append(n)
        score.insert(i, stream_obj)
    score.write('midi', fp=output_file, midiScale=resolution)
    print(f"MIDI file saved as: {output_file}")



def make_midi(composition, tempo=120, resolution=8, output_file=None):
    """
    Creates a MIDI file from a musical composition represented as sequences of vocab ids.

    Inputs:
        composition (numpy.ndarray): Musical composition represented as sequences of vocab ids.
        tempo (int): Tempo of the MIDI file in beats per minute (default is 120).
        resolution (int): MIDI resolution (default is 8).
        output_file (str): Output file path for the MIDI file (default is None).

    Returns:
        str: Output file path of the generated MIDI file.
    """
    tracks = []
    tempo = np.full(composition.shape[0], tempo)

    track_data = {0: ['Drums', 0], 1: ['Piano', 0], 2: ['Guitar', 24], 3:['Bass', 32], 4:['Strings', 48]}
    if INSTRUMENTS == 1:
        track_data = {0: track_data[1]}

    piano_rolls = []
    instrument_names = []

    for i, track_name_program in track_data.items():
        track_name, program = track_name_program
        piano_roll = vocab_id_to_pitch(composition[:, i], rvocab)
        piano_rolls += [piano_roll]
        instrument_names += [track_name]

    pitches_to_midi(piano_rolls, instrument_names, resolution, output_file)
    return output_file







