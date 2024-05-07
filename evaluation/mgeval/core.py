# coding:utf-8
"""core.py
Include feature extractor and musically informed objective measures.
"""
import pretty_midi
import numpy as np
import sys
import os
import mido
import glob
import math


# feature extractor
def extract_feature(_file):
    """
    This function extracts two midi feature:
    pretty_midi object: https://github.com/craffel/pretty-midi
    midi_pattern: https://github.com/vishnubob/python-midi

    Returns:
        dict(pretty_midi: pretty_midi object,
             midi_pattern: midi pattern contains a list of tracks)
    """
    
    feature = {'pretty_midi': pretty_midi.PrettyMIDI(_file),
               'midi_pattern': mido.MidiFile(_file)}
    if not feature['pretty_midi'].instruments:
        print("No instruments defined in the MIDI data. Assuming default piano.")
        # Optionally create a default instrument if that aligns with your application needs
        default_instrument = pretty_midi.Instrument(program=0)  # Default to Acoustic Grand Piano
        feature['pretty_midi'].instruments.append(default_instrument)
    return feature


# musically informed objective measures.
class metrics(object):
    def total_used_pitch(self, feature):
        """
        total_used_pitch (Pitch count): The number of different pitches within a sample.

        Returns:
        'used_pitch': pitch count, scalar for each sample.
        """
        piano_roll = feature['pretty_midi'].instruments[0].get_piano_roll(fs=100)
        sum_notes = np.sum(piano_roll, axis=1)
        used_pitch = np.sum(sum_notes > 0)
        return used_pitch

    def bar_used_pitch(self, feature, track_num=1, num_bar=None):
        pattern = feature['midi_pattern']
        metrics.make_ticks_abs(pattern)
        resolution = pattern.ticks_per_beat
        time_sig = [4, 2]  # Default time signature (4/4)
        bar_length = 4 * resolution  # Default bar length for 4/4 time

        for msg in pattern.tracks[track_num]:
            if msg.type == 'time_signature':
                # msg.numerator and msg.denominator provide the time signature details
                time_sig = [msg.numerator, msg.denominator]
                bar_length = time_sig[0] * resolution * 4 / (2**time_sig[1])
                break  # Assuming the first time signature applies for the whole track

        if num_bar is None:
            num_bar = int(round(float(pattern.tracks[track_num][-1].tick) / bar_length))
        
        used_notes = np.zeros(num_bar)  # Simplify to a 1D array for counting
        note_set = [set() for _ in range(num_bar)]  # List of sets to track unique pitches

        # Second pass: count pitches per bar
        for msg in pattern.tracks[track_num]:
            if msg.type == 'note_on' and msg.velocity != 0:
                bar_index = int(msg.time / bar_length)
                if bar_index < num_bar:
                    note_set[bar_index].add(msg.note)

        # Convert sets to counts of unique pitches
        used_pitch = np.array([len(notes) for notes in note_set]).reshape(num_bar, 1)
        return used_pitch

    def total_used_note(self, feature, track_num=1):
        """
        total_used_note (Note count): The number of used notes.
        As opposed to the pitch count, the note count does not contain pitch information but is a rhythm-related feature.

        Args:
        'track_num' : specify the track number in the midi pattern, default is 1 (the second track).

        Returns:
        'used_notes': a scalar for each sample.
        """
        pattern = feature['midi_pattern']
        used_notes = 0
        for msg in pattern.tracks[track_num]:
            if msg.type == 'note_on' and msg.velocity != 0:
                used_notes += 1
        return used_notes


    def bar_used_note(self, feature, track_num=1, num_bar=None):
        """
        Counts the number of NoteOn events per bar, considering only those with velocity > 0.

        Args:
        'track_num' : Index of the track in the MIDI pattern, default is 1 (second track).
        'num_bar': Number of bars to analyze; if None, it calculates based on the total duration.

        Returns:
        'used_notes': Array with the count of used notes per bar.
        """
        pm_object = feature['pretty_midi']
        midi_data = feature['midi_pattern']
        resolution = midi_data.ticks_per_beat
        # Default time signature
        time_sig_numerator = 4  # Most common time signature numerator (4/4)
        time_sig_denominator = 4  # Most common time signature denominator (4/4)
        beats_per_bar = time_sig_numerator * (4 / time_sig_denominator)

        # Calculate total length in ticks and bars if num_bar is None
        total_ticks = 0
        for msg in midi_data.tracks[track_num]:
            total_ticks += msg.time
        bar_length_ticks = resolution * beats_per_bar
        if num_bar is None:
            num_bar = total_ticks // bar_length_ticks + (total_ticks % bar_length_ticks > 0)

        used_notes = np.zeros(num_bar)

        # Calculate note events per bar
        current_tick = 0
        for msg in midi_data.tracks[track_num]:
            current_tick += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                bar_index = int(current_tick / bar_length_ticks)
                if bar_index < num_bar:
                    used_notes[bar_index] += 1

        return used_notes.reshape((num_bar, 1))

    def total_pitch_class_histogram(self, feature):
        """
        total_pitch_class_histogram (Pitch class histogram):
        The pitch class histogram is an octave-independent representation of the pitch content with a dimensionality of 12 for a chromatic scale.
        In our case, it represents to the octave-independent chromatic quantization of the frequency continuum.

        Returns:
        'histogram': histrogram of 12 pitch, with weighted duration shape 12
        """
        piano_roll = feature['pretty_midi'].instruments[0].get_piano_roll(fs=100)
        histogram = np.zeros(12)
        for i in range(0, 128):
            pitch_class = i % 12
            histogram[pitch_class] += np.sum(piano_roll, axis=1)[i]
        if sum(histogram)>0:
            histogram = histogram / sum(histogram)
        return histogram

    def bar_pitch_class_histogram(self, feature, track_num=1, num_bar=8, bpm=120):
        """
        Calculate a pitch class histogram per bar for the specified track in the MIDI file.

        Args:
        feature: Dictionary containing 'pretty_midi' and 'midi_pattern' objects.
        track_num: Index of the track to analyze (1-based index for convenience, default is first track).
        num_bar: Number of bars to analyze; if None, calculates based on the total duration.
        bpm: Beats per minute, default is 120 if not provided.

        Returns:
        A 2D numpy array where each row represents a histogram of pitch classes for each bar.
        """
        pm_object = feature['pretty_midi']  # Extract PrettyMIDI object from feature dictionary
        if pm_object.time_signature_changes:
            time_sig = pm_object.time_signature_changes[-1]
            numerator = time_sig.numerator
            denominator = time_sig.denominator
        else:
            # Default to 4/4 time if no time signature change is found
            numerator = 4
            denominator = 4

        ticks_per_beat = pm_object.resolution
        bar_length_ticks = ticks_per_beat * numerator  # Assumes the denominator is 4
        total_ticks = int(pm_object.get_end_time() * pm_object.resolution)
        if num_bar is None:
            num_bar = total_ticks // bar_length_ticks + (total_ticks % bar_length_ticks > 0)

        histograms = np.zeros((num_bar, 12))  # Initialize a 2D array for 12 pitch classes

        # Iterate through all instruments (excluding drum tracks)
        for instrument in pm_object.instruments:
            if not instrument.is_drum and instrument.program == track_num-1:  # Correct track index check
                for note in instrument.notes:
                    bar_index = int(note.start * pm_object.resolution // bar_length_ticks)
                    if bar_index < num_bar:
                        pitch_class = note.pitch % 12
                        histograms[bar_index, pitch_class] += note.get_duration()  # Weight by note duration

        return histograms

    def pitch_class_transition_matrix(self, feature, normalize=0):
        """
        pitch_class_transition_matrix (Pitch class transition matrix):
        The transition of pitch classes contains useful information for tasks such as key detection, chord recognition, or genre pattern recognition.
        The two-dimensional pitch class transition matrix is a histogram-like representation computed by counting the pitch transitions for each (ordered) pair of notes.

        Args:
        'normalize' : If set to 0, return transition without normalization.
                      If set to 1, normalizae by row.
                      If set to 2, normalize by entire matrix sum.
        Returns:
        'transition_matrix': shape of [12, 12], transition_matrix of 12 x 12.
        """
        pm_object = feature['pretty_midi']
        # Assuming each instrument contributes to a combined transition matrix
        transition_matrix = np.zeros((12, 12))
        for instrument in pm_object.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    # Example logic for determining transitions, adjust as needed
                    current_pitch_class = note.pitch % 12
                    # Some logic to find the next note's pitch class...
                    # This is just placeholder logic; you'll need to implement actual transition tracking
                    next_pitch_class = (note.pitch + 1) % 12  # Placeholder
                    transition_matrix[current_pitch_class, next_pitch_class] += 1

        if normalize == 1:
            row_sums = np.sum(transition_matrix, axis=1, keepdims=True)
            transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums != 0)
        elif normalize == 2:
            total_sum = np.sum(transition_matrix)
            transition_matrix = transition_matrix / total_sum if total_sum > 0 else transition_matrix

        return transition_matrix

    def pitch_range(self, feature):
        """
        pitch_range (Pitch range):
        The pitch range is calculated by subtraction of the highest and lowest used pitch in semitones.

        Returns:
        'p_range': a scalar for each sample.
        """
        piano_roll = feature['pretty_midi'].instruments[0].get_piano_roll(fs=100)
        pitch_index = np.where(np.sum(piano_roll, axis=1) > 0)[0]
        if pitch_index.size == 0:  # Checking if the array is empty
            return 0
        p_range = np.max(pitch_index) - np.min(pitch_index)
        return p_range

    def make_ticks_abs(midi_pattern):
        """
        Converts MIDI delta times to absolute times for each event in the tracks of a mido MidiFile object.
        """
        abs_time = 0
        for track in midi_pattern.tracks:
            abs_time = 0
            for msg in track:
                abs_time += msg.time
                msg.time = abs_time
                
    def avg_pitch_shift(self, feature, track_num=1):
        """
        Calculates average pitch interval between consecutive NoteOn events in the given track number.
        Args:
        'feature': dictionary containing 'midi_pattern' key with mido.MidiFile object.
        'track_num': track index to calculate intervals from.
        Returns:
        'pitch_shift': average pitch shift in the selected track.
        """
        
        midi_pattern = feature['midi_pattern']
        metrics.make_ticks_abs(midi_pattern)  # Convert delta times to absolute times
    
        last_pitch = None
        intervals = []
        for msg in midi_pattern.tracks[track_num]:
            if msg.type == 'note_on' and msg.velocity > 0:  # Checking for NoteOn with non-zero velocity
                if last_pitch is not None:
                    intervals.append(abs(msg.note - last_pitch))
                last_pitch = msg.note

        if intervals:  # To avoid division by zero in case there are no intervals
            return sum(intervals) / len(intervals)
        else:
            return 0

    def avg_IOI(self, feature):
        """
        avg_IOI (Average inter-onset-interval):
        To calculate the inter-onset-interval in the symbolic music domain, we find the time between two consecutive notes.

        Returns:
        'avg_ioi': a scalar for each sample.
        """

        pm_object = feature['pretty_midi']
        onset = pm_object.get_onsets()
        ioi = np.diff(onset)
        avg_ioi = np.mean(ioi) if np.any(ioi) else 0
        return avg_ioi

    def note_length_hist(self, feature, track_num=1, normalize=True, pause_event=False):
        """
        Calculate the note length histogram, handling note lengths according to specified beat length classes.
        Optionally includes rests if pause_event is True.

        Args:
        'track_num': Index of the track in the MIDI pattern, default is 1 (second track).
        'normalize': If True, normalize by vector sum.
        'pause_event': When activated, will double the vector size to represent the same lengths for rests.

        Returns:
        'note_length_hist': The output vector has a length of either 12 or 24 (when pause_event is True).
        """
        midi_data = feature['midi_pattern']
        track = midi_data.tracks[track_num]
        resolution = midi_data.ticks_per_beat
        bar_length = resolution * 4  # Assuming 4/4 time
        hist_size = 24 if pause_event else 12
        note_length_hist = np.zeros(hist_size)

        # Define histogram bins (these are just examples and might need adjustment)
        lengths = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048][:hist_size // 2]
        
        last_note_on = {}
        last_note_off = {}

        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                last_note_on[msg.note] = msg.time
            elif (msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)):
                if msg.note in last_note_on:
                    note_length = msg.time - last_note_on[msg.note]
                    # Convert note_length to a bar fraction
                    note_length_in_beats = note_length / resolution
                    # Find the closest length
                    idx = (np.abs(np.array(lengths) - note_length_in_beats)).argmin()
                    note_length_hist[idx] += 1
                    if pause_event:
                        # Check if there was a note off before this note on
                        if msg.note in last_note_off:
                            rest_length = last_note_on[msg.note] - last_note_off[msg.note]
                            rest_length_in_beats = rest_length / resolution
                            rest_idx = (np.abs(np.array(lengths) - rest_length_in_beats)).argmin()
                            note_length_hist[rest_idx + hist_size // 2] += 1
                last_note_off[msg.note] = msg.time

        if normalize:
            note_length_hist /= np.sum(note_length_hist) if np.sum(note_length_hist) > 0 else 1

        return note_length_hist

    def note_length_transition_matrix(self, feature, track_num=1, normalize=0, pause_event=False):
        """
        Note length transition matrix provides information about the rhythm transitions in music.
        Args:
        'track_num': Index of the track in the MIDI pattern, default is 1.
        'normalize': Normalization mode (0: none, 1: by row, 2: by matrix sum).
        'pause_event': If True, consider rest lengths and double the size of the matrix.
        Returns:
        'transition_matrix': Transition matrix of note lengths.
        """
        midi_data = feature['midi_pattern']
        track = midi_data.tracks[track_num]
        resolution = midi_data.ticks_per_beat

        # Define histogram bins (based on quarter note lengths)
        lengths = [1/4, 1/2, 1, 2, 4]  # example lengths in quarter notes
        matrix_size = len(lengths) * 2 if pause_event else len(lengths)
        transition_matrix = np.zeros((matrix_size, matrix_size))
        
        # To store the last note length index for each note
        last_note_length_idx = {}
        last_note_off_time = {}
        last_note_on_time = {}

        # Iterate through track messages
        current_time = 0
        for msg in track:
            current_time += msg.time  # Accumulate time to get absolute timing
            if msg.type == 'note_on' and msg.velocity > 0:
                last_note_on_time[msg.note] = current_time
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in last_note_on_time:
                    note_length_ticks = current_time - last_note_on_time[msg.note]
                    note_length_quarters = note_length_ticks / resolution
                    idx = (np.abs(np.array(lengths) - note_length_quarters)).argmin()

                    if pause_event and msg.note in last_note_off_time:
                        rest_length_ticks = last_note_on_time[msg.note] - last_note_off_time[msg.note]
                        rest_length_quarters = rest_length_ticks / resolution
                        rest_idx = (np.abs(np.array(lengths) - rest_length_quarters)).argmin() + len(lengths)
                        last_idx = last_note_length_idx.get(msg.note, None)
                        if last_idx is not None:
                            transition_matrix[last_idx][rest_idx] += 1

                    last_note_length_idx[msg.note] = idx if not pause_event else idx + len(lengths)
                    last_note_off_time[msg.note] = current_time

        if normalize == 1:
            row_sums = transition_matrix.sum(axis=1, keepdims=True)
            transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums!=0)
        elif normalize == 2:
            total_sum = transition_matrix.sum()
            transition_matrix = transition_matrix / total_sum if total_sum != 0 else transition_matrix

        return transition_matrix

    # def chord_dependency(self, feature, bar_chord, bpm=120, num_bar=None, track_num=1):
    #     pm_object = feature['pretty_midi']
    #     # compare bar chroma with chord chroma. calculate the ecludian
    #     bar_pitch_class_histogram = self.bar_pitch_class_histogram(pm_object, bpm=bpm, num_bar=num_bar, track_num=track_num)
    #     dist = np.zeros((len(bar_pitch_class_histogram)))
    #     for i in range((len(bar_pitch_class_histogram))):
    #         dist[i] = np.linalg.norm(bar_pitch_class_histogram[i] - bar_chord[i])
    #     average_dist = np.mean(dist)
    #     return average_dist
