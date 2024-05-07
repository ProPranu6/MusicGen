import os
import mido
from mido import MidiFile, MidiTrack, Message

def is_polyphonic(midi_file):
    """Check if a MIDI file is polyphonic."""
    active_notes = set()
    for track in midi_file.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                if msg.note in active_notes:
                    return True
                active_notes.add(msg.note)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                active_notes.discard(msg.note)
    return False

def convert_to_monophonic(midi_file):
    """Convert polyphonic MIDI file to monophonic by selecting the highest note at each time."""
    new_midi = MidiFile()
    for track in midi_file.tracks:
        new_track = MidiTrack()
        new_midi.tracks.append(new_track)
        current_time = 0
        highest_note = None
        for msg in track:
            if not msg.is_meta:
                if msg.time != 0:
                    if highest_note is not None:
                        new_msg = Message('note_on', note=highest_note, velocity=64, time=current_time)
                        new_track.append(new_msg)
                        new_track.append(Message('note_off', note=highest_note, velocity=64, time=msg.time))
                    current_time = 0
                    highest_note = None
                if msg.type == 'note_on' and msg.velocity > 0:
                    if highest_note is None or msg.note > highest_note:
                        highest_note = msg.note
            current_time += msg.time
        # Ensure that we only append a 'note_off' if a note was actually played
        if highest_note is not None:
            new_track.append(Message('note_off', note=highest_note, velocity=64, time=0))
    return new_midi


def process_directory(source_dir, target_dir):
    """Process all MIDI files in the directory."""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    for file_name in os.listdir(source_dir):
        if file_name.endswith('.mid'):
            path = os.path.join(source_dir, file_name)
            try:
                midi_file = MidiFile(path)
                
                if is_polyphonic(midi_file):
                    print(f"Converting {file_name} to monophonic...")
                    midi_file = convert_to_monophonic(midi_file)
                else:
                    print(f"{file_name} is already monophonic.")
                    
                midi_file.save(os.path.join(target_dir, file_name))
            except (EOFError, IOError) as e:
                print(f"Failed to process {file_name}: {e}")



# Example usage
source_directory = 'baseline_m'
target_directory = 'baseline'
process_directory(source_directory, target_directory)
