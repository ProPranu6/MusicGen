# MusicGen

## Overview


dataset_info : The following Drive Link contains the Lakh Pianoroll Dataset (LPD). It is available in 2 version at present the original version is named as LPD while the processed version which is 10th in size of the original is in LPD_batched. 

MusicGen-Datasets [uploads onto drive in process]: https://drive.google.com/drive/folders/1zI3TlrzxGUndemSmiagPtTUAmJ23MOR2?usp=sharing

[(MusicGen-Presentation Deck)](https://github.com/ProPranu6/MusicGen/blob/spotlight/ML%20Project%20Initial%20Check-in%20Presentation.pdf)


[(MusicGen-Paper)](https://www.overleaf.com/read/mbrnzybhqxfd#df6fb8)

This repository contains scripts and interactive notebooks for generating music melodies using two major modules: `monophony.py` and `polyphony.py`. The `monophony.py` module generates chordless melodies, while the `polyphony.py` module generates melodies with chords included. Additionally, there are interactive notebooks (`music_gen_monophony.ipynb` and `music_gen_polyphony.ipynb`) for loading datasets, preprocessing data according to the paper's specifications, and experimenting with different models.

## Modules

### 1. `monophony.py`

- Generates chordless music melodies.

### 2. `polyphony.py`

- Generates music melodies with chords included.

### 3. `models.py`

- Contains a list of different models for both monophony and polyphony music generation. Includes decoder-only models and transformer-encoder-decoder models as described in the paper available [here](https://www.overleaf.com/read/mbrnzybhqxfd#df6fb8).

## Interactive Notebooks

### 1. `music_gen_monophony.ipynb`

- Loads datasets and preprocesses data for generating chordless melodies.

### 2. `music_gen_polyphony.ipynb`

- Loads datasets and preprocesses data for generating melodies with chords included.

### 3. `music_gen_metrics.ipynb`

- Generates histograms and music sheets based on the compositions produced, as described in the paper.

## Usage

1. Clone the repository to your local machine.
2. Ensure you have the required dependencies installed (details provided in the notebooks).
3. Run the interactive notebooks (`music_gen_monophony.ipynb` and `music_gen_polyphony.ipynb`) to experiment with different models and generate music compositions.
4. Optionally, run `music_gen_metrics.ipynb` to generate additional metrics and visualizations.

## Output

Running the cells in the interactive notebooks generates a folder named `generation_sets`, which contains the MIDI files and `.wav` files for the generated compositions.

## References

For more details on the models and methodologies used, please refer to the paper available [here](https://www.overleaf.com/read/mbrnzybhqxfd#df6fb8).




