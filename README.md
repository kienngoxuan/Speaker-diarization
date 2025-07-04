# Speaker-diarization

# ➡️ Here is the dataset i use: 
https://www.kaggle.com/datasets/washingtongold/voxconverse-dataset

# ➡️ Here is the requirements to run my speaker-diarization:

!pip install -q openai-whisper

!pip install -q pyannote.audio==3.1.1

!pip install -q pydub

!pip install -q matplotlib

!pip install -q librosa

!pip install -q soundfile

!pip install -q huggingface_hub

!pip install -q numpy<2.0


# ➡️ Here is the import libraries:

import whisper

import re

import os

import warnings

import numpy as np

import torch

import matplotlib.pyplot as plt

import librosa

import librosa.display

from pydub import AudioSegment

import colorsys

from huggingface_hub import login

from pyannote.audio import Pipeline

from pyannote.core import Annotation, Segment

from pyannote.metrics.diarization import DiarizationErrorRate

import random

import glob

➡️ You have to agree to share your contact information to access Pyannote model
➡️ Then create access token with Token type = 'Read', copy that token and put it into the output of this code:

from huggingface_hub import login

login()

# ➡️ You can change the number of input files and random seed:

#Set a random seed for reproducibility

random_seeds = [42, 7, 99]

num_files = 10

# ➡️ If you want to have the communication, you can use this code:

![image](https://github.com/user-attachments/assets/0b5f2a64-40df-4e2a-bfe2-a8dca9a25b26)

#You can change the tiny.en into base/small/medium/large/large-v2

model = whisper.load_model("tiny.en")

audio_file_path = "your/directory/abcxyz.wav"

#Should be .wav file

result = model.transcribe(audio_file_path)

text = result["text"]

Split by sentence using regular expressions

sentences = re.split(r'(?<=[.!?])\s+', text)

#Join with newline after each sentence

formatted_text = "\n".join(sentences)

print("\n--- Transcription ---")

print(formatted_text)

# ➡️ The final cell would display the DER (Diarization Error Rate) of each file compared with its ground_truth rttm file
Then it gives you the dataframe which displays name, predicted_diarization, ground_truth, DER score
