# Whisper & Pyannote Speaker Diarization
This project evaluates speaker diarization performance using [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and calculates the Diarization Error Rate (DER) against RTTM ground truth files.

VoxConverse dataset: https://www.kaggle.com/datasets/washingtongold/voxconverse-dataset

My kaggle notebook: https://www.kaggle.com/code/kienngx/fork-of-speaker-diarization-project
# 🚀 Setup

# Prerequisites

- Python 3.8+

- NVIDIA GPU with CUDA (recommended)

- Audio ```(.wav)``` and ground truth ```(.rttm)``` files placed in the ```/data``` directory.

# Installation & Configuration

- Install dependencies from the ```requirements.txt``` file:

``` pip install -r requirements.txt```

- Authenticate with Hugging Face:
  - Accept user conditions for [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) and [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

  - Create a read-only access token at [hf.co/settings/tokens](https://huggingface.co/settings/tokens).

  - Open ```main.py``` and paste your token into the ```HUGGING_FACE_TOKEN``` variable.

# ❔ What the code does
1: Matching Speakers Between Hypothesis and Ground Truth (with Logical Mapping)

- This code normalizes speaker labels in the hypothesis for easier comparison with ground truth but does not explicitly align speaker identities. The DER metric handles speaker permutation when comparing.

2: You can Display the Conversation Using Whisper (Optional)
```python
model = whisper.load_model("tiny.en")
# Audio file path (from your example)
audio_file_path = "/Your/directory/abc.wav"
result = model.transcribe(audio_file_path)
text = result["text"]
# Split by sentence using regular expressions
sentences = re.split(r'(?<=[.!?])\s+', text)
formatted_text = "\n".join(sentences)
print(formatted_text)
```
3: Calculates DER, Gives Per-File Results, Saves CSV:
```python
df_results.to_csv("diarization_results.csv", index=False)
```

# ▶️ Run Evaluation
Execute the main script to run the evaluation on a random sample of files from the ```/data``` directory.
```python main.py```

# 📈 Example Output
The script will first log the real-time progress for each file, then display a final summary table with the results.
The first column will display the file_name, then duration_sec, ground_truth, hypothesis and DER respectively.

Processing Log
```python
🚀 Starting Speaker Diarization Evaluation...
✅ Using device: cuda
Loading models...
✅ Models loaded successfully.

Evaluating 7 files (Seed: 42)...

[1/7] Processing: yrsve.wav | Duration: 587.04s | DER: 6.05%
[2/7] Processing: usbgm.wav | Duration: 58.39s | DER: 0.14%
...
```
Final Results Table
``` ==================================================
✅ EVALUATION COMPLETE
==================================================

Average Diarization Error Rate (DER): 4.42%

--- Detailed Results ---
 File Name  Duration (sec)  DER (%)
 yrsve.wav          587.04     6.05
 usbgm.wav           58.39     0.14
 yuzyu.wav          631.06     3.94
 tiams.wav          151.17     2.70
 paibn.wav          341.89     3.70
 vmbga.wav          659.57     9.98
 kckqn.wav          349.75     4.46
==================================================
```
