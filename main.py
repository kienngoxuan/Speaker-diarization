# main.py
#Here is my kaggle notebook
import os
import glob
import random
import warnings
import pandas as pd
import torch
import librosa
import whisper
from huggingface_hub import login
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

# --- Configuration ---
# Note: You must first log in to Hugging Face and accept the user agreement for the models.
# from huggingface_hub import login; login()

DATA_DIR = "/kaggle/input/voxconverse-dataset" # Or your local data path
NUM_FILES_TO_EVALUATE = 10
RANDOM_SEED = 42
WHISPER_MODEL_SIZE = "tiny.en" # Options: tiny.en, base, small, medium, large, large-v2
HUGGING_FACE_TOKEN = "YOUR_HUGGING_FACE_READ_TOKEN" # Replace with your token

# --- Setup ---

def setup_environment():
    """Initializes random seeds and suppresses warnings for a clean run."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    warnings.filterwarnings("ignore")

def get_device():
    """Checks for and returns the available Torch device (CUDA or CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")
    return device

# --- Core Functions ---

def parse_rttm_file(rttm_file_path: str) -> Annotation:
    """
    Parses an RTTM file and returns its content as a pyannote.core.Annotation object.

    Args:
        rttm_file_path: The path to the RTTM file.

    Returns:
        A pyannote Annotation object containing the ground truth speaker segments.
    """
    ground_truth = Annotation()
    with open(rttm_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts[0] == "SPEAKER":
                start_time = float(parts[3])
                duration = float(parts[4])
                speaker_label = parts[7]
                segment = Segment(start_time, start_time + duration)
                ground_truth[segment] = speaker_label
    return ground_truth

def create_speaker_label_mapping(diarization: Annotation) -> dict:
    """
    Creates a mapping from the predicted speaker labels (e.g., 'SPEAKER_00') to
    more readable labels (e.g., 'SPEAKER_A', 'SPEAKER_B').

    Args:
        diarization: The pyannote Annotation object with predicted speaker turns.

    Returns:
        A dictionary mapping original speaker labels to new, ordered labels.
    """
    speaker_labels = sorted(diarization.labels())
    mapping = {label: f"SPEAKER_{chr(65 + i)}" for i, label in enumerate(speaker_labels)}
    return mapping

# --- Main Evaluation Logic ---

def main():
    """
    Main function to run the speaker diarization evaluation pipeline.
    """
    print("🚀 Starting Speaker Diarization Evaluation...")
    setup_environment()
    
    # Authenticate with Hugging Face Hub
    try:
        login(token=HUGGING_FACE_TOKEN)
    except Exception as e:
        print(f"🛑 Hugging Face login failed. Please provide a valid token. Error: {e}")
        return

    # Load models
    device = get_device()
    print("Loading models...")
    asr_model = whisper.load_model(WHISPER_MODEL_SIZE, device=device)
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1").to(device)
    print("✅ Models loaded successfully.")

    # Find audio and RTTM files
    audio_files = glob.glob(os.path.join(DATA_DIR, "**", "*.wav"), recursive=True)
    rttm_files = {os.path.splitext(os.path.basename(f))[0]: f for f in glob.glob(os.path.join(DATA_DIR, "**", "*.rttm"), recursive=True)}

    if not audio_files:
        print(f"🛑 No .wav files found in {DATA_DIR}. Please check the path.")
        return

    # Select a random sample of files for evaluation
    selected_audio_files = random.sample(audio_files, min(NUM_FILES_TO_EVALUATE, len(audio_files)))
    evaluation_results = []
    der_metric = DiarizationErrorRate()

    print(f"\nEvaluating {len(selected_audio_files)} files (Seed: {RANDOM_SEED})...")

    # --- Evaluation Loop ---
    for i, audio_path in enumerate(selected_audio_files):
        file_basename = os.path.splitext(os.path.basename(audio_path))[0]
        print(f"\n[{i+1}/{len(selected_audio_files)}] Processing: {file_basename}.wav")

        # Check for corresponding RTTM file
        if file_basename not in rttm_files:
            print(f"   ⚠️ Warning: Ground truth RTTM file not found for {file_basename}. Skipping.")
            continue
        
        rttm_path = rttm_files[file_basename]

        # 1. Run Diarization Pipeline
        diarization = diarization_pipeline(audio_path)
        
        # 2. Get Ground Truth
        ground_truth = parse_rttm_file(rttm_path)

        # 3. Compute Diarization Error Rate (DER)
        der_score = der_metric(ground_truth, diarization)
        
        # Optional: Transcribe for context (not used in DER calculation)
        # transcription_result = asr_model.transcribe(audio_path)
        # transcription_text = transcription_result["text"]

        evaluation_results.append({
            "File Name": f"{file_basename}.wav",
            "DER (%)": round(der_score * 100, 2),
            "Hypothesis Speakers": len(diarization.labels()),
            "Ground Truth Speakers": len(ground_truth.labels()),
        })
        print(f"   📊 DER: {der_score * 100:.2f}%")


    # --- Final Results ---
    if not evaluation_results:
        print("\n🛑 No files were evaluated successfully.")
        return

    results_df = pd.DataFrame(evaluation_results)
    average_der = results_df["DER (%)"].mean()

    print("\n" + "="*50)
    print("✅ EVALUATION COMPLETE")
    print("="*50)
    print(f"\nAverage Diarization Error Rate (DER): {average_der:.2f}%\n")
    
    print("--- Detailed Results ---")
    print(results_df.to_string(index=False))
    print("="*50)


if __name__ == "__main__":
    main()
