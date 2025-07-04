import os
import glob
import random
import argparse
import logging
from typing import List, Dict, Any

import numpy as np
import torch
import pandas as pd
import librosa
from pydub import AudioSegment
from huggingface_hub import login
import whisper
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate


def setup_logging(level: str = "INFO") -> None:
    """
    Configure logging for the pipeline.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Speaker diarization evaluation pipeline"
    )
    parser.add_argument(
        "--data-dir", 
        type=str, 
        required=True, 
        help="Path to directory containing .wav and .rttm files"
    )
    parser.add_argument(
        "--num-files", 
        type=int, 
        default=10, 
        help="Number of random audio files to process"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--whisper-model", 
        type=str, 
        default="tiny.en", 
        help="Whisper model checkpoint"
    )
    parser.add_argument(
        "--diarization-model", 
        type=str, 
        default="pyannote/speaker-diarization-3.1", 
        help="Pyannote pretrained diarization pipeline"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        choices=["cpu", "cuda"], 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device"
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO",
        help="Logging level"
    )
    return parser.parse_args()


def parse_rttm(rttm_file_path: str) -> Annotation:
    """
    Parse an RTTM file into a pyannote Annotation.
    """
    ref = Annotation()
    with open(rttm_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                start = float(parts[3])
                dur = float(parts[4])
                ref[Segment(start, start + dur)] = parts[7]
    return ref


def create_label_mapping(speakers: List[str]) -> Dict[str, str]:
    """
    Map raw speaker labels to logical SPEAKER_XX labels.
    """
    return {spk: f"SPEAKER_{i:02d}" for i, spk in enumerate(sorted(speakers))}


def evaluate_files(
    audio_paths: List[str],
    rttm_paths: List[str],
    args: argparse.Namespace,
    whisper_model: Any,
    diarization_pipeline: Pipeline
) -> pd.DataFrame:
    """
    Run evaluation on a list of audio files and return results as DataFrame.
    """
    results = []
    der_scores = []

    # Seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    selected = random.sample(audio_paths, min(args.num_files, len(audio_paths)))
    logging.info(f"Selected {len(selected)} files for evaluation")

    for audio_file in selected:
        base = os.path.splitext(os.path.basename(audio_file))[0]
        rttm_file = next((f for f in rttm_paths if base in os.path.basename(f)), None)
        if not rttm_file:
            logging.warning(f"No RTTM found for {audio_file}, skipping")
            continue

        duration = librosa.get_duration(filename=audio_file)
        _ = whisper_model.transcribe(audio_file)

        diarization = diarization_pipeline(audio_file)
        hyp = Annotation()
        speakers = set()
        for turn, _, spk in diarization.itertracks(yield_label=True):
            hyp[Segment(turn.start, turn.end)] = spk
            speakers.add(spk)

        mapping = create_label_mapping(list(speakers))
        mapped = Annotation()
        for seg, _, lbl in hyp.itertracks(yield_label=True):
            mapped[seg] = mapping[lbl]

        ref_ann = parse_rttm(rttm_file)
        der_metric = DiarizationErrorRate()
        score = der_metric(ref_ann, mapped, uem=ref_ann.get_timeline().extent())
        der_scores.append(score)

        results.append({
            "file_name": os.path.basename(audio_file),
            "duration_sec": round(duration, 2),
            "DER": round(score, 4)
        })

        logging.info(f"{os.path.basename(audio_file)}: Duration {duration:.2f}s, DER {100*score:.2f}%")

    avg_der = sum(der_scores) / len(der_scores) if der_scores else 0.0
    logging.info(f"Average DER: {100*avg_der:.2f}%")

    df = pd.DataFrame(results)
    df["average_DER"] = round(avg_der, 4)
    return df


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    # Login to Huggingface
    login()

    # Load models
    whisper_model = whisper.load_model(args.whisper_model).to(args.device)
    diarization_pipeline = Pipeline.from_pretrained(args.diarization_model).to(args.device)

    # Discover files
    audio_files = glob.glob(os.path.join(args.data_dir, "**", "*.wav"), recursive=True)
    rttm_files = glob.glob(os.path.join(args.data_dir, "**", "*.rttm"), recursive=True)

    if not audio_files or not rttm_files:
        logging.error("No audio or RTTM files found, exiting")
        return

    # Run evaluation
    df_results = evaluate_files(audio_files, rttm_files, args, whisper_model, diarization_pipeline)

    # Save results
    output_csv = os.path.join(os.getcwd(), "diarization_results.csv")
    df_results.to_csv(output_csv, index=False)
    logging.info(f"Saved results to {output_csv}")


if __name__ == '__main__':
    main()
