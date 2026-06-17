#!/usr/bin/env python3
"""basic_run.py — simple wrapper to run audio_features with a single file path.

Usage:
  python basic_run.py "C:\path\to\file.flac"
  python basic_run.py "file.wav" --use-gpu --use-pyfftw
"""
import argparse
import logging
from pathlib import Path
import json

import audio_features

def parse_args():
    p = argparse.ArgumentParser(description="Basic runner for audio_features — supply a single input file path")
    p.add_argument("input", help="Input audio file path")
    p.add_argument("--out-json", "-o", help="Output JSON path (default: replace input extension with .features.json)", default=None)
    p.add_argument("--spectrogram", "-s", help="Spectrogram image path (default: replace input extension with .spectrogram.png)", default=None)
    p.add_argument("--use-gpu", action="store_true", help="Use GPU (PyTorch) for STFT when available")
    p.add_argument("--use-pyfftw", action="store_true", help="Use pyFFTW for faster CPU FFTs when available")
    p.add_argument("--use-librosa", action="store_true", help="Prefer librosa implementations for chroma/pyin (higher-quality but slower)")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"], help="Logging level")
    return p.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    input_path = Path(args.input)
    if not input_path.exists():
        logging.error("Input file not found: %s", input_path)
        raise SystemExit(2)

    out_json = args.out_json or str(input_path.with_suffix(".features.json"))
    spectrogram = args.spectrogram or str(input_path.with_suffix(".spectrogram.png"))

    logging.info("Running audio_features on %s", input_path)
    try:
        features = audio_features.main(
            input_path=str(input_path),
            out_json=out_json,
            spectrogram_path=spectrogram,
            use_gpu=args.use_gpu,
            use_pyfftw=args.use_pyfftw,
            prefer_librosa=args.use_librosa,
        )
        # Print summary
        logging.info("Extraction complete — features written to %s", out_json)
        try:
            print(json.dumps(features, indent=2))
        except Exception:
            logging.debug("Could not pretty-print features; printing keys only")
            print("feature keys:", list(features.keys()))
    except Exception as e:
        logging.exception("Error running audio_features: %s", e)
        raise

if __name__ == "__main__":
    main()