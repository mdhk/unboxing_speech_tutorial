import json
import librosa
import soundfile as sf
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

from utils import find_audio_subset

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--target_sr",
        required=True,
        type=int,
        help="target sample rate",
    )
    parser.add_argument(
        "--source_dir",
        default="audio",
        type=str,
        help="directory to load audio from",
    )
    parser.add_argument(
        "--output_dir",
        default="audio_subset",
        type=str,
        help="directory to save the downsampled audio to",
    )
    parser.add_argument(
        "--subset_config",
        default="subset_config.json",
        type=str,
        help="subset configuration file",
    )
    args, unk_args = parser.parse_known_args()

    SOURCE_DIR = Path(args.source_dir)
    OUTPUT_DIR = Path(args.output_dir)
    subset_config = json.load(open(args.subset_config, "r"))

    subset_files = find_audio_subset(SOURCE_DIR, subset_config)

    for f in tqdm(subset_files, f"Downsampling files from {SOURCE_DIR}.."):
        y, sr = librosa.load(f, sr=args.target_sr)
        if subset_config["split"] != "all":
            output_filepath = f.replace(
                str(SOURCE_DIR / subset_config["split"]), str(OUTPUT_DIR)
            )
        else:
            output_filepath = f.replace(str(SOURCE_DIR), str(OUTPUT_DIR))
        Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_filepath, y, sr)

    print(f"Done! Saved downsampled files to {OUTPUT_DIR}")
