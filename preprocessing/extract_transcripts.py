import json
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
import glob
from tqdm import tqdm

from utils import get_speech_segments, get_speaker_turns


def get_turn_transcripts(transcript_files, speech_activity):
    skipped_recs = []
    turn_dfs = []

    for transcript_fp in tqdm(transcript_files, "Extracting transcripts.."):
        rec_name = Path(transcript_fp).stem

        # exclude transcripts & recordings for which we don't have speech activity annotations
        if rec_name not in speech_activity["recording"].unique():
            skipped_recs.append(rec_name)
            continue

        # preprocess transcript texts
        transcripts = [
            l
            for l in [
                l.strip().replace("\n", " ")
                for l in open(transcript_fp, "r").read().split("\n\n")
            ]
            if not l == ""
        ]
        speakers = [l.split("  ")[0].strip() for l in transcripts]
        transcripts = [l.strip("Romeo").strip("Juliet").strip() for l in transcripts]

        # find the speech activity info for this recording and take the timestamp from whichever camera is listed first
        rec_info = speech_activity[(speech_activity["recording"] == rec_name)]
        camera = rec_info.head(1)["camera"].item()
        rec_info = rec_info[rec_info["camera"] == camera].reset_index(drop=True)

        # get speaker turns from the annotated speech activities
        romeo_segments = get_speech_segments(rec_info["SA_male"], rec_info["time"])
        juliet_segments = get_speech_segments(rec_info["SA_female"], rec_info["time"])
        turn_df = get_speaker_turns("Romeo", romeo_segments, "Juliet", juliet_segments)

        # merge transcripts if there is only one speaker (monologues)
        if len(np.unique(speakers)) == 1 and len(transcripts) > 1:
            transcripts = [" ".join(transcripts)]
        # remove turn segments for which there is no transcript (vocal activity was not speech)
        if len(turn_df) > len(transcripts):
            turn_df = turn_df.loc[: len(transcripts) - 1]

        assert len(turn_df) == len(transcripts)
        turn_df["transcript"] = transcripts
        turn_df["recording"] = rec_name
        turn_dfs.append(turn_df)

    turn_transcripts = pd.concat(turn_dfs).reset_index(drop=True)
    turn_transcripts = turn_transcripts[
        ["recording", "speaker", "start_time", "end_time", "transcript"]
    ]
    return skipped_recs, turn_transcripts


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--labels_dir",
        default="labels",
        type=str,
        help="directory to load speech activity and transcripts from",
    )
    parser.add_argument(
        "--output_dir",
        default="transcripts_subset",
        type=str,
        help="directory to save the preprocessed transcripts to",
    )
    parser.add_argument(
        "--subset_config",
        default="subset_config.json",
        type=str,
        help="subset configuration file",
    )
    args, unk_args = parser.parse_known_args()

    LABELS_DIR = Path(args.labels_dir)
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    subset_config = json.load(open(args.subset_config, "r"))

    # load transcripts and speech activity annotations
    transcript_files = sorted(glob.glob(str(LABELS_DIR / "transcripts/*.txt")))
    if subset_config["split"] != "all":
        speech_activity = pd.read_csv(
            LABELS_DIR / f"speech_activity/{subset_config['split']}_box_SA.csv"
        )
    else:
        speech_activity = pd.concat(
            [
                pd.read_csv(LABELS_DIR / f"speech_activity/development_box_SA.csv"),
                pd.read_csv(LABELS_DIR / f"speech_activity/test_box_SA.csv"),
            ]
        ).reset_index(drop=True)
    speech_activity["recording"] = speech_activity["name"].apply(
        lambda x: x.split("-")[0]
    )
    speech_activity["camera"] = speech_activity["name"].apply(lambda x: x.split("-")[1])

    # collect transcripts per speaker turn
    skipped_recs, turn_transcripts = get_turn_transcripts(
        transcript_files, speech_activity
    )

    # save skipped recording names to exclude them from the subset
    subset_config["exclude"] = skipped_recs
    json.dump(subset_config, open(args.subset_config, "w"), indent=7)

    # merge transcripts for each recording
    rec_transcripts = pd.DataFrame(
        [
            (
                rec_name,
                " ".join(
                    turn_transcripts[turn_transcripts["recording"] == rec_name][
                        "transcript"
                    ]
                ),
            )
            for rec_name in turn_transcripts["recording"].unique()
        ],
        columns=["recording", "transcript"],
    )

    # save turn & recording transcripts to output directory
    turn_transcripts.to_csv(OUTPUT_DIR / "turn_transcripts.csv", index=False)
    rec_transcripts.to_csv(OUTPUT_DIR / "recording_transcripts.csv", index=False)

    print(f"Done! Saved transcripts to {OUTPUT_DIR}")
