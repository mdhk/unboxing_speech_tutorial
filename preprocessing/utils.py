import glob
import pandas as pd
from pathlib import Path


def find_audio_subset(path, subset_config):
    """
    This function finds the files in the audio directory (specified by path) as defined in the
    subset_config, i.e. selecting the specified split and microphone tracks, and leaving out
    the files to exclude.
    """
    if subset_config["split"] != "all":
        subset_dir = path / subset_config["split"]
    else:
        subset_dir = path / "*"
    mic_labels = [str(mic_id).zfill(2) for mic_id in subset_config["microphones"]]
    subset_files = sorted(
        [
            f
            for l in [
                glob.glob(str(subset_dir / f"*/{mic}*.wav")) for mic in mic_labels
            ]
            for f in l
        ]
    )
    subset_files = [
        f for f in subset_files if not Path(f).stem in subset_config["exclude"]
    ]
    return subset_files


def get_speech_segments(speaking_array, times):
    """
    This function finds the transitions between NOT_SPEAKING and SPEAKING in a speech activity column,
    and returns the speech segment start and end times.
    """
    speech_start_idx = [
        i + 1
        for i, (t1, t2) in enumerate(zip(speaking_array[:-1], speaking_array[1:]))
        if t1 == "NOT_SPEAKING" and t2 == "SPEAKING"
    ]
    speech_end_idx = [
        i + 1
        for i, (t1, t2) in enumerate(zip(speaking_array[:-1], speaking_array[1:]))
        if t1 == "SPEAKING" and t2 == "NOT_SPEAKING"
    ]
    speech_segments = [
        (float(times[start_i]), float(times[end_i]))
        for start_i, end_i in zip(speech_start_idx, speech_end_idx)
    ]
    return speech_segments


def get_speaker_turns(
    speaker1_name, speaker1_segments, speaker2_name, speaker2_segments
):
    """
    This function takes two lists of speech segments (for the two named speakers),
    and returns a DataFrame with the start and end times of each speaker turn
    (merging all consecutive speech segments for the same speaker).
    """
    segment_df = pd.DataFrame(
        sorted(
            [(speaker1_name, st, et) for st, et in speaker1_segments]
            + [(speaker2_name, st, et) for st, et in speaker2_segments],
            key=lambda x: x[1],
        ),
        columns=["speaker", "start_time", "end_time"],
    )

    current_turn_start = segment_df["start_time"][0]
    turns = []
    for r, current_row in segment_df.iterrows():
        # the end_time in the last row is the end_time of the last turn
        if r == len(segment_df) - 1:
            turns.append(
                (current_row["speaker"], current_turn_start, current_row["end_time"])
            )
            break
        # otherwise, we'll see if the next speech segment is a new speaker turn
        else:
            next_row = segment_df.loc[r + 1]

        # if the speaker of the next segment is the same as the current segment, it's not a new turn
        if current_row["speaker"] == next_row["speaker"]:
            continue
        # otherwise, the end_time of the current segment is the end_time of the current turn,
        # and the next segment starts a new turn
        else:
            turns.append(
                (current_row["speaker"], current_turn_start, current_row["end_time"])
            )
            current_turn_start = next_row["start_time"]

    turn_df = pd.DataFrame(turns, columns=["speaker", "start_time", "end_time"])

    return turn_df
