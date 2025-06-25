# Preprocess Tragic Talkers audio & transcripts

This folder contains preprocessing scripts to extract the subset of downsampled audio files and text transcripts that are used by the tutorial notebooks in this repository.

To (re)create the subset:
1. Request access for the [TragicTalkers](https://cvssp.org/data/TragicTalkers/) dataset. Using your access credentials, download and untar the **Audio (1.6 GB)** and **Labels (240 MB)** data into this directory, which should subsequently be structured as follows:
    ```
    .
    ├── audio
    │   	├── development
    │   	└── test
    ├── labels
    │       ├── 3D_mouth_detections
    │       ├── bboxes
    │       ├── pose_keypoints
    │       ├── speech_activity
    │       └── transcripts
    ├── download_audio.py
    ├── extract_transcripts.py
    ├── README.md
    ├── subset_config.json
    └── utils.py
    ```
2. Run
   ```
   python extract_transcripts.py
   ```
   to extract transcripts by recording and speaker turn into `transcripts_subset/`, as gathered from the speech_activity and transcripts files in the labels directory. This will also write recordings to exclude into `subset_config.json`, for which speech activity annotations are not available.
3. Run
   ```
   python downsample_audio.py --target_sr=16000
   ```
   to extract downsampled audio recordings into `audio_subset/`. We downsample audio to 16 kHz to make it suitable for processing by models like Whisper.

To extract a different subset, change the _split_ or _microphones_ fields in `subset_config.json`.