import os

NPZ_DATASET_SHARDS_PATH = f"data{os.sep}npz"
CSV_DATASET_SHARDS_PATH = f"data{os.sep}csv"
SAMPLE_LENGTH = 30
SAMPLE_OVERLAP = 10
N_GESTURES = 4
NP_DATASET_PATH = f"data{os.sep}myo_ds_{SAMPLE_LENGTH}l_{SAMPLE_OVERLAP}ol.npz"
NP_DATASET_PATH_RAW = f"data{os.sep}myo_ds_raw.npz"
NP_DATASET_KMEANS_PATH = f"data{os.sep}myo_ds_{SAMPLE_LENGTH}l_{SAMPLE_OVERLAP}ol_kmeans_labels.npz"
CROSS_VALIDATION_REPORT_PATH = 'cv_report.json'

# recording protocol
FREQ = 50 # hz
GESTURE_PERIOD = 5 # sec
SAMPLE_PER_PERIOD = GESTURE_PERIOD * FREQ
MAX_SAMPLES = 120 * FREQ + 1