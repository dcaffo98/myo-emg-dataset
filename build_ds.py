import numpy as np
import os
from constants import *
from gestures import Gestures, GESTURES_DICT
import re
import pandas as pd
import argparse


REGEX_NPZ = re.compile('.*?(.*)_([\d\w-]*)\.npz$')
REGEX_CSV = re.compile('.*?(.*)_([\d\w-]*)\.csv$')



def build_shards_from_csv(csv_path, npz_path):
    for entry in os.scandir(csv_path):
        if entry.is_file():
            re_out = REGEX_CSV.match(entry.name)
            if re_out is not None:
                gesture_str, id = re_out.groups()
                gesture = GESTURES_DICT[gesture_str.upper()]
                shard_filename = os.path.join(npz_path, gesture_str + '_' + id + '.npz')
                df = pd.read_csv(entry.path, header=None)
                if df.shape[-1] > 8:
                    print(f"WARNING: `{entry.name}` has more than 8 columns. Ignoring the exceeding ones...")
                    df = df.iloc[:, :8]
                elif df.shape[-1] < 8:
                    raise ValueError(f"`{entry.name}` has less than 8 columns.")

                # recording protocol
                df['y'] = np.array(Gestures.NEUTRAL, dtype=np.uint8)
                for i in range(SAMPLE_PER_PERIOD, len(df), 2 * SAMPLE_PER_PERIOD):
                    df.iloc[i:i + SAMPLE_PER_PERIOD, -1] = gesture
                df.iloc[MAX_SAMPLES:, -1] = Gestures.NEUTRAL

                X = df.iloc[:, :-1].to_numpy().astype(np.int16)
                y = df.y.to_numpy()
                np.savez_compressed(shard_filename, X=X, y=y)
            else:
                print(f"`{entry.name}` doesn't follow the convention. A compliant file name is `gesture-name_id.csv`")


def build_ds_raw(npz_path):
    X, y = [], []
    for entry in os.scandir(npz_path):
        if entry.is_file() and entry.name.endswith('.npz'):
            shard = np.load(entry.path)
            X.append(shard['X'])
            y.append(shard['y'])
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X, y


def build_ds_from_shards(npz_path, sample_length, sample_overlap):
    X, y = [], []
    for entry in os.scandir(npz_path):
        if entry.is_file() and REGEX_NPZ.match(entry.name) is not None:
            shard = np.load(entry.path)
            shard_X = shard['X']
            shard_y = shard['y']

            samples, labels = [], []
            step = sample_length - sample_overlap
            # managing overlapping gestures. We simply take the most frequent one for each window
            for i in range(0, len(shard_X) - sample_length, step):
                samples.append(shard_X[i:i + sample_length])
                classes, counts = np.unique(shard_y[i:i + sample_length], return_counts=True)
                labels.append(classes[np.argmax(counts)])
            X.extend(samples)
            y.extend(labels)

    if np.any(X):
        X = np.stack(X, axis=0)
        y = np.stack(y, axis=0)
    return X, y


def build_ds(npz_path, sample_length, sample_overlap, csv_path=None):
    if csv_path:
        build_shards_from_csv(csv_path, npz_path)
    return build_ds_from_shards(npz_path, sample_length, sample_overlap)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str, default='',
        help="Path to `.csv` files.")
    parser.add_argument("--npz-path", type=str, default=NPZ_DATASET_SHARDS_PATH,
        help="Path to `.npz` files.")

    args = parser.parse_args()
    
    if args.csv_path:
        build_shards_from_csv(args.csv_path, args.npz_path)
    X, y = build_ds_from_shards(args.npz_path, SAMPLE_LENGTH, SAMPLE_OVERLAP)
    
    if np.any(X):
        print(X.shape, y.shape, X.dtype, y.dtype)
        print(np.unique(y, return_counts=True))
        np.savez_compressed(NP_DATASET_PATH, X=X, y=y)
    else:
        print(f"Unable to create dataset. `{args.csv_path}` (.csv) and/or `{args.npz_path}` (.npz) path empty?")