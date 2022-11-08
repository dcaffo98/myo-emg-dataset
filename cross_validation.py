from constants import NP_DATASET_KMEANS_PATH, CROSS_VALIDATION_REPORT_PATH, NP_DATASET_PATH
from utils import get_dataset
from grid_search import scoring
from pipelines import preprocessing_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
import json
import numpy as np
import argparse



def np2list(dict_obj):
    for attr in dict_obj:
        if isinstance(dict_obj[attr], np.ndarray):
            dict_obj[attr] = dict_obj[attr].tolist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default=NP_DATASET_KMEANS_PATH,
        help="Path to `.npz` dataset.")
    
    args = parser.parse_args()

    X, y = get_dataset(args.dataset_path)
    model = SVC()
    pipeline = make_pipeline(preprocessing_pipeline, model)
    scores = cross_validate(pipeline, X, y, cv=5, scoring=scoring, n_jobs=-1)
    np2list(scores)
    print(scores)
    print(f"\n\n{'*' * 51}\n* See `{CROSS_VALIDATION_REPORT_PATH}` for a better visualization *\n{'*' * 51}")
    with open(CROSS_VALIDATION_REPORT_PATH, 'w') as f:
        json.dump(scores, f)