from utils import get_dataset
from constants import N_GESTURES, NP_DATASET_KMEANS_PATH
from sklearn.cluster import KMeans
import numpy as np
from pipelines import preprocessing_pipeline
from sklearn.pipeline import Pipeline



clustering_pipeline = Pipeline(steps=(
    ('preprocessing', preprocessing_pipeline),
    ('kmeans', KMeans(n_clusters=N_GESTURES))
))


if __name__ == '__main__':
    # don't shuffle data if you want to keep the same order as the original dataset
    X, y = get_dataset(dtype_data=np.float64, shuffle=False)
    
    clustering_pipeline.fit(X)
    labels = clustering_pipeline._final_estimator.labels_
    print('Original labels distribution:')
    print(np.unique(y, return_counts=True))
    print('\n******\nKmeans labels distribution:')
    print(np.unique(labels, return_counts=True))
    np.savez_compressed(NP_DATASET_KMEANS_PATH, X=X.astype(np.int16), y=labels.astype(np.uint8))
