import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler


def _sequence_channel_median(x):
    return np.median(x, axis=1)


sequence_channel_median = FunctionTransformer(_sequence_channel_median, validate=False)


preprocessing_pipeline = Pipeline(steps=(
    ('median', sequence_channel_median),
    ('min_max_scaling', MinMaxScaler(feature_range=(-1, 1)))   
))