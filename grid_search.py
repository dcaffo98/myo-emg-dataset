from constants import NP_DATASET_KMEANS_PATH
from pipelines import preprocessing_pipeline
from utils import get_dataset
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
import joblib



pipeline = Pipeline(steps=(
    ('preprocessing', preprocessing_pipeline),
    ('svm', SVC())
))


Cs = np.arange(1, 1100) / 1000


svm_params = {
    'svm__C': Cs
}


model_path = 'svm__30l_10ol__median__min_max_scaler__kmeans.pkl'

scoring = ['accuracy', 'f1_macro', 'f1_micro']



if __name__ == '__main__':
    X, y = get_dataset(NP_DATASET_KMEANS_PATH)
    gscv = GridSearchCV(pipeline, svm_params, scoring=scoring, cv=5, n_jobs=-1, refit='f1_micro', verbose=3)
    gscv.fit(X, y)
    print(f"Best score: {gscv.best_score_}")
    best_model = gscv.best_estimator_
    joblib.dump(best_model, model_path)
