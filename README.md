# myo-emg-dataset
The purpose of this repo is to provide a quick and effective way to create a numpy-based dataset for tasks involving gesture recognition using the 8 electromyography sensors of the Myo Armband developed at [Thalmic Labs](https://github.com/thalmiclabs) (now discontinued).

Data is labeled in 2 possible ways:
 - manually, as shown in [build_ds.py](build_ds.py)
 - using [sklearn's K-means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) algorithm. I personally found this to work very well, achieving a micro F1-score above 0.97 for all gestures using a SVM classifier and a proper [*preprocessing_pipeline*](pipelines.py). As downside, it requires you to link the clusters to their related gestures.

<br/><br/>
Currently, there are only 4 supported gestures:
 - neutral pose
 - fist
 - flexion
 - extension
 
 You can easily add your own ones by editing [gestures.py](gestures.py).

 [Here](https://www.kaggle.com/datasets/dcaffo/myoemgdata) you can find a ready-to-use dataset.

<br/><br/>
<br/><br/>

## How to build your own dataset
You first have to record your data in *.csv* format, following the [recording protocol](#recording-protocol). Then, you just have to put the *.csv* files in the [**./data/csv**](data/csv) folder.

Alternatively, you can put your *.npz* files in the [**./data/npz**](data/npz) folder. One such file must be a compressed file created with 
 [np.savez_compressed](https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html) as following: 
```
 np.savez_compressed(dst_filename, X=X, y=y)
```
where
 - X is a [N, 8] ndarray of integers, containing N recordings with the 8 EMG channel values each
 - y is a [N] ndarray of integers containing the labels

<br/><br/>
Then just run (the arguments are the default ones and can be omitted)
```
python build_ds.py --csv-path data/csv --npz-path data/npz
```
You'll find a new *.npz* file in the *data* folder named `myo_ds_XXXl_YYYol.npz`.
*XXX* indicate the length of the sliding window employed to build the dataset. *YYY* is the length of overlapping between 2 subsequent windows. Default values are *XXX=30* and *YYY=30*, but can be modified in [constants.py](constants.py).
The final dataset can be loaded with [np.load](https://numpy.org/doc/stable/reference/generated/numpy.load.html). It will return a dict-like object containing
 - X: an [N, *XXX*, 8] ndarray of integers, containing N windows of *XXX* samples each
 - y: is a [N] ndarray of integers containing the labels

 To directly obtain X and y, optionally already shuffled, you can call `get_dataset`  in [utils.py](utils.py).

<br/><br/>
<br/><br/>


## Recording protocol
You have to put your recording *csv* in the [**./data/csv**](data/csv) folder. Each row of such *.csv* file must contain the raw 8 EMG values coming from the Myo.
The file must be named as `gestureName_someIdentifier.csv`.
<br/><br/>
Each recording session has a duration of 120 seconds. The candidate must starts and finish in the `neutral` pose. During the session, the candidate is asked to alternate between the `neutral` and the given `target gesture` for that session every 5 seconds. The frequency of the EMG sensor is expected to be around 50Hz. 
All these parameters can be customized by editing [constants.py](constants.py).

<br/><br/>
<br/><br/>
## Annotate data through clustering
If you want to achieve better results, I highly encourage you to get rid of your hand-made labels and exploit a clustering algorithm to annotate your data, as shown in [clustering.py](clustering.py). Performance on a SVM (or whatever classifier you want) is very good using this kind of dataset (see [cross_validation.py](cross_validation.py),  Myo data is collected using one of the following ros-based packages [1](https://github.com/uts-magic-lab/ros_myo), [2](https://github.com/roboTJ101/ros_myo)).
Of course, you then have to associates somehow each cluster to its relative label. That's quite easy if your model is good and there aren't too much labels.