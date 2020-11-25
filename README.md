# HPV Status Prediction for Oropharyngeal Cancer Patients

Code for our paper [Deep Learning Based HPV Status Prediction for Oropharyngeal Cancer Patients](https://arxiv.org/abs/2011.08555) 

## Organization

Code for each of the three models is given in a separate branch:

*  `C3D`: code for the video pre-trained model
*  `CNN`: code for the 3D model trained from scratch
*  `VGG16`: code for the ImageNet pre-trained model

each branch involves a individual `sequence.py`, `parameter/par.yml` and `model.py` file.
The files `main.py`, `metrics.py` and `run.sh` are shared between branches. 

**parameter file**
The parameter file `parameter/par.yml` stores all the hyperparameter settings and file paths
to be used by other files.

**sequence file**
The `sequence.py` file constructs a Tensorflow `Sequence`.
It reads in the complete CT image files and splits them in smaller sections, called bundles.
Bundles are stored on disk in order to be called by the `__getitem__()` function during training.

**model file**
The `model.py` file contains the respective Tensorflow model, with `get_model()` returning
the complete model to be used during training.

**metrics file**
The `metrics.py` file holds all the metrics to be taped during training.

**run file**
The `run.sh` file can be used in order to run the models within a docker container.

**main file**
The `main.py` file calls all the other modules.
