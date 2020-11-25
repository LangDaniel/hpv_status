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
The file to construct the docker images is given in `docker/hpv_status`.

**main file**

The `main.py` file calls all the other modules.

## Usage

**Image input**

The `sequence.py` file expects the CT images to be stored in `hdf5` file format
with the following structure:

```
./data/image_data/image_files.h5
|
|---ct_images
|       |------pid_1
|       |------pid_2
|       ...
|       |------pid_n
|
|
|---ct_sgmt
        |------pid_1
        |------pid_2
        ...
        |------pid_n

```
with `pid` given by the respective [TCIA](https://www.cancerimagingarchive.net/) Subject IDs.

**Pre-trained weights**

The pre-trained networks expect the weights to be found at `./data/weights/<weight_file.h5>`.
Names of the files can be sprecified in `parameter/par.yml`.
For the C3D model the weights can be downloaded as a BVLC caffe file
[from the official web page](https://vlg.cs.dartmouth.edu/c3d/).
In order to convert them to `numpy/hdf5` format `utils/convert.py` can be used.
A docker image to install caffe can be constructed with the file given in `docker/convert_caffe`,
`utils/convert.sh` can be used to run a container constructed with this file.
Weights for the VGG16 model can be downloaded from
[here](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5).
 
