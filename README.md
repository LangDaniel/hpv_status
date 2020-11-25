# HPV Status Prediction for Oropharyngeal Cancer Patients

Code for our paper [Deep Learning Based HPV Status Prediction for Oropharyngeal Cancer Patients](https://arxiv.org/abs/2011.08555) 

## Organization

Code for each of the three models is given in a separate branch:

*  `C3D`: code for the video pre-trained model
*  `CNN`: code for the 3D model trained from scratch
*  `VGG16`: code for the ImageNet pre-trained model

each branch involves a individual `sequence.py` and `model.py` file.
The files `main.py`, `metrics.py` and `run.sh` are shared between branches. 
