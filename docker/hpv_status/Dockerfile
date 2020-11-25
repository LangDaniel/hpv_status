FROM tensorflow/tensorflow:2.0.1-gpu-py3 

RUN \
    apt-get update && apt-get install -y \
    vim \
    libsm6 \
    libxext6 \
    libxrender-dev

RUN pip3 install --user pandas scikit-learn h5py opencv-python pyyaml 
