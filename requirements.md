## Install required packages  
sudo apt-get update  
sudo apt-get install python-pip python-dev libatlas-base-dev gcc gfortran g++  
sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose  


## Install Naoqi for NAO  
http://doc.aldebaran.com/2-5/dev/python/install_guide.html  
After doing export,do:  echo 'export PYTHONPATH={PYTHONPATH}:/home/username/pynaoqi/' >> ~./bashrc  

## Install MuseSDK
MUSE SDK for python http://developer.choosemuse.com/  
Note: since we use 64-bit system for TensorFlow, check the following link for MuseSDK on 64-bit system: http://forum.choosemuse.com/t/issues-running-muselab-and-muse-io/112/20  

## Install Tensorflow  
sudo pip install tensorflow      # Python 2.7; CPU support (no GPU support)  
or upgrade  
sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.3.0-cp27-none-linux_x86_64.whl  

## Install Keras  
sudo pip install keras  
or upgrade  
sudo pip install git+git://github.com/fchollet/keras.git --upgrade

## Install h5py for loading/saving models  
sudo apt-get install libhdf5-dev  
sudo pip install h5py
