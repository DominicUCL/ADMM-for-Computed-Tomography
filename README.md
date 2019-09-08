# ADMM-for-Computed-Tomography
MSc Disseration Inverse Problems in Medical Imaging
# Alternating Direction Method of Multipliers and Machine Learning for Computed-Tomography

This project contains a variety of functions that use ADMM for CT reconstructions of images. The scripts in the learned section are inspired by the Learned Primal-Dual Reconstruction algorithm (https://arxiv.org/abs/1707.06474). 

## Dependencies

ODL (https://github.com/odlgroup/odl)
```console
$ pip install https://github.com/odlgroup/odl/archive/master.zip
```

Utility library adler (https://github.com/adler-j/learned_primal_dual)
```console
$ pip install https://github.com/adler-j/adler/archive/master.zip
```
Tensorflow (It is best to create a new virtual environment and use)
```console
$ pip install --upgrade tensorflow
```

## Others
- matplotlib
 
- numpy 

- scipy

## Usage
The code in this project has been designed to run as scripts

### In Linux 
```console
$ python3 (script_name_here).py
```
### In Windows
```console
$ py (script_name_here).py
```
All methods in this project that use Machine Learning have a script to learn the parameters and a corresponding script with prefix ##evaluate_## to produce results using the parameters that have been learned. These will generate results into a folder of the same name as the function.

All non-learned methods return outputs to the Results folder.
