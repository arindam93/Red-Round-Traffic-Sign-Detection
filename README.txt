Red round sign detection

Dependencies required:

numpy
pandas
tensorflow
sklearn
PIL
shutil
matplotlib
jupyter notebook

Due to time constraint, the original single class detection problem is reformulated as a binary classification task, where we classify each traffic sign as a red round class or negative.

The codes can be accessed here: https://github.com/arindam93/Red-Round-Traffic-Sign-Detection.git

1) The entire pipeline (data loading, preprocessing, training and testing the CNN model) is detailed in the IPython notebook Red_round_sign.ipynb


If you don’t have Jupyter notebook installed:

2) You can run the data_preprocessing.py file in the command prompt, which will read the original dataset, create a new directory by converting all images to JPEG format, perform traffic sign extraction, resizing and contrast enhancement and save the train and test data in the current directory as numpy arrays.

3) Then run the train_test.py file which will read the saved data, split train into train and validation sets, instantiate, train and test the CNN model, and save the computational graph and model weights and biases in the current directory.


The saved model can be accessed from here: https://www.dropbox.com/sh/doobdqrtr94cjul/AAALxK1Gli0kbeivAAbsSIpca?dl=0

The JPEG converted images should be downloaded from here:
https://drive.google.com/open?id=1lhF33cKqeiMdHOvkeNX-p_f3pDaMuI9t

The original data can be downloaded from here:
