"""
    Parker Dunn
    EK 381

    HW 10 Challenge

I decided to try to learn how to use TensorFlow Keras API to
to do the binary classification that we were working on in HW 10.

I had a lot to learn, so I wasn't able to build a very powerful
classification model, but I wanted to submit my work anyway to demonstrate
what I have learned.

    FILES IN MY SUBMISSION
    (1) *this file*           - contains the pet_classifier function which runs model
    (2) hw10model.ipynb       - the file where I worked on developing and training the model
    (3) petclassifier.h5      - the trained binary classification model
    (4) eigmat.npy            - these are data files needed the petclassifier function to work
        xsample.npy
        muX.npy

"""

"""  INSTRUCTIONS!

You will the following packages/modules available:
(1) tensorflow
(2) h5py
(3) numpy

You will need the following files available to this script


* "h5py" was a module that I found to save datastructures and models

"""

# SETUP FOR THE FUNCTION

import numpy as np
import h5py
from tensorflow.keras.models import load_model

def load_data():
    model = load_model("petclassifier.h5")
    # X_demo = h5py.File("X_sample.hdf5", 'r');
    with open('eigmat.npy', 'rb') as eigmat:
        V = np.load(eigmat)
    with open('xsample.npy', 'rb') as sample:
        X_demo = np.load(sample)
    with open('muX.npy','rb') as meanvec:
        mu = np.load(meanvec)
    
    return model, V, X_demo, mu

# NOW DEFINING PET CLASSIFIER FUNCTION

def pet_classifier(X, demo=False):
    model, V, X_demo, mu = load_data()
    
    Xpreppared = np.array([(X - mu) @ V])
    X_demo_prepped = np.array([(X_demo - mu) @ V])
    print(Xpreppared.shape)
    print(X_demo_prepped.shape)
    
    if (demo):
        y_pred = model.predict(X_demo_prepped)
        if (y_pred >= 0.5):
            return 1.0
        else:
            return -1.0
        
    else:
        y_out = model.predict(Xpreppared)
        yhat = np.array([np.zeros(y_out.shape[0])]).T
        for i in range(X.shape[0]):
            if (y_out[i,0] >= 0.5):
                yhat[i,0] = 1.0
            else:
                yhat[i,0] = -1.0
                
        return yhat
# end of pet classifier

yguess = pet_classifier(np.zeros(4096), demo=True)
# pet_classifier(np.zeros(4096), demo=True)
print(yguess)