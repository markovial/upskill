import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
from public_tests import *

# Common steps for pre-processing a new dataset are:

    # Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
    # Reshape the datasets such that each example is now a vector of size (num_px * num_px * 3, 1)
    # "Standardize" the data
        # substract the mean of the whole numpy array from each example, 
        # then divide each example by the standard deviation of the whole numpy array



    # Preprocessing the dataset is important.
    # You implemented each function separately
    # sigmoid()
        # initialize() : initialize weights and baises to be arrays of correct size and value
        # propagate()  : weight , bias , inputs , expected outputs
            # calculate loss (feedforward)
            # calculate current gradient (backpropogation)
        # optimize()   : 
            # for a given number of iterations
                # get gradient , get cost = propogate()
                # update weights
                # update biases
            # return last updated values of weight , bias , gradients , costs
        # model()      : combine functions
    # Tuning the learning rate (which is an example of a "hyperparameter") can make a big difference to the algorithm. You will see more examples of this later in this course!


