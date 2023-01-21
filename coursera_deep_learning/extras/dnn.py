import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
#from dnn_app_utils_v3 import *





learning_rate = 0.0075

layers_dims   = [ 12288 , 20, 7, 5, 1] #  4-layer model

# def read_data {{{

def read_data():

    # - The input is a (64,64,3) image which is flattened to a vector of size (12288,1).
    # - Multiply [x_0,x_1,...,x_{12287}]^T with weight matrix W^{[1]}
    # - add the intercept b^{[1]}
    # - result is called the linear unit
    # - Take rely of linear unit
    # - This process could be repeated several times for each $(W^{[l]}, b^{[l]})$ depending on the model architecture.
    # - Finally, take the sigmoid of the final linear unit. If it is greater than 0.5, classify it as a cat.




    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

    m_train = train_x_orig.shape[0]
    num_px  = train_x_orig.shape[1]
    m_test  = test_x_orig.shape[0]

    print ("Number of training examples : " + str(m_train))
    print ("Number of testing examples  : " + str(m_test))
    print ("Each image is of size       : (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print ("train_x_orig shape          : " + str(train_x_orig.shape))
    print ("train_y shape               : " + str(train_y.shape))
    print ("test_x_orig shape           : " + str(test_x_orig.shape))
    print ("test_y shape                : " + str(test_y.shape))

    # Flatten the input into a 1D vector
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten  = test_x_orig.reshape(test_x_orig.shape[0], -1).T
    train_x         = train_x_flatten/255.
    test_x          = test_x_flatten/255.

    print ("train_x's shape : " + str(train_x.shape))
    print ("test_x's shape  : " + str(test_x.shape))


# }}} def read_data

# def initialize_parameters_deep {{{

def initialize_parameters_deep(layer_dims):
    """
    ARGUMENTS
        layer_dims : python array (list) containing the dimensions of each layer in our network

    RETURNS
        parameters : python dictionary containing your parameters "W1", "b1", ..., "WL", "bL"   :
                Wl : weight matrix of shape (layer_dims[l], layer_dims[l-1])
                bl : bias vector of shape   (layer_dims[l], 1)
    """

    parameters = {}

    # list elements = number of units per layer
    # list length   = number of network layers

    # layer_dims    = list , with dimensions of each layer
    # [ 5 , 3 , 4 ] = 3 layers : 1 input , 2 hidden
    # Layer 0 w. 5 inputs / neurons
    # Layer 1 w. 3 neurons , W = (4,5) , b = (4,1)
    # Layer 2 w. 4 neurons , W = (3,4) , b = (3,1)

    L = len(layer_dims)

    for l in range(1, L):

        # Shape of W          : ( # neurons      , # prev neurons )
        # Shape of W          : ( # hidden units , # inputs )

        # Shape of W          : ( n[l] , n[l-1] )
        # Shape of b          : ( n[l] , 1      )
        # Shape of Activation : ( n[l] , m      ) , m = X.shape[...]

        parameters['W' + str(l)] = np.random.randn( layer_dims[l], layer_dims[l-1]) * 0.01
        parameters["b" + str(l)] = np.zeros       ((layer_dims[l], 1))
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


# }}} initialize_parameters_deep

# def linear_forward {{{
def linear_forward(A, W, b):
    """
    IMPLEMENT
        linear part of a layer's forward propagation.

    ARGUMENTS
        A        -- previous layer activations , np array , (size of previous layer    , number of examples     )
        W        -- weights matrix             , np array , (size of current layer     , size of previous layer )
        b        -- bias vector                , np array , (size of the current layer , 1                      )

    RETURNS
        Z        -- input of the activation function = pre-activation parameter
        cache    -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently

    """

    Z     = (W @ A) + b
    cache = (A, W, b)

    return Z, cache

# }}} def linear_forward
# def linear_activation_forward {{{

def linear_activation_forward(A_prev, W, b, activation):
    """
    IMPLEMENT
        forward propagation for the LINEAR->ACTIVATION layer

    ARGUMENTS
        A_prev     -- previous layer activations , np array , (size of previous layer    , number of examples     )
        W          -- weights matrix             , np array , (size of current layer     , size of previous layer )
        b          -- bias vector                , np array , (size of the current layer , 1                      )
        activation -- activation function        , string   , "sigmoid" / "relu"

    RETURNS
        A          -- np array , post-activation values
        cache      -- tuple    , stored for computing the backward pass efficiently
            Z                -- pre-activation values
            linear_cache     -- tuple , (Z,W,b) values
            activation_cache -- tuple , (A,W,b) values

    """
    Z, linear_cache     = linear_forward(A_prev, W, b)

    if   ( activation == "relu"    ) : A, activation_cache = relu    ( Z )
    elif ( activation == "sigmoid" ) : A, activation_cache = sigmoid ( Z )

    cache = (linear_cache, activation_cache)

    return A, cache

# }}} linear_activation_forward
# def L_model_forward {{{

def L_model_forward(X, parameters):
    """
    IMPLEMENT
        Forward propagation : [LIN->RELU]*(L-1)->LIN->SIG
            LINEAR->RELU for (L-1) layers
            LINEAR->SIGMOID for Lth layer

    ARGUMENTS
        X          -- data , numpy array of shape (input size, num of examples)
        parameters -- dict , containing weight and bias matrices

    RETURNS
        AL         -- activation value from the output (last) layer
        caches     -- list of caches containing:
                        every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
    """

    caches = []              # initialize empty caches list
    A = X                    # inputs
    L = len(parameters) // 2 # number of layers in the neural network

    # The for loop starts at 1 because layer 0 is the input
    for l in range(1, L):
        A_prev     = A                        # get input values from previous layer
        W          = parameters['W' + str(l)] # get appropriate weight matrix
        b          = parameters['b' + str(l)] # get appropriate bias matrix
        activation = "relu"                   # choose your activation function
        A, cache   = linear_activation_forward(A_prev, W, b, activation)
        caches.append(cache)

    A_prev = A
    W = parameters['W' + str(L)]
    b = parameters['b' + str(L)]
    activation = "sigmoid"
    AL, cache = linear_activation_forward(A_prev, W, b, activation)
    caches.append(cache)

    return AL, caches


# }}} L_model_forward


# def linear_backward {{{

def linear_backward(dZ, cache):
    """
    IMPLEMENT
        Linear portion of backward propagation for a single layer (layer l)

    ARGUMENTS
        dZ    -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- values saved during forward prop of current layer , tuple , (A_prev, W, b)

    RETURNS
        dA_prev   -- Gradient of the cost w.r.t the activation (of the previous layer l-1) , same shape as A_prev
        dW        -- Gradient of the cost w.r.t W (current layer l)                        , same shape as W
        db        -- Gradient of the cost w.r.t b (current layer l)                        , same shape as b
    """

    A_prev, W, b = cache

    m       = A_prev.shape[1]

    # A , W , b  = cache[0] , cache[1] , cache[2]
    # A       = cache[0]
    # W       = cache[1]
    # b       = cache[2]
    # dW      = (1/m) * dZ @ A.T

    dW      = (1/m) * dZ @ A_prev.T
    db      = (1/m) * np.sum(dZ , axis = 1 , keepdims = True)
    dA_prev = W.T @ dZ

    return dA_prev, dW, db


# }}} def linear_backward

# linear_activation_backward {{{



def linear_activation_backward(dA, cache, activation):
    """
    IMPLEMENT
        backward propagation for the LINEAR->ACTIVATION layer.

    ARGUMENTS
        dA         -- post-activation gradient for current layer l
        cache      -- saved values for efficient backprop , tuple  , (linear_cache  , activation_cache)
        activation -- activation function                 , string , sigmoid / relu

    RETURNS
        dA_prev    -- Gradient of the cost w.r.t the activation (of the previous layer l-1) , same shape as A_prev
        dW         -- Gradient of the cost w.r.t W (current layer l)                        , same shape as W
        db         -- Gradient of the cost w.r.t b (current layer l)                        , same shape as b
    """


    linear_cache, activation_cache = cache

    if   ( activation == "relu"    ) : dZ = relu_backward   ( dA , activation_cache )
    elif ( activation == "sigmoid" ) : dZ = sigmoid_backward( dA , activation_cache )

    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


# }}} linear_activation_backward

# def L_model_backward {{{

def L_model_backward(AL, Y, caches):
    """
    IMPLEMENT
        Backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    ARGUMENTS
        AL     -- probability vector, output of the forward propagation (L_model_forward())
        Y      -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches with results of:
                    forward propogation till layer L-1 with relu
                    forward propogation of   layer L   with sigmoid
                    every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                    the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    RETURNS
        grads   -- A dictionary with the gradients
                    grads["dA" + str(l)] = gradient for activation matrix for layer l
                    grads["dW" + str(l)] = gradient for weight     matrix for layer l
                    grads["db" + str(l)] = gradient for bias       matrix for layer l
    """
'''


'''
    grads = {}                             # initialize empty gradients dictionary
    L     = len(caches)                    # get num layers
    m     = AL.shape[1]                    # get shape of output layer
    Y     = Y.reshape(AL.shape)            # set Y to be same shape as AL (final feed forward layer)
    dAL   = -(Y/AL) + ((1 - Y) / (1 - AL)) # get derivative of cost with respect to AL

    current_cache                  = caches[L-1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, "sigmoid")
    grads["dA" + str(L-1)]         = dA_prev_temp
    grads["dW" + str(L)]           = dW_temp
    grads["db" + str(L)]           = db_temp

    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA_prev_temp, current_cache, "relu")
        grads["dA" + str(l)]           = dA_prev_temp
        grads["dW" + str(l + 1)]       = dW_temp
        grads["db" + str(l + 1)]       = db_temp

    return grads


# }}} def L_model_backward

# def compute_cost {{{

def compute_cost(AL, Y):
    """
    IMPLEMENT
        cost function

    ARGUMENTS
        AL   -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y    -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    RETURNS
        cost -- cross-entropy cost
    """

    m     = Y.shape[1]
    cost  = (-1/m) * np.sum( (Y * np.log(AL)) + ( (1-Y) * np.log(1-AL) ) )
    cost  = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    return cost

# }}} def compute_cost

# def update_parameters {{{

def update_parameters(params, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    params -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """

    parameters = params.copy()
    L = len(parameters) // 2 # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

    return parameters


# }}} def update_parameters

# def L_layer_model {{{

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    IMPLEMENTS:
        A L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    ARGUMENTS:
        X              -- data, numpy array of shape (num_px * num_px * 3, number of examples)
        Y              -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        layers_dims    -- list containing the input size and each layer size, of length (number of layers + 1).
        learning_rate  -- learning rate of the gradient descent update rule
        num_iterations -- number of iterations of the optimization loop
        print_cost     -- if True, it prints the cost every 100 steps

    RETURNS:
        parameters     -- parameters learnt by the model. They can then be used to predict.
    """

    # keep track of cost
    costs = []

    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)

        # Compute cost
        cost = compute_cost(AL, Y)

        # Backward propagation
        grads = L_model_backward(AL, Y, caches)

        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 iterations
        if (print_cost and i % 100 == 0 or i == num_iterations - 1) : print("Cost after iteration {} : {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations                      : costs.append(cost)

    return parameters, costs


# }}} def L_layer_model

print(L_model_backward.__doc__)
