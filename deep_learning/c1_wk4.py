# Read topology of network
# Read data
# Initialize the parameters according to given topology
# Repeat for given num of iterations
    # Forward propagation
        # Propogate forward : Compute Z[l]
        # Activate          : Compute g(Z[l])
    # Calculate Loss
    # Back propogation
        # Propogate backward :
        # Compute gradients  : dW[l] , db[l]
    # Update parameters


def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(1)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    # Shape of W          : ( # neurons      , # prev neurons )
    # Shape of W          : ( # hidden units , # inputs )

    # Shape of W          : ( n[l] , n[l-1] )
    # Shape of b          : ( n[l] , 1      )
    # Shape of Activation : ( n[l] , m      ) , m = X.shape[...]

    # layer_dims    = list , with dimensions of each layer
    # [ 5 , 3 , 4 ] = 3 layers = 1 input , 2 hidden
    # Layer 0 w. 5 inputs
    # Layer 1 w. 3 neurons , W = (4,5) , b = (4,1)
    # Layer 2 w. 4 neurons , W = (3,4) , b = (3,1)

    # len(layer_dims) = # hidden layers


    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    return parameters

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A     -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W     -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b     -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z     -- the input of the activation function
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently

    """

    #  the "cache" records values from the forward propagation units and are
    #  used in backward propagation units because it is needed to compute the
    #  chain rule derivatives.

    Z     = (W @ A) + b
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    if ( activation == "sigmoid" ) :
        Z, linear_cache     = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif ( activation == "relu" ) :
        Z, linear_cache     = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X          -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL         -- activation value from the output (last) layer
    caches     -- list of caches containing:
                every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network

    # The for loop starts at 1 because layer 0 is the input
    for l in range(1, L):
        A_prev     = A
        W          = parameters['W' + str(l)]
        b          = parameters['b' + str(l)]
        activation = "relu"
        A, cache   = linear_activation_forward(A_prev, W, b, activation)
        caches.append(cache)

    A_prev = A
    W = parameters['W' + str(L)]
    b = parameters['b' + str(L)]
    activation = "sigmoid"
    AL, cache = linear_activation_forward(A_prev, W, b, activation)
    caches.append(cache)

    return AL, caches


# GRADED FUNCTION: compute_cost

def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y  -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """

    m     = Y.shape[1]
    cost  = (-1/m) * np.sum( (Y * np.log(AL)) + ( (1-Y) * np.log(1-AL) ) )
    cost  = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    return cost


def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """

    A_prev, W, b = cache
    m            = A_prev.shape[1]
    A            = cache[0]
    W            = cache[1]
    b            = cache[2]
    dW           = (1/m) * dZ @ A.T
    db           = (1/m) * np.sum(dZ , axis = 1 , keepdims = True)
    dA_prev      = W.T @ dZ

    return dA_prev, dW, db



# If g(.) is the activation function, sigmoid_backward , relu_backward compute dZ[l] = dA[l-1] * g'[l]





def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """

    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ              = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ              = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L     = len(caches) # the number of layers
    m     = AL.shape[1]
    Y     = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    dAL   = -(Y/AL) + ((1 - Y) / (1 - AL)) # derivative of cost with respect to AL

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
# tuple
# (
    # tuple
    # (
        # tuple( array() , array() , array() ),
        # array()
    # ) ,
    # tuple
    # (
        # tuple( array() , array() , array() ),
        # array()
    # )
# )

# (
    # (
        # (
         # array([[ 0.09649747, -1.8634927 ], [-0.2773882 , -0.35475898], [-0.08274148, -0.62700068], [-0.04381817, -0.47721803]]),
         # array([[-1.31386475,  0.88462238,  0.88131804,  1.70957306], [ 0.05003364, -0.40467741, -0.54535995, -1.54647732], [ 0.98236743, -1.10106763, -1.18504653, -0.2056499 ]]),
         # array([[ 1.48614836], [ 0.23671627], [-1.02378514]])
        # ),
        # array(
            # [
                # [-0.7129932 ,  0.62524497],
                # [-0.16051336, -0.76883635],
                # [-0.23003072,  0.74505627]
            # ]
        # )
    # ),
    # (
        # (
            # array([[ 1.97611078, -1.24412333], [-0.62641691, -0.80376609], [-2.41908317, -0.92379202]]),
            # array([[-1.02387576,  1.12397796, -0.13191423]]),
            # array([[-1.62328545]])
        # ),
        # array([[ 0.64667545, -0.35627076]])
    # )
# )

# GRADED FUNCTION: update_parameters

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
