# INITIALIZATION METHODS
# zeros initialization        : fails to break symmetry
# large random initialization : too large weights
# He / Xavier initialization  : recommended method

# GRADED FUNCTION: initialize_parameters_zeros 

def initialize_parameters_zeros(layers_dims):
    parameters = {}
    L = len(layers_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l-1]))
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1 ))
    return parameters


def initialize_parameters_random(layers_dims):
    parameters = {}
    L = len(layers_dims) - 1

    for l in range(1, L + 1):
        parameters['W' + str(l)] = np.random.randn( layers_dims[l], layers_dims[l-1]) * 10
        parameters['b' + str(l)] = np.zeros       ((layers_dims[l], 1 ))

    return parameters


def initialize_parameters_he(layers_dims):
    parameters = {}
    L = len(layers_dims) - 1
    for l in range(1, L + 1):
        parameters['W' + str(l)] = np.random.randn( layers_dims[l], layers_dims[l-1]) * np.sqrt(2./layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros       ((layers_dims[l], 1 ))

    return parameters


# REGULARIZATION

def L2_regularization_cost(AL, Y, parameters, lambd):
    """
    Implement the cost function with L2 regularization. See formula (2) above.

    Arguments:
    AL -- post-activation, output of forward propagation, of shape (output size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    parameters -- python dictionary containing parameters of the model

    Returns:
    cost - value of the regularized loss function (formula (2))
    """
    m = Y.shape[1]

    cross_entropy_cost = compute_cost(A3, Y) # This gives you the cross-entropy part of the cost

    L = len(parameters) // 2 # number of layers in the neural network
    weight_matrix_sums = 0
    for l in range(L):
        weight_matrix_sums += (np.sum(np.square( parameters["W" + str(l+1)] )))

    L2_regularization_cost = (1/m) * (lambd/2) * weight_matrix_sums

    cost = cross_entropy_cost + L2_regularization_cost

    return cost


# GRADED FUNCTION: backward_propagation_with_regularization

def L2_regularization_backprop(X, Y, cache, lambd):
    """
    Implements the backward propagation of our baseline model to which we added an L2 regularization.

    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    cache -- cache output from forward_propagation()
    lambd -- regularization hyperparameter, scalar

    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """

    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    #(≈ 1 lines of code)
    # dW3 = 1./m * np.dot(dZ3, A2.T) + None
    # YOUR CODE STARTS HERE
    dW3 = 1./m * np.dot(dZ3, A2.T) + (lambd/m) * W3

    # YOUR CODE ENDS HERE
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    #(≈ 1 lines of code)
    # dW2 = 1./m * np.dot(dZ2, A1.T) + None
    # YOUR CODE STARTS HERE
    dW2 = 1./m * np.dot(dZ2, A1.T) + (lambd/m) * W2

    # YOUR CODE ENDS HERE
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    #(≈ 1 lines of code)
    # dW1 = 1./m * np.dot(dZ1, X.T) + None
    # YOUR CODE STARTS HERE
    dW1 = 1./m * np.dot(dZ1, X.T) + (lambd/m) * W1

    # YOUR CODE ENDS HERE
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients


    theta_plus  = theta + epsilon
    theta_minus = theta - epsilon
    J_plus      = backward_propagation(x, theta_plus)
    J_minus     = backward_propagation(x, theta_plus)
    gradapprox  = (J_plus - J_minus) / ( 2 * epsilon)
