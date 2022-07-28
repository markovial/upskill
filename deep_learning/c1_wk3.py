# The general methodology to build a Neural Network is to:

# 1. Define the neural network structure ( # of input units,  # of hidden units, etc). 
# 2. Initialize the model's parameters
# 3. Loop:
    # - Implement forward propagation
    # - Compute loss
    # - Implement backward propagation to get the gradients
    # - Update parameters (gradient descent)

# layer_sizes(X, Y):
n_x = X.shape[0]
n_h = 4
n_y = Y.shape[0]
# return (n_x, n_h, n_y)

# initialize_parameters(n_x, n_h, n_y):
W1 = np.random.randn(n_h, n_x) * 0.01
b1 = np.zeros((n_h, 1))
W2 = np.random.randn(n_y, n_h) * 0.01
b2 = np.zeros((n_y, 1))
parameters = {"W1": W1,
          "b1": b1,
          "W2": W2,
          "b2": b2}
# return parameters

# forward_propagation(X, parameters):
W1 = parameters["W1"]
b1 = parameters["b1"]
W2 = parameters["W2"]
b2 = parameters["b2"]
Z1 = (W1 @ X) + b1
A1 = np.tanh(Z1)
Z2 = (W2 @ A1) + b2
A2 = sigmoid(Z2)
assert(A2.shape == (1, X.shape[1]))
cache = {"Z1": Z1,
         "A1": A1,
         "Z2": Z2,
         "A2": A2}
# return A2, cache


# compute_cost(A2, Y):
m = Y.shape[1] # number of examples
inner = (Y * np.log(A2)) + ( (1-Y) * np.log(1-A2) )
cost = (-1/m) * np.sum( inner )
cost = float(np.squeeze(cost))
# return cost

# backward_propagation(parameters, cache, X, Y)
m = X.shape[1]
W1 = parameters["W1"]
W2 = parameters["W2"]
A1 = cache["A1"]
A2 = cache["A2"]
dZ2 = A2 - Y
dW2 = (1/m) * (dZ2 @ A1.T)
db2 = (1/m) * np.sum(dZ2,axis=1,keepdims=True)
# a = g(z) , g'(z) = 1-a^2 if g(z) = tanh
dZ1 = (W2.T @ dZ2) * (1 - np.power(A1, 2))
dW1 = (1/m) * dZ1 @ X.T
db1 = (1/m) * np.sum(dZ1,axis=1,keepdims=True)
grads = {"dW1": dW1,
         "db1": db1,
         "dW2": dW2,
         "db2": db2}
# return grads

# update_parameters(parameters, grads, learning_rate):
W1 = copy.deepcopy(parameters["W1"])
W2 = copy.deepcopy(parameters["W2"])
b1 = parameters["b1"]
b2 = parameters["b2"]
dW1 = grads["dW1"]
dW2 = grads["dW2"]
db1 = grads["db1"]
db2 = grads["db2"]
W1 = W1 - learning_rate * dW1
W2 = W2 - learning_rate * dW2
b1 = b1 - learning_rate * db1
b2 = b2 - learning_rate * db2
parameters = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2}
# return parameters

# nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False)

parameters = initialize_parameters ( n_x, n_h, n_y )
for i in range(0,num_iterations):

    A2 , cache = forward_propagation  ( X, parameters           )
    cost       = compute_cost         ( A2, Y                   )
    grads      = backward_propagation ( parameters, cache, X, Y )
    parameters = update_parameters    ( parameters, grads       )

# predict(parameters, X):
A2 , cache  = forward_propagation  ( X, parameters )
predictions = np.array( A2 > 0.5 , dtype=int) ;

