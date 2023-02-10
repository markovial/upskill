function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% compute h for every example x_i


% theta1 : parameters for layer 1 : 401x25
% theta2 : parameters for layer 1 : 401x25
% X      : input pixel values     : 5000x400
%
% prediction : class that has max probability with current parameters

% feedforward one layer at a time

% layer 1
a1 = [ones(m,1) X];

% layer 2
z2 = a1*Theta1';
a2 = [ones(size(z2),1) sigmoid(z2)];

% layer 3
z3 = a2*Theta2';
a3 = sigmoid(z3);

[predict_max, index_max] = max(a3, [], 2);

p = index_max;

% =========================================================================


end
