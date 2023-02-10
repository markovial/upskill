function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha


%data = load('ex1data1.txt');
%X = data(:, 1);
%y = data(:, 2);
%m = length(y); % number of training examples
%theta = zeros(2, 1); % initialize fitting parameters
%num_iters = 1500;
%alpha = 0.01;


% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

J_partial_0 = 0;
J_partial_1 = 0;
h = theta'.*X;

for i = 1:m

J_partial_0 = J_partial_0 + ((h(i,1) + h(i,2)) - y(i))*X(i,1);
J_partial_1 = J_partial_1 + ((h(i,1) + h(i,2)) - y(i))*X(i,2);

endfor

temp0 = theta(1,1) - (alpha*(1/m)*J_partial_0);
temp1 = theta(2,1) - (alpha*(1/m)*J_partial_1);
theta(1,1) = temp0;
theta(2,1) = temp1;

%cost = computeCost(X, y, theta)


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

theta

end
