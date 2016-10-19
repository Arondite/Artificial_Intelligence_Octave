function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%hypothesis = sigmoid(X*theta);

X2 = [ones(m,1) X];
A2 = sigmoid(X2*Theta1');
X3 = [ones(m, 1) A2];
hypothesis = sigmoid(X3*Theta2');

%regularizedCost = 0;
%regularizedGradient = 0;
Y2 = zeros(m, num_labels);
for i=1:m
    Y2(i, y(i)) = 1;
end
%J = sum(y .* log(hypothesis) + (1 - y) .* log(1 - hypothesis));
%J = J * (-1/m)
J = sum(sum((-Y2).*log(hypothesis) - (1 - Y2).*log(1-hypothesis),2));
J = (1/m) * J;
%regularizedCost = sum((theta).^2);
%regularizedCost = regularizedCost * (lambda/(2*m));
regularizedCost = (sum(sum(Theta1(:,2:end).^2))) + (sum(sum(Theta2(:,2:end).^2)));
regularizedCost = (lambda/(2*m))*regularizedCost;
%J = J + regularizedCost;
J = J + regularizedCost;


%Start of the backwards algorithm
%A1 = [ones(m,1) X];
%Z2 = A1 * Theta1';
%Apart2 = [ones(size(Z2,1),1) sigmoid(Z2)];
%Z3 = Apart2 * Theta2';
%A3 = sigmoid(Z3);

%This finds the error
%delta3 = A3 - Y2;
%delta2 = ((delta3*Theta2) .* sigmoidGradient([ones(size(Z2, 1),1) Z2]))(:,2:end);

%upperDelta1 = delta2'*A1;
%upperDelta2 = delta3'*A2;
%This accumulates the gradient
%Theta1_grad = (upperDelta1./m) + (lambda/m)*[zeros(size(Theta1,1), 1) Theta1(:, 2:end)];
%Theta1_grad = (delta2'*A1) + Theta1(:, 2:end);
%Theta2_grad = (upperDelta2./m) + (lambda/m)*[zeros(size(Theta2,1), 1) Theta2(:, 2:end)];
%Theta2_grad = (upperDelta2./m) + (lambda/m)*(Theta2(:, 2:end));


%Step 5 division of 1/m
%Theta1_grad = (1/m).*(Theta1_grad);
%Theta2_grad = (1/m).*(Theta2_grad);
for t = 1:m
    A1 = X(t,:)';
    A1 = [1; A1];
    Z2 = Theta1 * A1;
    A2 = sigmoid(Z2);
    A2 = [1; A2];
    Z3 = Theta2 * A2;
    A3 = sigmoid(Z3);
    
    Y3 = Y2(t,:)';
    delta3 = A3 - Y3; 
    
    delta2 = (Theta2' * delta3) .* sigmoidGradient([1; Z2]);
    delta2 = delta2(2:end); 

    upperDelta1 = delta2 * A1';
    upperDelta2 = delta3 * A2';
 
    Theta1_grad = Theta1_grad + upperDelta1;
    Theta2_grad = Theta2_grad + upperDelta2;
end


Theta1_grad = (1/m) * Theta1_grad;
Theta2_grad = (1/m) * Theta2_grad;

Theta1Regularized = Theta1 * (lambda/m);
Theta2Regularized = Theta2 * (lambda/m);

Theta1Regularized(:,1) = 0;
Theta2Regularized(:,1) = 0;


Theta1_grad = Theta1_grad + Theta1Regularized;
Theta2_grad = Theta2_grad + Theta2Regularized;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
