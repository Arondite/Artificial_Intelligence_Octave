function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
hypothesis = sigmoid(X*theta);
regularizedCost = 0;
regularizedGradient = 0;

%Basic Cost function of logistic regression
J = sum(y .* log(hypothesis) + (1 - y) .* log(1 - hypothesis));
J = J * (-1/m)
%Added parameter for the regularization
regularizedCost = sum((theta).^2);
regularizedCost = regularizedCost * (lambda/(2*m));
%Completed answer
J = J + regularizedCost;

%This begins the gradient calculation
for k = 1:m  
    grad = (grad +(hypothesis(k) - y(k)) .* X(k, :)' );
end
%This is the added parameter for the regularization
for k = 2:length(theta)
  regularizedGradient = regularizedGradient + theta(k);
end

regularizedGradient = regularizedGradient * (lambda/m);

grad = grad + regularizedGradient;
% =============================================================

end
