function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
% 
%Vectors that copy the first and second column      
  A = X(: , 1);
  B = X(: , 2);
%Finds the mean of the Vector One just created and created a new vector with the 
%mean in all elements
  mu(1) = mean(A);
  tempMu1 = ones(length(X), 1) * mu(1);
%Finds the mean of the Vector Two just created and created a new vector with the 
%mean in all elements
  mu(2) = mean(B);
  tempMu2 = ones(length(X), 1) * mu(2);
%Standard deviation of the vectors
  sigma(1) = std(A);
  sigma(2) = std(B);
%Then the formula  
  tempCol1 = ( (X_norm(:, 1) - tempMu1 )/( sigma(1) ) );
  tempCol2 = ( (X_norm(:, 2) - tempMu2 )/( sigma(2) ) );
%Reassign values
  X_norm(:, 1) = tempCol1;
  X_norm(:, 2) = tempCol2;
% ============================================================

end
