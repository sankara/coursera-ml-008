function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = size(X,2);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

s = 0;
for i = 1:m
  x_i = X(i,:);
  h_i = sigmoid(theta' * x_i');
  s = s + (- y(i) * log(h_i) - (1 - y(i)) * log(1 - h_i));
end
J = s / m + lambda / (2 * m) * (sum(theta.^2) - theta(1) ^ 2);

for j = 1:size(X,2)
  s = 0;
  for i = 1:m
    x_i = X(i,:);
    h_i = sigmoid(theta' * x_i');
    s = s + (h_i - y(i)) * x_i(j);
  end
  s = s / m;
  if j > 1
    s = s + lambda / m * theta(j);
  end
  grad(j) = s;
% =============================================================
end
