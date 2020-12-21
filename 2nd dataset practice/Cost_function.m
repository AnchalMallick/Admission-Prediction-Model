function [cost,grad] = Cost_function(X,Y,theta)
[m,n] = size(X);

cost = 0;
grad = zeros(size(theta));

pred_vec = X*theta;   %7 by 1
cost = (1/(2*m))*sum((pred_vec-Y).^2);
for i = 1:m;
  grad = grad + (pred_vec(i)-Y(i))*X(i,:)';
end
grad = grad/m;
