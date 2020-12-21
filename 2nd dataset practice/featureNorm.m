function X_norm = featureNorm(X)
[m,n] = size(X);

X_norm = zeros(size(X));

for i = 1:n;
  Mean = mean(X(:,i));
  X(:,i) = (X(:,i)-Mean)/(max(X(:,i))-min(X(:,i)));
  
end
X_norm = X_norm + X;