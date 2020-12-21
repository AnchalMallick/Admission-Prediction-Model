clc;

data = load('admission_prediction.txt');

%----fetching relevant data from the dataset----
X = data(:,2:8);  %400 by 7
Y = data(:,9);    %400 by 1

initialTheta = zeros(7,1);    %7 by 1

%----Applying feature normalization on X----
X_sub = X(:,1:6);
%X_norm = featureNorm(X_sub);    %400 by 6
%X_norm = [X_norm,X(:,7)];          %400 by 7
X_norm = X;


%displaying cost for theta = [0;0;0;0;0;0;0]
[cost,grad] = Cost_function(X,Y,initialTheta);
fprintf('Value of cost function at initial theta = [0;0;0;0;0;0;0]:%f\n',cost);

%----Using fminunc to get optimal value of theta----
options = optimset('GradObj', 'on', 'MaxIter', 500);
[theta, cost] = ...
	fminunc(@(t)(Cost_function(X, Y, t)), initialTheta, options);
pause;


%displaying final value of theta and minimized cost
fprintf('Final value of theta(coefficients):\n');
disp(theta);
%
fprintf('Minimized cost:%f\n',cost);
pause;
prediction = X*theta;

comp_vec = [Y,prediction];
fprintf('\n\nComparison between known and predicted values:\n\n');
disp(comp_vec);

%-------Data Visualization-------
pred_binary = prediction>=0.5; 
Y_binary = Y>=0.5;
comp_binary = [Y_binary,pred_binary];
Accuracy = sum(pred_binary==Y_binary)/4;  %Model accuracy
fprintf('\nAccuracy: %f',Accuracy);
fprintf(' percent\n');
pause;


%----Model Accuracy in the form of a pie chart----
ax = subplot(1,1,1);
correct = sum(Y_binary==pred_binary);
wrong = sum(Y_binary~=pred_binary);
x = [correct wrong];
labels = {'Correct prediction','Wrong prediction'};
p = pie(ax,x);
pText = findobj(p,'Type','text');
percentValues = get(pText,'String');
txt = {'Correct Prediction:';'Wrong Prediction:'};
combinedtxt = strcat(txt, percentValues);
pText(1).String = combinedtxt(1);
pText(2).String = combinedtxt(2);
title('Model Accuracy');





