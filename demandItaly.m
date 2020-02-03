% Assume data are loaded in variable prices
data = readtable('../AssignmentComputational/ElectricPowerItaly.xls', 'Sheet', 2);
myIndex = 3;   % My team No is 2
prices = data.(4 + myIndex);
pricesIndices = (data.(4) == myIndex);
prices = prices(pricesIndices);

% 1) Detrending
% Remove trend with first differences
X = prices(2:end) - prices(1:end-1);
% Remove seasonality by substracting the seasonal components
% of period 7 from each value of the time series
pricesSeasonal = seasonalcomponents(X, 7);
X = X(~isnan(pricesSeasonal)) - pricesSeasonal(~isnan(pricesSeasonal));
n = length(X);

%%%%%%%%%%%%%%%%% A) LINEAR ANALYSIS %%%%%%%%%%%%%%%%%%%%%%%
% Plot the acutal time series and the first differences time
% series(detrended)
figure(1);
plot(prices);
xlabel('t(days)');
ylabel('Demand');
title('Original Time Series with trend');
figure(2);
plot(X);
xlabel('t(days)');
ylabel('First differences of demand');
title('Detrended Time Series');

% 2) Autocorrelation
maxtau = 30;    % max lag value
alpha = 0.05;   % 95% confidence interval
[acM] = autocorrelation(X, maxtau);
zalpha = norminv(1-alpha/2);
autlim = zalpha/sqrt(n);
figure(3)
hold on
for ii=1:maxtau
    plot(acM(ii+1,1)*[1 1],[0 acM(ii+1,2)],'b','linewidth',1.5)
end
plot([0 maxtau+1],[0 0],'k','linewidth',1.5)
plot([0 maxtau+1],autlim*[1 1],'--c','linewidth',1.5)
plot([0 maxtau+1],-autlim*[1 1],'--c','linewidth',1.5)
xlabel('\tau')
ylabel('r(\tau)')
title('Autocorrelation and 95% confidence intervals')

% Partial Autocorrelation
pac = parautocor(X, maxtau);
figure(4)
hold on
for ii=1:maxtau
    plot(acM(ii+1,1)*[1 1],[0 pac(ii)],'b','linewidth',1.5)
end
plot([0 maxtau+1],[0 0],'k','linewidth',1.5)
plot([0 maxtau+1],autlim*[1 1],'--c','linewidth',1.5)
plot([0 maxtau+1],-autlim*[1 1],'--c','linewidth',1.5)
xlabel('\tau')
ylabel('\phi_{\tau,\tau}')
title('Partial Autocorrelation and 95% confidence intervals')

% 3) Fit ARMA(p, q) model to time series
p = 2;
q = 2;
fprintf("Fitting time series with ARMA(%d, %d) model\n", p, q);
[nrmseV ,phiall, thetaall, SDz, aicS, fpeS]=fitARMA(X, p, q);
fprintf("NRMSE: %f\n", nrmseV);
fprintf("Std of noise: %f\n", SDz);
fprintf("AIC: %f\n", aicS);

% 4) Predict with AR(5) -> ARMA(5, 0)
% Array with the percentage of the whole dataset that will be used as test
% dataset each time. For example for the value 0.1 we will use the 10% of
% the values as test set and 90% of them as training set. The results for
% all the values of the array will be compared at the end.
lastCoeffs = [0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5];
nlastArr = lastCoeffs*n;
nlastArr = uint32(nlastArr);    % Convert to int in order to use as index
% Array that holds the NRMSE value for each split
nrmseArr = NaN*ones(length(nlastArr), 1);

for i=1:length(nlastArr)
    [nrmse, preM] = predictARMAnrmse(X, 5, 0, 1, nlastArr(i));
    nrmseArr(i) = nrmse;
end

% Plot NRMSE with respect to cross-validation dataset proportion
figure(5)
plot(lastCoeffs, nrmseArr)
xlabel('Size of cross-validation set as a percentage of n')
ylabel('NRMSE')
title('NRMSE values for AR(5)')

%%%%%%%%%%%%%%%%%%%%% B) NON LINEAR ANALYSIS %%%%%%%%%%%%%%%%%%%%%%%
% Use the model from c)
[~ ,~, ~, ~, ~, ~, model]=fitARMA(X, p, q);

% Use the generated model to make  predictions on the
% initial dataset and substract them from the acutal values to get the 
% remainders times series
y = predict(model, X);
% The first 5 samples cannot be predicted because of AR(6) part
z = X(2:end) - y(2:end);

% Create matrix of permutations
M = 20;
zM = zeros(M, length(z));
rng(1);
for i=1:M
    indcs = randperm(length(z));
    zM(i, :) = z(indcs);
end

% Calculate linear statistc: Autocorrelation for tau=1
tau = 1;
corrM = zeros(M, 1);
temp = autocorrelation(z, tau);
corr_init = temp(tau+1, 2);
for i=1:M
    temp = autocorrelation(zM(i, :), tau);
    corrM(i) = temp(tau+1, 2);
end

% Plot the histogram with 6 bins and a red vertical line at the value of
% the correlation of the remainders time series. It is clear that this
% value could belong to the histogram because it is around the middle. That
% means that the remainders z are uncorrelated.
figure(6);
hold on;
histogram(corrM, 5);
plot([corr_init, corr_init], [0, 8]);
title('Histogram of permutations and acutal value(red line)');

% Calculate non-linear statistic: Mutual Information for tau=1
figure(7);
tau=1;
mutM = zeros(M, 1);
[temp] = mutualinformation(z, tau);
mut_init = temp(tau+1, 2);
for i=1:M
    [temp] = mutualinformation(zM(i, :), tau);
    mutM(i) = temp(tau+1, 2);
end

% Plot the histogram with 6 bins and a red vertical line at the value of
% the Mutual Information of the remainders time series. It is clear that 
% this value cannot belong to the histogram as it is larger than the
% largest value of the histogram. That means that the presence of
% non-linear correlation is very likely to exist.
figure(8);
histogram(mutM, 6);
hold on;
plot([mut_init, mut_init], [0, 8]);
title('Mutual Information of permutations(histogram) and actual value(red line)');

% Calculate non-linear statistic: Correlation dimension
% Use the lab function correlation dimension. All syntax is from that file.
% tau = 4, mmax = 8
mmax = 8;
tau = 4;
figure(9);
corrdimM = zeros(M, 1);
for i=1:M
    [~, ~, ~, ~, temp] = correlationdimension(zM(i, :)', tau, mmax);
    corrdimM(i) = temp(mmax, 4);
end
[~, ~, ~, ~, temp1] = correlationdimension(z, tau, mmax);
corrdim_init = temp1(mmax, 4);

% Plot histogram and actual value
figure(10);
histogram(corrdimM, 6);
hold on;
plot([corrdim_init, corrdim_init], [0, 6]);
title('Histogram of permutations(histogram) and actual value of Corr. Dim.(red line)');
% We conclude that the correlation dimension is around 0.5 so we have a low
% dimension system. Furthermore the actual dimension is at the edge of the
% histogram so we can safely say that it is different from
% the dimension of the permutations timeseries.