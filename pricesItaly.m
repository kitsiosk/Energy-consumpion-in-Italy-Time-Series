% Assume data are loaded in variable prices
myIndex = 3;   % My team No is 2
prices = ElectricPowerItaly.(4 + myIndex);
pricesIndices = (ElectricPowerItaly.(4) == myIndex);
prices = prices(pricesIndices);

% Remove trend with first differences
X = prices(2:end) - prices(1:end-1);
% Remove seasonality by substracting the seasonal components
% of period 7 from each value of the time series
pricesSeasonal = seasonalcomponents(X, 7);
X = X(~isnan(pricesSeasonal)) - pricesSeasonal(~isnan(pricesSeasonal));
n = length(X);

% Plot the acutal time series and the first differences time
% series(detrended)
figure(1);
plot(prices);
xlabel('t(days)');
ylabel('Prices');
title('Original Time Series with trend');
figure(2);
plot(X);
xlabel('t(days)');
ylabel('First differences of prices');
title('Detrended Time Series');

% Autocorrelation
maxtau = 15;    % max lag value
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
% 2b. Partial autocorrelation
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

% Fit ARMA(p, q) model to time series
p = 3;
q = 6;
fprintf("Fitting time series with ARMA(%d, %d) model\n", p, q);
[nrmseV ,phiallV, thetaallV, SDz, aicS, fpeS]=fitARMA(X, p, q);
fprintf("NRMSE: %f\n", nrmseV);
fprintf("Std of noise: %f\n", SDz);
fprintf("AIC: %f\n", aicS);

% Predict with AR(5) -> ARMA(5, 0)
lastCoeffs = [0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5];
nlastArr = lastCoeffs*n;
nlastArr = uint32(nlastArr);    % Convert to int in order to use as index
nrmseArr = NaN*ones(length(nlastArr), 1);
for i=1:length(nlastArr)
    [nrmse, preM] = predictARMAnrmse(X, 5, 0, 1, nlastArr(i));
    nrmseArr(i) = nrmse;
    % For the actual prediction we need to add the seasonal average
    % we substracted before if we want to use the prediction(Here we don't)
    finalPrediction = preM + pricesSeasonal(mod(nlastArr(i), 7) + 1);
end
% Plot NRMSE with respect to cross-validation dataset proportion
figure(5)
plot(lastCoeffs, nrmseArr, 'rx')
xlabel('Size of cross-validation set as a percentage of n')
ylabel('NRMSE')
title('NRMSE values for AR(5)')
