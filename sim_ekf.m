%% Model definition
% d/dt y =  x*w + y*(a - x^2 - y^2)/tau + k1*u + n_y
% d/dt x = -y*w + x*(a - x^2 - y^2)/tau + k2*u + n_x
% d/dt w = 0 + n_w
% d/dt a = 0 + n_a
% d/dt k1 = 0 + n_k
% d/dt k2 = 0 + n_k

% Model constants
fs = 500;   % Hz
dt = 1/fs;  % s
tau = 1;    % s (arbitrary constant)

% Initial parameters
X0 = [ 0 .1 2*pi*6 .01 -1e1 1e1 ];
S = [ 1e-2 1e-2 1e-1 1e-3 1e0 1e0 ]; % noise covariances
W = 1e-1; % measurement noise variance


%% Simulate system (with fixed parameters and noise)
rng('default');
u = [zeros(4*fs, 1); (1-cos(2*pi*8*dt*(1:2*fs)))'; zeros(4*fs, 1)];
N = length(u);
D = length(X0);
S0 = S;

X = zeros(N, D);
X(1,:) = X0 + 3*normrnd(0, S0);
for k = 1:N-1
    X(k+1,:) = ekf_f(X(k,:), u(k), tau, dt) + normrnd(0, S0);
    X(k+1,3:end) = max(eps, X(k+1,3:end)); % prevent non-positive values
end

t = dt*(1:N);
y = X(:,1) + normrnd(0, W*ones(N,1));

figure;
plot(t, [X u]);
xlabel('t [s]');
legend({'y', 'x', 'w', 'a', 'k1', 'k2', 'u'});


%% Extended Kalman Filter (EKF)
% Object definition
ekf = extendedKalmanFilter(@ekf_f, @(X,u,tau,dt) X(1), X0);
% ekf.StateTransitionJacobianFcn = @ekf_J;
ekf.MeasurementJacobianFcn = @(X,u,tau,dt) [1 0 0 0 0 0];
ekf.MeasurementNoise = W;
ekf.ProcessNoise = diag(S);
ekf.StateCovariance = diag(S);

% Run EKF
N = length(u);
D = length(X0);
X_ekf = zeros(N, D);
X_ekf(1,:) = X0;
S_ekf = zeros(N, D, D);
S_ekf(1,:,:) = diag(S);
s_ekf = zeros(N, D);
s_ekf(1,:) = S;
for k = 1:N
    [X_ekf(k,:), S_ekf(k,:,:)] = correct(ekf, y(k), u(k), tau, dt);
    s_ekf(k,:) = diag(squeeze(S_ekf(k,:,:)));
    predict(ekf, u(k), tau, dt);
end

%% Predict trajectories
i_pred = 100;
n_pred = 100;
i0_pred = (1:i_pred:N-n_pred);
ni_pred = length(i0_pred);
X_pred = nan(n_pred, ni_pred, D);
s_pred = nan(n_pred, ni_pred, D);
for j = 1:ni_pred
    k = i0_pred(j);
    ekf.State = X_ekf(k,:);
    ekf.StateCovariance = squeeze(S_ekf(k,:,:));
    for i = 1:n_pred
        predict(ekf, u(k+i-1), tau, dt);
        X_pred(i,j,:) = ekf.State;
        s_pred(i,j,:) = sqrt(diag(ekf.StateCovariance));
    end
end
ii_pred = (1:n_pred)' + i0_pred;
t_pred  = t(ii_pred);

%% Plot simulation and EKF state estimation and prediction
h = 1+6;
w = 1;
style_pred = { 'LineStyle','--', 'Color',[0.9290 0.6940 0.1250] };

fig = figure();
set(fig, 'DefaultLineLineWidth', 2);

targets = [ y X(:,2:end) ];
labels = { 'y', 'x', 'w', 'a', 'k1', 'k2' };
scales = { 1, 1, 1/(2*pi), 1, 1, 1 };
units = { 'mV', 'mV', 'Hz', 'mV²', 'mV/V', 'mV/V' };

ha = zeros(h,1);
ha(1) = subplot(h,w,1);
plot(t, u);
xticklabels([]);
ylabel('u [V]');
legend({'u'});

for i = 1:6
    p = i+1;
    ha(p) = subplot(h,w,p); hold on;
    plot(t, scales{i}*targets(:,i));
    plot_errorarea(t, scales{i}*X_ekf(:,i), scales{i}*s_ekf(:,i));
    plot(t_pred, scales{i}*X_pred(:,:,i), style_pred{:});
    plot(t_pred(1,:), scales{i}*X_pred(1,:,i), '>r', 'MarkerSize',3);
    if i<5, xticklabels([]); end
    ylabel(sprintf('%s [%s]', labels{i}, units{i}));
    legend({labels{i}, [labels{i} '_{EKF}'], [labels{i} '_{pred}']});
end

xlabel('t [s]');
linkaxes(ha, 'x');

data.id = 'sim';
suptitle(['Extended Kalman Filter (EKF) - ' escape(data.id)]);
saveas(fig, ['Figures/sim_ekf'], 'png'); 

%% Plot details
ylim(ha(4), [4 8]);
ylim(ha(5), [0 1]);
ylim(ha(6), [0 80]);

ts = 2;
tt = 0:ts:t(end-1);
for i = 1:length(tt)
    xlim(tt(i) + [0 ts]);
    saveas(fig, ['Figures/sim_ekf' sprintf('_%d-%d', xlim())], 'png'); 
end

%xlim('auto');


%% Plot MSE and residuals
X1 = X(:, 1);
residuals_y = y(ii_pred) - X_pred(:,:,1);
residuals_X1 = X1(ii_pred) - X_pred(:,:,1);
mse = mean(residuals_y.^2, 2);
mse_X1 = mean(residuals_X1.^2, 2);
var_y = var(y);
var_X1 = var(X1(:));

fig = figure();
hold on;
plot(1:n_pred, mse);
plot(1:n_pred, mse_X1);
plot(xlim(), [var_y var_y], 'b--');
plot(xlim(), [var_X1 var_X1], 'r--');
xlabel('Prediction steps');
ylabel('MSE [mV²]');
legend({'y_{pred} vs. y', 'y_{pred} vs. X_1', 'var_{y}', 'var_{X_1}'});
suptitle(['Extended Kalman Filter (EKF) - MSE - ' escape(data.id)]);
saveas(fig, ['Figures/sim_ekf_mse'], 'png'); 

%%
fig = figure();
hold on;
pl = plot(1:n_pred, residuals_X1);
pl2 = plot(1:n_pred, mean(residuals_X1,2), 'k', 'LineWidth',3);
pl3 = plot(1:n_pred, median(residuals_X1,2), 'k--', 'LineWidth',3);
xlabel('Prediction steps');
ylabel('Residuals [mV]');
legend([pl(1) pl2 pl3], {'X_1 - y_{pred}', 'mean', 'median'});
suptitle(['Extended Kalman Filter (EKF) - Residuals - ' escape(data.id)]);
saveas(fig, ['Figures/sim_ekf_res'], 'png'); 

%%
ylim(.08*[-1 1])
saveas(fig, ['Figures/sim_ekf_res2'], 'png'); 

%TODO: plot states residual hist/cov_est

%% Function definitions
function X = ekf_f(X, u, tau, dt)
    % State transition function f, specified as a function handle.
    % The function calculates the Ns-element state vector of the system at
    % time step k, given the state vector at time step k-1.
    % Ns is the number of states of the nonlinear system.
    y = X(1);
    x = X(2);
    w = X(3);
    a = X(4);
    k1 = X(5);
    k2 = X(6);
    
    % Calculate derivatives
    y_ =  x*w + y*(a - x^2 - y^2)/tau + k1*u;
    x_ = -y*w + x*(a - x^2 - y^2)/tau + k2*u;
    
    X(:) = X(:) + dt*[y_ x_ 0 0 0 0]';
end

function J = ekf_J(X, u, tau, dt)
    % Jacobian of state transition function f.
    % The function calculates the partial derivatives of the state
    % transition function with respect to the states and process noise.
    % The number of inputs to the Jacobian function must equal the number
    % of inputs of the state transition function and must be specified in
    % the same order in both functions.
    % The function calculates the partial derivative of the state
    % transition function with respect to the states (∂f/∂x). The output is
    % an Ns-by-Ns Jacobian matrix, where Ns is the number of states.
    y = X(1);
    x = X(2);
    w = X(3);
    a = X(4);
    %k1 = X(5);
    %k2 = X(6);
    
    % Calculate jacobian at the given point X.
    J = eye(5) + dt * ...
        [ ...
        - (x^2 - a + y^2)/tau - (2*y^2)/tau,    ...
        w - (2*x*y)/tau,                        ...
        x,                                      ...
        y/tau,                                  ...
        u 0;                                    ...
        ...
        - w - (2*x*y)/tau,                      ...
        - (x^2 - a + y^2)/tau - (2*x^2)/tau,    ...
        -y,                                     ...
        x/tau,                                  ...
        0 u;                                    ...
        ...
        0 0 0 0 0 0; ...
        0 0 0 0 0 0; ...
        0 0 0 0 0 0; ...
        0 0 0 0 0 0; ...
        ];
end
