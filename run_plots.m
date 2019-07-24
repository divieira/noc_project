% Run and plot direct multiple shooting and MPC simulations

clear variables
close all
import casadi.*

% Set plot defaults
set(0,'defaultFigureRenderer','painters');  % For eps exporting
set(0,'defaultTextFontSize',11);            % title, label
set(0,'defaultTextFontWeight','bold');      % title, label
set(0,'defaultAxesFontSize',12);            % axis tick labels
set(0,'defaultLineLineWidth',2);            % normal plot
set(0,'defaultStairLineWidth',2);           % stairs plot
set(0,'defaultFigurePosition',[10 10 800 600]); % figure size
set(0,'defaultLegendLocation','none');      % manual legend position
LegendPositionNE  = [0.78 0.82 0.14 0.15];  % upper  right
LegendPositionCE  = [0.78 0.34 0.14 0.15];  % center right
LegendPositionNC2 = [0.42 0.86 0.06 0.11];  % upper  center (1x2 plots)
LegendPositionNE2 = [0.86 0.86 0.06 0.11];  % upper  right  (1x2 plots)
LegendPositionNE3 = [0.90 0.84 0.06 0.11];  % upper  right  (1x3 plots)
LegendPositionNE4 = [0.84 0.84 0.08 0.06];  % upper  right  (4x1 plots)
LegendPositionCE4 = [0.84 0.64 0.08 0.06];  % center right  (4x1 plots)
figpath = "Figures/";


%% Global parameters
% Simulation
fs = 160;       % Hz (sampling frequency)
ts = 1/fs;      % s (time step)
x0 = [0; 0.1];  % mV (initial conditions)

% Model parameters (nominal values)
param = [2*pi*6; .01; 1e2; 0];  % [w; a; k1; k2]
tau = 0.1;                      % s (stiffness constant)
sigma = 10;                     % mV/s (for state noise simulations)

% Reference (for multiple shooting and MPC)
f_ref = 8;      % Hz (reference sine frequency)
a_ref = .5;     % mV (reference sine amplitude)

% Control cost (for multiple shooting and MPC)
alpha = .01;    % control cost factor

% MPC parameters (different configurations)
shift_values = [ 1,  5];    % timesteps (MPC interval)
N_mpc_values = [20, 20];    % timesteps (MPC horizon)


%% Model definition
% Declare model variables
x1 = SX.sym('x1');
x2 = SX.sym('x2');
u  = SX.sym('u');
w  = SX.sym('w');
a  = SX.sym('a');
k1 = SX.sym('k1');
k2 = SX.sym('k2');

x = [x1; x2];
p = [w; a; k1; k2];

% Model equations
ode = @(x1,x2,u,w,a,k1,k2,tau) [ ...
    -x2.*w + x1.*(a - x1.^2 - x1.^2)./tau + k1.*u; ...
     x1.*w + x2.*(a - x1.^2 - x2.^2)./tau + k2.*u ];

xdot = ode(x1,x2,u,w,a,k1,k2,tau);


%% Optimal control problem
% Objective
t = SX.sym('t');
ref = @(t) a_ref*sin(2*pi*f_ref*t); % reference (defined as a function)
L = (ref(t)-x1)^2 + alpha*u^2;      % objective function

% Formulate discrete time dynamics for state and objective functions using
% a single-step Runge-Kutta 4th order method integrator at each timestep
F = rk4integrator(x, p, u, t, xdot, L, 1/fs);


%% 1) Uncontrolled system
% Simulation parameters
T = 1;      % s (simulation period)
N = T*fs;   % steps
time = ts*(0:N);

X1 = [0; 0.09]; % initial value for 1st trajectory
X2 = [0; 0.11]; % initial value for 2nd trajectory
for k=2:N+1
    F1k = F('x0',X1(:,k-1), 'p',param, 'u',0, 't',time(k));
    F2k = F('x0',X2(:,k-1), 'p',param, 'u',0, 't',time(k));
    X1(:,k) = full(F1k.xf);
    X2(:,k) = full(F2k.xf);
end

% Plot trajectories in single figure
fig = figure();
fig.Position(3) = 2*fig.Position(3); % resize width

% Plot time trajectory
subplot(1,2,1);
plot_X_trajectory(time, X1, param);
legend('Position',LegendPositionNC2);
title('Time trajectory');

% Plot phase plane trajectory
subplot(1,2,2);
plot_phase_trajectory(X1, X2, param, tau, ode);
legend('Position',LegendPositionNE2);
title('Phase trajectory');

save_figure(fig,figpath,'1_Model');


%% 2.a) Noise sensitivity: Multiple Shooting
% Simulation parameters
T = 1;      % s (simulation period)
N = T*fs;   % steps
time = ts*(0:N);

% Simulate the direct multiple shooting solution by passing "N" as the MPC
% interval and horizon parameters, effectively performing a single control
% evaluation over the entire simulation period
shift = N;  % MPC interval
N_mpc = N;  % MPC horizon

% Run simulation with zero and nominal state noise scenarios
rng default; % Fix RNG for reproducibility
[X_ms_ideal, U_ms_ideal] = MPC(F, x0, param, 0,     N, N_mpc, shift, ts);
[X_ms_noise, U_ms_noise] = MPC(F, x0, param, sigma, N, N_mpc, shift, ts);
mse_ms_ideal = mean((X_ms_ideal(1,:)-ref(time)).^2);
mse_ms_noise = mean((X_ms_noise(1,:)-ref(time)).^2);

% Plot solutions for both scenarios
fig = figure();
time = ts*(0:N);

% Ideal scenario
subplot(2,1,1);
plot_trajectory(time, ref, X_ms_ideal, U_ms_ideal);
legend('Position',LegendPositionNE);
title(sprintf('\\sigma = %g mV/s (MSE: %.2e mV^2)', 0, mse_ms_ideal));

% Noise scenario
subplot(2,1,2);
plot_trajectory(time, ref, X_ms_noise, U_ms_noise);
title(sprintf('\\sigma = %g mV/s (MSE: %.2e mV^2)', sigma, mse_ms_noise));

sgtitle('Direct multiple shooting (open loop)');
save_figure(fig,figpath,'2_Noise_MS');


%% 2.b) Noise sensitivity: MPC (different configurations)
% Run simulation for each MPC configuration
n_config = length(shift_values);
X_mpc_ideal = cell(n_config,1);
U_mpc_ideal = cell(n_config,1);
X_mpc_noise = cell(n_config,1);
U_mpc_noise = cell(n_config,1);
mse_mpc_ideal = zeros(n_config,1);
mse_mpc_noise = zeros(n_config,1);
for c=1:n_config
    % Set MPC configuration
    shift = shift_values(c); % MPC interval
    N_mpc = N_mpc_values(c); % MPC horizon

    % Run simulation with zero and nominal state noise scenarios
    rng default; % Fix RNG for reproducibility
    [X_mpc_ideal{c}, U_mpc_ideal{c}] = MPC(F, x0, param, 0,     N, N_mpc, shift, ts);
    [X_mpc_noise{c}, U_mpc_noise{c}] = MPC(F, x0, param, sigma, N, N_mpc, shift, ts);
    mse_mpc_ideal(c) = mean((X_mpc_ideal{c}(1,:)-ref(time)).^2);
    mse_mpc_noise(c) = mean((X_mpc_noise{c}(1,:)-ref(time)).^2);
end

% Plot solutions
for c=1:n_config
    % Get MPC configuration
    shift = shift_values(c); % MPC interval
    N_mpc = N_mpc_values(c); % MPC horizon

    fig = figure();

    % Ideal scenario
    subplot(2,1,1)
    plot_trajectory(time, ref, X_mpc_ideal{c}, U_mpc_ideal{c});
    legend('Position',LegendPositionNE);
    title(sprintf('\\sigma = %g mV/s (MSE: %.2e mV^2)', 0, mse_mpc_ideal(c)));

    % Noise scenario
    subplot(2,1,2)
    plot_trajectory(time, ref, X_mpc_noise{c}, U_mpc_noise{c});
    title(sprintf('\\sigma = %g mV/s (MSE: %.2e mV^2)', sigma, mse_mpc_noise(c)));

    sgtitle(sprintf('Model Predictive Control (interval: %d, horizon: %d)', shift, N_mpc));

    save_figure(fig,figpath,sprintf("3_Noise_MPC_%d_%d", shift, N_mpc));
end

%% 2.*) Summary plot: Noise sensitivity
fig = figure();
fig.Position(3) = 1.5*fig.Position(3); % resize width

% Multiple shooting
% Ideal scenario
subplot(2,3,1);
plot_trajectory(time, ref, X_ms_ideal, U_ms_ideal);
title(sprintf('Direct multiple shooting (open loop)\n\\sigma = %g mV/s (MSE: %.2e mV^2)', 0, mse_ms_ideal));

% Noise scenario
subplot(2,3,4);
plot_trajectory(time, ref, X_ms_noise, U_ms_noise);
title(sprintf('\\sigma = %g mV/s (MSE: %.2e mV^2)', sigma, mse_ms_noise));

% MPC
for c = 1:n_config
    % Get MPC configuration
    shift = shift_values(c); % MPC interval
    N_mpc = N_mpc_values(c); % MPC horizon

    % Ideal scenario
    subplot(2,3,1+c)
    plot_trajectory(time, ref, X_mpc_ideal{c}, U_mpc_ideal{c});
    title(sprintf('Model Predictive Control (interval: %d, horizon: %d)\n\\sigma = %g mV/s (MSE: %.2e mV^2)', shift, N_mpc, 0, mse_mpc_ideal(c)));

    % Noise scenario
    subplot(2,3,4+c)
    plot_trajectory(time, ref, X_mpc_noise{c}, U_mpc_noise{c});
    title(sprintf('\\sigma = %g mV/s (MSE: %.2e mV^2)', sigma, mse_mpc_noise(c)));
end

legend('Position',LegendPositionNE3);
save_figure(fig,figpath,"2_NoiseSummary");

%% 3.a) Model parameter deviations: Multiple shooting
T = 5;          % s
N = T*fs;       % steps
time = ts*(0:N);

% Create vector with values 2 in [ta,tb), .5 in [tc,tc), 1 otherwise
perturbation = @(ta,tb,tc,time) 1 + rectangularPulse(ta,tb,time) - 0.5*rectangularPulse(tb,tc,time);

% Parameter vector for simulation (perturbation on nominal values)
w_perturb  = param(1)*perturbation(0.5, 1.0, 1.5, time);
a_perturb  = param(2)*perturbation(2.0, 2.5, 3.0, time);
k1_perturb = param(3)*perturbation(3.5, 4.0, 4.5, time);
k2_perturb = param(4)*ones(1,N+1); %unperturbed
param_sim = [w_perturb; a_perturb; k1_perturb; k2_perturb];

% Simulate the direct multiple shooting solution by passing "N" as the MPC
% interval and horizon parameters, effectively performing a single control
% evaluation over the entire simulation period
shift = N;  % MPC interval
N_mpc = N;  % MPC horizon

% Run simulation with no noise and perturbed parameters
rng default; % Fix RNG for reproducibility
[X_ms_perturb, U_ms_perturb] = MPC(F, x0, param, 0, N, N_mpc, shift, ts, param_sim);
mse_ms_perturb = mean((X_ms_perturb(1,:)-ref(time)).^2);

% Plot solutions
% Plot the normalized parameters
fig = figure();
subplot(2,1,1);
plot(time, (param_sim./param)');
ylim([0 2]);
xlabel('t [s]');
legend('w/w_0','a/a_0','k_1/k_{1,0}', 'Position',LegendPositionNE);
title({"Normalized Parameters"});

% Plot the solution
subplot(2,1,2);
hold on
plot_trajectory(time, ref, X_ms_perturb, U_ms_perturb);
title(sprintf('Direct multiple shooting (MSE: %.2e mV^2)', 0, mse_ms_perturb));
legend('Position', LegendPositionCE);

sgtitle('Direct multiple shooting with parameter deviations');
save_figure(fig,figpath,"3_Perturb_MS");


%% 3.b) Model parameter deviations: MPC
% Run simulation for each MPC configuration
n_config = length(shift_values);
X_mpc_perturb = cell(n_config,1);
U_mpc_perturb = cell(n_config,1);
mse_mpc_perturb = zeros(n_config,1);
for c=1:n_config
    % Set MPC configuration
    shift = shift_values(c); % MPC interval
    N_mpc = N_mpc_values(c); % MPC horizon

    % Run simulation with no noise and perturbed parameters
    rng default; % Fix RNG for reproducibility
    [X_mpc_perturb{c}, U_mpc_perturb{c}] = MPC(F, x0, param, 0, N, N_mpc, shift, ts, param_sim);
    mse_mpc_perturb(c) = mean((X_mpc_perturb{c}(1,:)-ref(time)).^2);
end

% Plot solutions
for c=1:n_config
    % Get MPC configuration
    shift = shift_values(c); % MPC interval
    N_mpc = N_mpc_values(c); % MPC horizon

    % Plot the normalized parameters
    fig = figure();
    subplot(2,1,1);
    plot(time, (param_sim./param)');
    ylim([0 2]);
    xlabel('t [s]');
    legend('\omega/\omega_0','a/a_0','k_1/k_{1,0}', 'Position',LegendPositionNE);
    title({"Normalized Parameters"});

    % Plot the solution
    subplot(2,1,2);
    hold on
    plot_trajectory(time, ref, X_mpc_perturb{c}, U_mpc_perturb{c});
    title(sprintf('MPC (MSE: %.2e mV^2)', 0, mse_mpc_perturb(c)));
    legend('Position', LegendPositionCE);

    sgtitle(sprintf('Model Predictive Control (interval: %d, horizon: %d)\nwith parameter deviations', shift, N_mpc));

    save_figure(fig,figpath,sprintf("3_Perturb_MPC_%d_%d", shift, N_mpc));
end


%% 3.*) Summary plot: Model parameter deviations
fig = figure();
fig.Position(4) = 2*fig.Position(4); % resize width

% Plot the normalized parameters
subplot(4,1,1);
plot(time, (param_sim./param)');
ylim([0 2]);
xlabel('t [s]');
legend('\omega/\omega_0','a/a_0','k_1/k_{1,0}', 'Position',LegendPositionNE4);
title('Model parameter deviations');

% Multiple shooting
subplot(4,1,2);
hold on
plot_trajectory(time, ref, X_ms_perturb, U_ms_perturb);
title(sprintf('Direct multiple shooting (MSE: %.2e mV^2)', mse_ms_perturb));
legend('Position', LegendPositionCE4);

%MPC
for c=1:n_config
    % Get MPC configuration
    shift = shift_values(c); % MPC interval
    N_mpc = N_mpc_values(c); % MPC horizon

    % Plot the normalized parameters
    subplot(4,1,2+c);
    hold on
    plot_trajectory(time, ref, X_mpc_perturb{c}, U_mpc_perturb{c});
    title(sprintf('MPC (interval: %d, horizon: %d, MSE: %.2e mV^2)', shift, N_mpc, mse_mpc_perturb(c)));
end

save_figure(fig,figpath,"3_Perturb_Summary");


%% 4) Control cost factor (MPC)
T = 1;          % s
N = T*fs;       % steps
time = ts*(0:N);

% Control cost values to evaluate
alpha_values = logspace(-3,3,13);
n_alpha = length(alpha_values);

% MPC parameters to evaluate (different configurations)
shift_values = [ 1,  1,  5,  5, 10, 10]; % timesteps (MPC interval)
N_mpc_values = [20, 10, 20, 10, 20, 10]; % timesteps (MPC horizon)
n_config = length(shift_values);

% Run simulation for each MPC configuration
mse = zeros(n_alpha,n_config); % mean squared error to reference
mce = zeros(n_alpha,n_config); % mean control energy
for c=1:n_config
    % Set MPC configuration
    shift = shift_values(c); % MPC interval
    N_mpc = N_mpc_values(c); % MPC horizon

    for idx=1:n_alpha
        % Update objective and dynamics with alpha value
        alpha = alpha_values(idx);
        L_alpha = (ref(t)-x1)^2 + alpha*u^2;
        F_alpha = rk4integrator(x, p, u, t, xdot, L_alpha, 1/fs);

        % Run simulation with updated objective function
        rng default; % Fix RNG for reproducibility
        [X_applied, U_applied] = MPC(F_alpha, x0, param, sigma, N, N_mpc, shift, ts);
        mce(idx,c) = mean(U_applied.^2);
        mse(idx,c) = mean((X_applied(1,:)-ref(time)).^2);
    end
end

%% Plot the results
fig = figure();
subplot(2,1,1);
loglog(alpha_values,mce)
xlabel('Control cost factor /alpha');
ylabel('Mean control energy [a.u.^2]');
legend(arrayfun(@(c) {sprintf("interval: %d, horizon: %d", shift_values(c), N_mpc_values(c))}, 1:n_config), 'Position',LegendPositionNE);

subplot(2,1,2);
loglog(alpha_values,mse)
xlabel('Control cost factor /alpha');
ylabel('MSE [mV^2]');
title(sprintf('\\sigma: %g mV/s', sigma));

sgtitle('Control cost balance factor');

save_figure(fig,figpath,"4_ControlCost");


%% Plot functions
function plot_trajectory(time, ref, X, U)
    % Plot reference, state variables and applied control (if present)
    hold on;
    plot(time, ref(time),   'DisplayName','ref [mV]');
    plot(time, X(1,:), '-', 'DisplayName','x_1 [mV]');
    plot(time, X(2,:), '--','DisplayName','x_2 [mV]');
    if ~isempty(U)
        stairs(time, U([1:end end]), ':', 'DisplayName','u [a.u.]');
    end

    xlabel('t [s]');
    ylim([-1.5 1.5]);
end


function plot_X_trajectory(time, X, param)
    % Plot only state variables and limit cycle
    hold on;
    plot(time, X(1,:), '-', 'DisplayName','x_1');
    plot(time, X(2,:), '--','DisplayName','x_2');
    plot(time, ones(size(time))*sqrt(param(2)),'k--','DisplayName','Limit cycle');
    plot(time,-ones(size(time))*sqrt(param(2)),'k--','HandleVisibility','off');

    xlabel('t [s]');
    ylabel('x [mV]');
    ylim([-0.11 0.11]);
end


function plot_phase_trajectory(X1, X2, p, tau, ode)
    %plot first line
    hold on;
    plot(X1(1,:), X1(2,:),'c-','HandleVisibility','off');
    plot(X1(1,1),X1(2,1),'r*')
    plot(X1(1,end),X1(2,end),'bo')

    %plot limit cycle
    th = 0:pi/50:2*pi;
    xunit = sqrt(p(2)) * cos(th);
    yunit = sqrt(p(2)) * sin(th);
    plot(xunit, yunit, 'k--');

    %plot second line
    plot(X2(1,:), X2(2,:),'m-')
    plot(X2(1,1),X2(2,1),'r*')
    plot(X2(1,end),X2(2,end),'bo')

    %plot vector field
    [xx,yy] = meshgrid(-0.12:0.02:0.12,-0.12:0.02:0.12);
    u=zeros(size(xx));
    for ii=1:size(xx,1)
        for jj=1:size(xx,2)
            out = ode(xx(ii,jj),yy(ii,jj),0,p(1),p(2),p(3),p(4),tau);
            u(ii,jj)=out(1);
            v(ii,jj)=out(2);
        end
    end
    quiver(xx,yy,10*u,10*v,'LineWidth',0.8,'MaxHeadSize', 0.4)
    axis(0.12*[-1 1 -1 1]);

    xlabel('x_1 [mV]');
    ylabel('x_2 [mV]');
    title('State space')
    legend('Start Point', 'End Point', 'Limit cycle')
end


function save_figure(fig, figpath, name)
    % Save figure as .eps
    set(fig,'Units','points');
    set(fig,'PaperUnits','points');
    sizeP = get(fig,'Position');

    sizeP = sizeP(3:4);
    set(fig,'PaperSize',sizeP);
    set(fig,'PaperPosition',[0,0,sizeP(1),sizeP(2)]);

    if ~isfolder(figpath), mkdir(figpath); end
    filename = fullfile(figpath,name);

    print(fig,filename,'-depsc','-loose'); % Save figure as .eps file
end