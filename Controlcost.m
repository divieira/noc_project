%Plots control energy - MSE
clear
clc
close all
import casadi.*

%% Parameters
% Simulation
fs = 120;       % Hz
T = 5;          % s
N = T*fs;       % steps
ts = 1/fs;      % s
x0 = [1; 0];    % initial conditions

% Model
param = [2*pi*6 .01 -1e3 0]; % [w, a, k1, k2]
tau = 1.0;      % s (arbitrary stiffness constant)
sigma = 0.1;    % disturbance on each simulation step

% Objective
k = 1000.0;     % control cost

% MPC
shift = 1;  % MPC interval
N_mpc = 10; % MPC horizon

% Reference
f_ref = 8;      % Hz
a_ref = .5;     % mV
t2_ref = 2.5;   % s
a2_ref = .3;    % mV
ref = @(t) a_ref*sin(2*pi*f_ref*t) + a2_ref*heaviside(t-t2_ref).*sin(2*pi*f_ref*t);


%% Model definition
% Declare model variables
x1=SX.sym('x1');
x2=SX.sym('x2');
w=SX.sym('w');
a=SX.sym('a');
k_1=SX.sym('k1');
k_2=SX.sym('k2');

x = [x1; x2];
p = [w, a, k_1, k_2]';
u = SX.sym('u');

% Model equations
xdot = ode(x,u,[tau,w,a,k_1,k_2]);


%% Run MPC Simulation optimize k
rng default; % Fix RNG for reproducibility
k = logspace(-1,4,6);
MSE=zeros(size(k,2),1);
MeanControlEnergy=zeros(size(k,2),1);
time = ts*(0:N);
for jj=1:size(k,2)
    % Objective term
    t = SX.sym('t');
    L = (ref(t)-x1)^2 + k(jj)*u^2;
    % Formulate discrete time dynamics
    F = rk4integrator(x, p, u, t, xdot, L, 1/fs);
    [X_applied, U_applied] = MPC(F, x0, param, sigma, N, N_mpc, shift, ts);
    MeanControlEnergy(jj)=mean(U_applied.^2);
    MSE(jj)=mean((X_applied(1,:)-ref(time)).^2);
end
%toc

% Plot the solution
figure(1);
subplot(2,1,1)
semilogx(k,MeanControlEnergy)
xlabel('Control cost factor k')
ylabel('Mean control energy [arb. Unit^2]')
subplot(2,1,2)
semilogx(k,MSE)
xlabel('Control cost factor k')
ylabel('MSE [mV^2]')
title("Noise \sigma^2=" + sigma^2 + "(mV)^2")
sgtitle('Optimize k - cost of control energy')
print('OptimizeK','-depsc','-tiff')

%% plot Multiple Shooting
%parameters
sigma = 0.0;    % disturbance on each simulation step
% MPC
shift = N;  % MPC interval
N_mpc = N; % MPC horizon

% Objective
k = 100.0;     % control cost
L = (ref(t)-x1)^2 + k*u^2;
% Formulate discrete time dynamics
F = rk4integrator(x, p, u, t, xdot, L, 1/fs);

% Run MPC Simulation without noise
rng default; % Fix RNG for reproducibility

[X_applied, U_applied] = MPC(F, x0, param, sigma, N, N_mpc, shift, ts);

% Plot the solution
figure(2);
subplot(2,1,1)
hold on
time = ts*(0:N);
plot(time, ref(time))
plot(time, X_applied(1,:), '-')
plot(time, X_applied(2,:), '--')
stairs(time, U_applied([1:N N]), '-.')
xlabel('t [s]')
legend('x_{ref} [mV]','x1 [mV]','x2 [mV]','u [arb. Unit]')
title({"Noise \sigma^2=" + sigma^2 + "(mV)^2", "MSE= " + num2str((mean((X_applied(1,:)-ref(time)).^2)),'%10.5e\n') + "(mV)^2"})


% Run MPC Simulation with noise
sigma = 0.1;    % disturbance on each simulation step
[X_applied, U_applied] = MPC(F, x0, param, sigma, N, N_mpc, shift, ts);

% Plot the solution
figure(2);
subplot(2,1,2)
hold on
time = ts*(0:N);
plot(time, ref(time))
plot(time, X_applied(1,:), '-')
plot(time, X_applied(2,:), '--')
stairs(time, U_applied([1:N N]), '-.')
xlabel('t [s]')
legend('x_{ref} [mV]','x1 [mV]','x2 [mV]','u [arb. Unit]')
title({"Noise \sigma^2=" + sigma^2 + "(mV)^2", "MSE= " + num2str((mean((X_applied(1,:)-ref(time)).^2)),'%10.5e\n')+ "(mV)^2"})
sgtitle('Multiple shooting')
print('MultipleShooting','-depsc','-tiff')


%% plot MPC
%parameters
sigma = 0.0;    % disturbance on each simulation step
% MPC
shift = 1;  % MPC interval
N_mpc = 10; % MPC horizon

% Objective
k = 100.0;     % control cost
L = (ref(t)-x1)^2 + k*u^2;
% Formulate discrete time dynamics
F = rk4integrator(x, p, u, t, xdot, L, 1/fs);

% Run MPC Simulation without noise
rng default; % Fix RNG for reproducibility

[X_applied, U_applied] = MPC(F, x0, param, sigma, N, N_mpc, shift, ts);

% Plot the solution
figure(3);
subplot(2,1,1)
hold on
time = ts*(0:N);
plot(time, ref(time))
plot(time, X_applied(1,:), '-')
plot(time, X_applied(2,:), '--')
stairs(time, U_applied([1:N N]), '-.')
xlabel('t [s]')
legend('x_{ref} [mV]','x1 [mV]','x2 [mV]','u [arb. Unit]')
title({"Noise \sigma^2=" + sigma^2 + "(mV)^2", "MSE= " + num2str((mean((X_applied(1,:)-ref(time)).^2)),'%10.5e\n') + "(mV)^2"})


% Run MPC Simulation with noise
sigma = 0.1;    % disturbance on each simulation step
[X_applied, U_applied] = MPC(F, x0, param, sigma, N, N_mpc, shift, ts);

% Plot the solution
figure(3);
subplot(2,1,2)
hold on
time = ts*(0:N);
plot(time, ref(time))
plot(time, X_applied(1,:), '-')
plot(time, X_applied(2,:), '--')
stairs(time, U_applied([1:N N]), '-.')
xlabel('t [s]')
legend('x_{ref} [mV]','x1 [mV]','x2 [mV]','u [arb. Unit]')
title({"Noise \sigma^2=" + sigma^2 + "(mV)^2", "MSE= " + num2str((mean((X_applied(1,:)-ref(time)).^2)),'%10.5e\n')+ "(mV)^2"})
sgtitle('Model Predictive Control')
print('MPC','-depsc','-tiff')

%% Function definitions
function f = ode(X, u, p)
    % State transition function f, specified as a function handle.
    % The function calculates the Ns-element state vector of the system at
    % time step k, given the state vector at time step k-1.
    % Ns is the number of states of the nonlinear system.
    y = X(1);
    x = X(2);

    tau=p(1);
    w=p(2); 
    a=p(3); 
    k1=p(4); 
    k2=p(5);
    
    dy =  x*w + y*(a - x^2 - y^2)/tau + k1*u;
    dx = -y*w + x*(a - x^2 - y^2)/tau + k2*u;
    
   f=[dy;dx];
end


