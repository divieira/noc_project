% Run MPC simulation

import casadi.*


%% Global parameters
% Simulation
fs = 160;       % Hz (sampling frequency)
ts = 1/fs;      % s (time step)
x0 = [0; 0.1];  % mV (initial conditions)

% Model parameters (nominal values)
param = [2*pi*6; .01; 1e2; 0];  % [w; a; k1; k2]
tau = 10;                       % s (stiffness constant)
sigma = 10;                     % mV/s (for state noise simulations)

% Reference (for multiple shooting and MPC)
f_ref = 8;      % Hz (reference sine frequency)
a_ref = .5;     % mV (reference sine amplitude)

% Control cost (for multiple shooting and MPC)
alpha = .01;    % control cost factor

% MPC parameters
shift = 5;  % timesteps (MPC interval)
N_mpc = 20; % timesteps (MPC horizon)

% Simulation parameters
T = 1;      % s (simulation period)
N = T*fs;   % simulation steps

% Parameters for CASADI nlpsol
nlpsol_opts = struct;
%options.ipopt.max_iter = 6;            %Set maximum iterations
nlpsol_opts.ipopt.print_level=0;        %No printing of evaluations
nlpsol_opts.print_time= 0;              %No printing of time
nlpsol_opts.ipopt.linear_solver='ma27'; %Requires HLS library
nlpsol_opts.jit=true;                   %Enable JIT compilation


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
    -x2.*w + x1.*(1 - (x1.^2 + x1.^2)./a)./tau + k1.*u; ...
     x1.*w + x2.*(1 - (x1.^2 + x2.^2)./a)./tau + k2.*u ];

xdot = ode(x1,x2,u,w,a,k1,k2,tau);


%% Optimal control problem
% Objective
t = SX.sym('t');
ref = @(t) a_ref*sin(2*pi*f_ref*t); % reference (defined as a function)
L = (ref(t)-x1)^2 + alpha*u^2;      % objective function

% Formulate discrete time dynamics for state and objective functions using
% a single-step Runge-Kutta 4th order method integrator at each timestep
F = rk4integrator(x, p, u, t, xdot, L, 1/fs);


%% Run MPC
rng default; % Fix RNG for reproducibility
[X,U,timings] = MPC(F,x0,param,sigma,N,N_mpc,shift,ts,[],nlpsol_opts);
fprintf('Simulation run time: %g s (%g ± %g ms per MPC evaluation)\n', ...
        sum(timings), 1e3*mean(timings), 1e3*std(timings));


%% Plot trajectory
%set(0,'defaultFigureRenderer','painters');  % fix dashed lines not showing
figure;
hold on;
time = ts*(0:N);
plot(time, ref(time),             'DisplayName','ref [mV]');
plot(time, X(1,:),                'DisplayName','x_1 [mV]');
plot(time, X(2,:), '--',          'DisplayName','x_2 [mV]');
stairs(time, U([1:end end]), ':', 'DisplayName','u [a.u.]');
legend;
xlabel('t [s]');
