clear
clc
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
k = .1;     % control cost

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

% Objective term
t = SX.sym('t');
L = (ref(t)-x1)^2 + k*u^2;

% Formulate discrete time dynamics
F = rk4integrator(x, p, u, t, xdot, L, 1/fs);


%% Run MPC Simulation
rng default; % Fix RNG for reproducibility
tic
[X_applied, U_applied] = MPC(F, x0, param, sigma, N, N_mpc, shift, ts);
toc

%% Plot the solution
figure;
hold on
time = ts*(0:N);
plot(time, ref(time))
plot(time, X_applied(1,:), '-')
plot(time, X_applied(2,:), '--')
stairs(time, U_applied([1:N N]), '-.')
xlabel('t')
legend('x_ref','x1','x2','u')


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


