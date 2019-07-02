clear variables
close all
clc

import casadi.*
% Problem definition

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
tau = 1.0;    % s (arbitrary constant)

% Initial parameters
X0 = [ 0 .1 2*pi*6 .01 -1e1 1e1 ];

y=SX.sym('y');
x=SX.sym('x');
w=SX.sym('w')
a=SX.sym('a');
k1=SX.sym('k1');
k2=SX.sym('k2')
X = [y; x; w; a; k1; k2];

u = SX.sym('u');

fun=ode(X,u,tau)

tf = SX.sym('tf');
odestruct = struct('x', X, 'p', u, 'ode', fun);
opts = struct('abstol', 1e-8, 'reltol', 1e-8);
F = integrator('F', 'cvodes', odestruct, opts);



res = F('x0', X0, 'p', 0)




%% Function definitions
function f = ode(X, u, p)
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
    tau=p(1);
    
    dy =  x*w + y*(a - x^2 - y^2)/tau + k1*u;
    dx = -y*w + x*(a - x^2 - y^2)/tau + k2*u;
    dw = 0;
    da = 0;
    dk1 = 0;
    dk2 = 0;
   
    
   f=[dy;dx;dw;da;dk1;dk2];
end