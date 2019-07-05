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
X0 = [ 0 .1];

% parameters
par=[2*pi*6 .01 -1e1 1e1];

y=SX.sym('y');
x=SX.sym('x');
w=SX.sym('w')
a=SX.sym('a');
k1=SX.sym('k1');
k2=SX.sym('k2')
X = [y; x];

u = SX.sym('u');


tf = SX.sym('tf');
fun=ode(X,u,[tau,w,a,k1,k2]).*tf;



p=[u,tf,w, a, k1, k2]';
odestruct = struct('x', X, 'p', p, 'ode', fun);
opts = struct('abstol', 1e-8, 'reltol', 1e-8);
F = integrator('F', 'cvodes', odestruct, opts);


N=200;
X_test=zeros(N+1,2)
X_test(1,:)=X0;

T=2;
ts=T/N;
for ii=1:200
    res = F('x0', X_test(ii,:)', 'p', [0, ts, par]);
    X_test(ii+1,:)=full(res.xf)
end
plot(X_test(:,1))




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