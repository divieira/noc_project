% An implementation of direct multiple shooting
% Joel Andersson, 2016
clear
clc
import casadi.*

T = 1; % Time horizon
fs=500;
N = T/fs; % number of control intervals
tau = 1.0;    % s (arbitrary constant)
k=0.1; %cost of control
a=0.0000;


% parameters
par=[2*pi*6 1.01 -1e1 1e1];
% Declare model variables
y=SX.sym('y');
x=SX.sym('x');
w=SX.sym('w')
a=SX.sym('a');
k_1=SX.sym('k1');
k_2=SX.sym('k2')
X_state = [y; x];

u = SX.sym('u');


%x = [x1; x2];

% Model equations
xdot = ode(X_state,u,[tau,w,a,k_1,k_2]);

% Objective term
x_ref = SX.sym('x_ref');
L = (x_ref-x)^2 + k*u^2;
Loss=Function('Loss', {x,x_ref, u}, {L});


% Formulate discrete time dynamics
p=[u,w, a, k_1, k_2]';
odestruct = struct('x',X_state , 'p', p, 'ode', xdot);
opts = struct('abstol', 1e-8, 'reltol', 1e-8,'tf',1/fs);
F = integrator('F', 'cvodes', odestruct, opts);

%%
N=500;
X_test=zeros(N+1,2);
% Initial parameters
X0 = [ 1 .1];
X_test(1,:)=X0;
% parameters
param=[2*pi*6 .01 -100e1 0e1];
T=N*1/fs;
ts=T/N;
for ii=1:N
    res = F('x0', X_test(ii,:)', 'p', [0, param]);
    X_test(ii+1,:)=full(res.xf);
end
plot(0:ts:T,X_test(:,1))

%%
% Start with an empty NLP
w={};
w0 = [];
lbw = [];
ubw = [];
J = 0;
g={};
lbg = [];
ubg = [];

% "Lift" initial conditions
Xk = MX.sym('X0', 2);
w = {w{:}, Xk};
lbw = [lbw; 1; 0.1];
ubw = [ubw; 1; 0.1];
w0 = [w0; 0; 1];

% Formulate the NLP

%Define Reference
T=N*1/fs;
t_prob=0:1/fs:T;
f_sine=8;
A=0.5;
x_ref=(A+A*heaviside(t_prob-0.55)).*sin(2*pi*f_sine*t_prob);


for k=0:N-1
    % New NLP variable for the control
    Uk = MX.sym(['U_' num2str(k)]);
    w = {w{:}, Uk};
    lbw = [lbw; 0];
    ubw = [ubw;  1];
    w0 = [w0;  0];

    % Integrate till the end of the interval
    Fk = F('x0', Xk, 'p', [Uk, param]);
    Xk_end = Fk.xf;
    %J=J+(full(Xk_end(1))-x_ref(k+1))^2;%;
    J=J+Loss(Xk_end(1),x_ref(k+2),Uk);

    % New NLP variable for state at end of interval
    Xk = MX.sym(['X_' num2str(k+1)], 2);
    w = [w, {Xk}];
    lbw = [lbw; -inf; -inf];
    ubw = [ubw;  inf;  inf];
    w0 = [w0; 0; 0];

    % Add equality constraint
    g = [g, {Xk_end-Xk}];
    lbg = [lbg; 0; 0];
    ubg = [ubg; 0; 0];
end

% Create an NLP solver
prob = struct('f', J, 'x', vertcat(w{:}) , 'g', vertcat(g{:}));
solver = nlpsol('solver', 'ipopt', prob);

% Solve the NLP
sol = solver('x0', w0, 'lbx', lbw, 'ubx', ubw ,...
            'lbg', lbg, 'ubg', ubg);
w_opt = full(sol.x);

% Plot the solution
figure(2)
x1_opt = w_opt(1:3:end);
x2_opt = w_opt(2:3:end);
u_opt = w_opt(3:3:end);
tgrid = linspace(0, T, N+1);
clf;
hold on
plot(0:ts:T,x_ref)
plot(tgrid, x1_opt, '--')
plot(tgrid, x2_opt, '-')
stairs(tgrid, [u_opt; nan], '-.')
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
