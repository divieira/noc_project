% See: CasADi example direct_multiple_shooting.m
clear
clc
import casadi.*

%% Parameters
% Simulation
fs = 120;       % Hz
T = 5;          % s
N = T*fs;       % steps
ts = 1/fs;      % s
t = ts*(1:N);   % time vector
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
ref = @(k) a_ref*sin(2*pi*f_ref*k/fs) + a2_ref*heaviside(k/fs-t2_ref).*sin(2*pi*f_ref*k/fs);


%% Model definition
% Declare model variables
x1=SX.sym('x1');
x2=SX.sym('x2');
w=SX.sym('w');
a=SX.sym('a');
k_1=SX.sym('k1');
k_2=SX.sym('k2');

x = [x1; x2];
u = SX.sym('u');

% Model equations
xdot = ode(x,u,[tau,w,a,k_1,k_2]);

% Objective term
x_ref = SX.sym('x_ref');
L = (x_ref-x1)^2 + k*u^2;

% Formulate discrete time dynamics
%p=[u, w, a, k_1, k_2]';
%odestruct = struct('x',X_state , 'p', p, 'ode', xdot);
%opts = struct('abstol', 1e-8, 'reltol', 1e-8,'tf',1/fs);
%F = integrator('F', 'cvodes', odestruct, opts);
%Loss=Function('Loss', {x,x_ref, u}, {L});
p = [x_ref, u, w, a, k_1, k_2]';
F = rk4integrator(x, p, xdot, L, 1/fs);


%% MPC Simulation
import casadi.*
U_applied=[];
X_applied=[];
X_generated=x0;   %initial conditions
i_next_mpc = 1;
for i = 1:shift:N
    %% Formulate NLP
    % Start with an empty NLP
    w={};
    w0 = [];
    lbw = [];
    ubw = [];
    J = 0;
    g = {};
    lbg = [];
    ubg = [];

    % "Lift" initial conditions
    Xk = MX.sym('X0', 2);
    w = {w{:}, Xk};
    lbw = [lbw; X_generated(1); X_generated(2)];
    ubw = [ubw; X_generated(1); X_generated(2)];
    w0 = [w0; X_generated(1); X_generated(2)];

    %TODO: initialize w0 with previous guess
    for k=0:N_mpc-1
        % New NLP variable for the control
        Uk = MX.sym(['U_' num2str(k)]);
        w = {w{:}, Uk};
        lbw = [lbw; -1];
        ubw = [ubw;  1];
        w0 = [w0;  0];

        % Integrate till the end of the interval
        Fk = F('x0', Xk, 'p', [ref(i+k), Uk, param]);
        Xk_end = Fk.xf;
        J = J+Fk.qf;

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
    options = struct;
    %options.ipopt.max_iter = 3; %FIXME: really necessary?
    solver = nlpsol('solver', 'ipopt', prob, options);

    % Solve the NLP
    sol = solver('x0', w0, 'lbx', lbw, 'ubx', ubw ,...
                'lbg', lbg, 'ubg', ubg);
    w_opt = full(sol.x);

    u_opt = w_opt(3:3:end);
    for k=0:shift-1
        U_applied = [U_applied, u_opt(k+1)];
        X_generated = F('x0', X_generated, 'p', [ref(i+k), U_applied(end), param]);
        X_generated = (full(X_generated.xf))+normrnd(0,sigma,[2,1]);
        X_applied = [X_applied, X_generated];
    end
end

% Plot the solution
figure;
hold on
plot(t, ref(1:N))
plot(t, X_applied(1,:), '-')
plot(t, X_applied(2,:), '--')
stairs(t, U_applied, '-.')
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


function F = rk4integrator(x, p, xdot, L, T)
   % Fixed step Runge-Kutta 4 integrator
   M = 1;%4; % RK4 steps per interval
   DT = T/M;
   f = casadi.Function('f', {x, p}, {xdot, L});
   X0 = x;%MX.sym('X0', 2);
   P = p;%MX.sym('U');
   X = X0;
   Q = 0;
   for j=1:M
       [k1, k1_q] = f(X, P);
       [k2, k2_q] = f(X + DT/2 * k1, P);
       [k3, k3_q] = f(X + DT/2 * k2, P);
       [k4, k4_q] = f(X + DT * k3, P);
       X=X+DT/6*(k1 +2*k2 +2*k3 +k4);
       Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q);
    end
    F = casadi.Function('F', {X0, P}, {X, Q}, {'x0','p'}, {'xf', 'qf'});
end