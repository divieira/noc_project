% An implementation of direct multiple shooting
% Joel Andersson, 2016
clear
clc
import casadi.*

% parameters
param=[2*pi*6 .01 -100e1 0e1];
fs=120;
tau = 1.0;    % s (arbitrary constant)
k=1.; %cost of control

%%simulation / MPC
sigma=0.4;  %disturbance on each simulation step
Control_cyc=500;
shift=1;%Shift of MPC
N=10;   %Horizon Optimal control

%Ref
f_sine=8;
A=1.5;
B=-0.5;
C=1.5;

% parameters
par=[2*pi*6 1.01 -1e1 0e1];
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


U_applied=[];
X_applied=[];
X_generated=[1, 0.1];   %initial conditions
for jj=0:round(Control_cyc/shift)
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
    lbw = [lbw; X_generated(1); X_generated(2)];
    ubw = [ubw; X_generated(1); X_generated(2)];
    w0 = [w0; X_generated(1); X_generated(2)];

    % Formulate the NLP

    %Define Reference
    T=N*1/fs;
    t_prob=0+jj*shift*1/fs:1/fs:T+jj*shift*1/fs;
    x_ref=f_ref(t_prob,A,B,C,f_sine);
   % figure(3)
   % plot(t_prob,x_ref)
   % hold on


    for k=0:N-1
        % New NLP variable for the control
        Uk = MX.sym(['U_' num2str(k)]);
        w = {w{:}, Uk};
        lbw = [lbw; -1];
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
    options=struct;
    options.ipopt.max_iter = 3;
    solver = nlpsol('solver', 'ipopt', prob, options);

    % Solve the NLP
    sol = solver('x0', w0, 'lbx', lbw, 'ubx', ubw ,...
                'lbg', lbg, 'ubg', ubg);
    w_opt = full(sol.x);

    u_opt = w_opt(3:3:end);
    for kk=1:shift
        U_applied=[U_applied, u_opt(kk)];
        X_generated=F('x0', X_generated, 'p', [U_applied(end), param]);
        X_generated=(full(X_generated.xf))+normrnd(0,sigma,[2,1]);
        X_applied=[X_applied, X_generated];
    end
end

% Plot the solution
figure(2)

tgrid = linspace(0, 1/fs*length(U_applied), length(U_applied));
x_ref=f_ref(tgrid,A,B,C,f_sine);
clf;
hold on
plot(tgrid,x_ref)
plot(tgrid, X_applied(1,:), '--')
plot(tgrid, X_applied(2,:), '-')
stairs(tgrid, [U_applied], '-.')
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

function x=f_ref(t_prob, A,B,C,f_sine)
    x=A*sin(2*pi*f_sine*t_prob)+B*t_prob.*sin(2*pi*f_sine*t_prob)+C*heaviside(t_prob-0.543).*sin(2*pi*f_sine*t_prob+1/2*pi);
end