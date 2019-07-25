function [X, U] = MPC(F, x0, param, sigma, N, N_mpc, shift, ts, varargin)
% [X, U] = MPC(F, x0, param, sigma, N, N_mpc, shift, ts, [param_sim])
% Run a model predictive control simulation.
% Inputs:
%   F       integrator function receiving (x0,p,u,t) as parameters
%   x0      initial state values [Dx1]
%   param   parameters passed to F
%   sigma   Gaussian state noise in ODE
%   N       timesteps for simulation
%   N_mpc   optimization horizon
%   shift   evaluation interval
%   ts      timestep size
%   [param_sim] separate time-varying parameters for simulation (optional)
% Returns:
%   X       DxN array with simulated state trajectory
%   U       1xN array with applied control trajectory
%
% See: CasADi example direct_multiple_shooting.m

import casadi.*

% Optional argument
% If not provided, use the parameters used by the controller in simulation
narginchk(8,9);
if nargin < 9
    param_sim = repmat(param,1,N);
else
    param_sim = varargin{1};
end


%% Formulate NLP once
% Start with an empty NLP
nx = length(x0);
nu = 1;
J = 0;
g = {};

% "Lift" initial conditions
Xk = MX.sym('Xk', nx);
w = {Xk};

% Add shootings for each timestemp in the horizon
t0 = MX.sym('t0', 1);
for k=0:N_mpc-1
    % New NLP variable for the control
    Uk = MX.sym(['U_' num2str(k)]);
    w = {w{:}, Uk};

    % Integrate till the end of the interval
    Fk = F('x0',Xk, 'p',param, 'u',Uk, 't',t0+(ts*k));
    Xk_end = Fk.xf;
    J = J+Fk.qf;

    % New NLP variable for state at end of interval
    Xk = MX.sym(['X_' num2str(k+1)], nx);
    w = [w, {Xk}];

    % Add equality constraint
    g = [g, {Xk_end-Xk}];
end
lbg = zeros(nx*N_mpc,1);
ubg = zeros(nx*N_mpc,1);

% Create an NLP solver (with time index ik as parameter)
prob = struct('f', J, 'x', vertcat(w{:}) , 'g', vertcat(g{:}),'p',t0);
options = struct;
%options.ipopt.max_iter = 6;     %Set maximum iterations - normally 4 iterations are enough
options.ipopt.print_level=0;     %No printing of evaluations
options.print_time= 0;           %No printing of time
solver = nlpsol('solver', 'ipopt', prob, options);

%% Run MPC Simulation
X = x0;
U = [];
w_opt=zeros(nx+N_mpc*3,1);
for i = 0:shift:N-1
    % Extend values for warm starting
    w_opt(end+1:end+shift*(nx+nu)) = repmat(w_opt(end-(nx+nu)+1:end),shift,1);

    % Initial guess (warm start by shifting previous solution)
    w0  = [X(:,end); w_opt(nx+shift*(nx+nu)+1:end)];

    % Variable constraints
    lbw = X(:,end);
    ubw = X(:,end);
    for k=0:N_mpc-1
        lbw = [lbw; 0];
        ubw = [ubw; 1];
        lbw = [lbw; -inf(nx,1)];
        ubw = [ubw;  inf(nx,1)];
    end

    % Solve the NLP
    sol = solver('x0',w0, 'lbx',lbw, 'ubx',ubw, 'lbg',lbg, 'ubg',ubg,'p',i*ts);
    w_opt = full(sol.x);
    u_opt = w_opt(nx+1:nx+1:end);

    % Apply control steps to plant
    for k=0:shift-1
        U = [U, u_opt(k+1)];
        Fk = F('x0',X(:,end), 'p',param_sim(:,1+i+k), 'u',U(end), 't',ts*(i+k));
        X = [X, full(Fk.xf)+normrnd(0,ts*sigma,[2,1])];
    end
end
end
