function [X, U, timings] = MPC(F, x0, param, sigma, N, N_mpc, shift, ts, param_sim, nlpsol_opts)
% [X, U, timings] = MPC(F, x0, param, sigma, N, N_mpc, shift, ts, [param_sim], [nlpsol_opts])
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
%   [nlpsol_opts] options struct for nlpsol functions (optional)
% Returns:
%   X       DxN array with simulated state trajectory
%   U       1xN array with applied control trajectory
%   timings (N/shift)x1 array of elapsed time in each MPC evaluation
%
% See: CasADi example direct_multiple_shooting.m

import casadi.*

% Optional arguments
if ~exist('param_sim','var') || isempty(param_sim)
    % If not provided, use the parameters used by the controller in simulation
    param_sim = repmat(param,1,N);
end

if ~exist('nlpsol_opts','var')
    % Default parameters for CASADI nlpsol
    nlpsol_opts = struct;
    %options.ipopt.max_iter = 6;            %Set maximum iterations - normally 4 iterations are enough
    nlpsol_opts.ipopt.print_level=0;        %No printing of evaluations
    nlpsol_opts.print_time= 0;              %No printing of time
    %nlpsol_opts.ipopt.linear_solver='ma27'; %Requires HLS library
    %nlpsol_opts.jit=true;                   %Enable JIT compilation
end


%% Formulate NLP once
% Start with an empty NLP
nx = length(x0);
nu = 1;
J = 0;
g = {};

% "Lift" initial conditions
Xk = SX.sym('Xk', nx);
w = {Xk};

% Add shootings for each timestemp in the horizon
t0 = SX.sym('t0', 1);
for k=0:N_mpc-1
    % New NLP variable for the control
    Uk = SX.sym(['U_' num2str(k)]);
    w = [w, {Uk}];

    % Integrate till the end of the interval
    Fk = F('x0',Xk, 'p',param, 'u',Uk, 't',t0+(ts*k));
    Xk_end = Fk.xf;
    J = J+Fk.qf;

    % New NLP variable for state at end of interval
    Xk = SX.sym(['X_' num2str(k+1)], nx);
    w = [w, {Xk}];

    % Add equality constraint
    g = [g, {Xk_end-Xk}];
end
lbg = zeros(nx*N_mpc,1);
ubg = zeros(nx*N_mpc,1);

% Create an NLP solver (with time index ik as parameter)
prob = struct('f', J, 'x', vertcat(w{:}) , 'g', vertcat(g{:}),'p',t0);
solver = nlpsol('solver', 'ipopt', prob, nlpsol_opts);

%% Run MPC Simulation
X = [x0 zeros(nx,N)];
U = zeros(nu,N);
w_opt=zeros(nx+N_mpc*3,1);
N_cycles = floor(N/shift);
timings = zeros(N_cycles,1);
p = 1;
for i = 1:N_cycles
    % Set starting time for 'toc'
    tic;

    % Extend values for warm starting
    w_opt(end+1:end+shift*(nx+nu)) = repmat(w_opt(end-(nx+nu)+1:end),shift,1);

    % Initial guess (warm start by shifting previous solution)
    w0  = [X(:,p); w_opt(nx+shift*(nx+nu)+1:end)];

    % Variable constraints
    lbw = [X(:,p); repmat([0; -inf(nx,1)], N_mpc,1)];
    ubw = [X(:,p); repmat([1;  inf(nx,1)], N_mpc,1)];

    % Solve the NLP
    t0 = (i-1)*shift*ts;
    sol = solver('x0',w0, 'lbx',lbw, 'ubx',ubw, 'lbg',lbg, 'ubg',ubg,'p',t0);
    w_opt = full(sol.x);
    u_opt = w_opt(nx+1:nx+1:end);

    % Apply control steps to plant
    for k=0:shift-1
        U(:,p) = u_opt(k+1);
        Fk = F('x0',X(:,p), 'p',param_sim(:,1+i+k), 'u',U(:,p), 't',t0+ts*k);
        p = p+1;
        X(:,p) = full(Fk.xf)+normrnd(0,ts*sigma,[2,1]);
    end

    % Save timing for current iteration
    timings(i) = toc;
end
end
