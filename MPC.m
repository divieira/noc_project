function [X, U, timings] = MPC(F, x0, param, sigma, N, N_mpc, shift, ts, param_sim, nlpsol_opts, warm_start)
% [X, U, timings] = MPC(F, x0, param, sigma, N, N_mpc, shift, ts, [param_sim], [nlpsol_opts], [warm_start])
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
%   [param_sim]     separate time-varying parameters for simulation (optional)
%   [nlpsol_opts]   options struct for nlpsol functions (optional)
%   [warm_start]    warm start mode (0 [default]: none, ±1: repeat, ±2: shift, negative: w0 only)
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

if ~exist('warm_start','var')
    % Default: no warm starting
    warm_start = 0;
end

if (warm_start > 0 &&                                           ...
    (~isfield(nlpsol_opts, 'ipopt') ||                      ...
     ~isfield(nlpsol_opts.ipopt, 'warm_start_init_point')))
    % Set warm start option if applicable (but do not override if supplied)
    nlpsol_opts.ipopt.warm_start_init_point = 'yes';
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
w_opt     = zeros(nx+N_mpc*(nx+nu),1);
lam_w_opt = zeros(nx+N_mpc*(nx+nu),1);
lam_g_opt = zeros(nx*N_mpc,1);
N_cycles = floor(N/shift);
timings = zeros(N_cycles,1);
p = 1;
for i = 1:N_cycles
    % Set starting time for 'toc'
    tic;

    % Initial guess for w0 (and lambda multipliers if warm starting)
    switch warm_start
        case 0  % none
            % Only set x0 in initial guess
            w0      = [X(:,p);    zeros(N_mpc*(nx+nu),1)];
            lam_args = {};
        case 1  % repeat
            % Use optimal values from previous evaluation
            w0      = w_opt;
            lam_w0  = lam_w_opt;
            lam_g0  = lam_g_opt;
            lam_args = { 'lam_x0',lam_w0, 'lam_g0',lam_g0 };
        case -1 % repeat w0 only
            w0      = w_opt;
            lam_args = {};
        case 2  % shift
            % Shift optimal values from previous evaluation (clamping last)
            w0      = [    w_opt(shift*(nx+nu)+1:end); repmat(    w_opt(end-(nx+nu)+1:end),shift,1)];
            lam_w0  = [lam_w_opt(shift*(nx+nu)+1:end); repmat(lam_w_opt(end-(nx+nu)+1:end),shift,1)];
            lam_g0  = [lam_g_opt(shift*(nx   )+1:end); repmat(lam_g_opt(end-(nx   )+1:end),shift,1)];
            lam_args = { 'lam_x0',lam_w0, 'lam_g0',lam_g0 };
        case -2 % shift w0 only
            w0      = [    w_opt(shift*(nx+nu)+1:end); repmat(    w_opt(end-(nx+nu)+1:end),shift,1)];
            lam_args = {};
    end

    % Variable constraints
    lbw = [X(:,p); repmat([zeros(nu,1); -inf(nx,1)], N_mpc,1)];
    ubw = [X(:,p); repmat([ ones(nu,1);  inf(nx,1)], N_mpc,1)];

    % Solve the NLP
    t0 = (i-1)*shift*ts;
    sol = solver('x0',w0, lam_args{:}, 'lbx',lbw, 'ubx',ubw, 'lbg',lbg, 'ubg',ubg, 'p',t0);
    w_opt = full(sol.x);
    lam_w_opt = full(sol.lam_x);
    lam_g_opt = full(sol.lam_g);
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
