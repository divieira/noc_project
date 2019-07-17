function [X_applied, U_applied] = MPC(F, x0, param, sigma, N, N_mpc, shift, ts,varargin)
% See: CasADi example direct_multiple_shooting.m
import casadi.*
narginchk(8,9);
%dfine paramlist of simulation, if no param is defined use same as for MPC
%otherwise use array which contains in every row the value for that
%simulation step
if nargin~=9
    param2=repmat(param,N,1);
else
    param2=varargin{1};
end


%% MPC Simulation
X_applied=x0;
U_applied=[];
%Init for warm start X, u =0
x1_opt=zeros(N_mpc,1);
x2_opt=zeros(N_mpc,1);
u_opt=zeros(N_mpc,1);

for i = 0:shift:N-1
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
    lbw = [lbw; X_applied(:,end)];
    ubw = [ubw; X_applied(:,end)];
    w0  = [w0;  X_applied(:,end)];

    %Create values for warmstart
    x1_opt(end+1:end+shift) = x1_opt(end);
    x2_opt(end+1:end+shift) = x2_opt(end);
    u_opt(end+1:end+shift)  = u_opt(end);
    for k=0:N_mpc-1
        % New NLP variable for the control
        Uk = MX.sym(['U_' num2str(k)]);
        w = {w{:}, Uk};
        lbw = [lbw; -1];
        ubw = [ubw;  1];
        w0 = [w0;  u_opt(shift+1+k)];

        % Integrate till the end of the interval
        Fk = F('x0',Xk, 'p',param, 'u',Uk, 't',ts*(i+k));
        Xk_end = Fk.xf;
        J = J+Fk.qf;

        % New NLP variable for state at end of interval
        Xk = MX.sym(['X_' num2str(k+1)], 2);
        w = [w, {Xk}];
        lbw = [lbw; -inf; -inf];
        ubw = [ubw;  inf;  inf];
        w0 = [w0; x1_opt(shift+1+k); x2_opt(shift+1+k)];

        % Add equality constraint
        g = [g, {Xk_end-Xk}];
        lbg = [lbg; 0; 0];
        ubg = [ubg; 0; 0];
    end

    % Create an NLP solver
    prob = struct('f', J, 'x', vertcat(w{:}) , 'g', vertcat(g{:}));
    options = struct;
    options.ipopt.max_iter = 6; %Set maximum Iterations - normally 4 Iterations are enough
    options.ipopt.print_level=0;
    solver = nlpsol('solver', 'ipopt', prob, options);

    % Solve the NLP
    sol = solver('x0',w0, 'lbx',lbw, 'ubx',ubw, 'lbg',lbg, 'ubg',ubg);
    w_opt = full(sol.x);
    x1_opt=w_opt(1:3:end);
    x2_opt=w_opt(2:3:end);
    u_opt = w_opt(3:3:end);
    for k=0:shift-1
        U_applied = [U_applied, u_opt(k+1)];
        Fk = F('x0',X_applied(:,end), 'p',param2(1+i+k,:), 'u',U_applied(end), 't',ts*(i+k));
        X_applied = [X_applied, full(Fk.xf)+normrnd(0,sigma,[2,1])];
    end
end
end