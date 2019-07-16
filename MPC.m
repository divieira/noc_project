function [X_applied, U_applied] = MPC(F, x0, param, sigma, N, N_mpc, shift, ts)
% See: CasADi example direct_multiple_shooting.m
import casadi.*

%% MPC Simulation
X_applied=x0;
U_applied=[];
%Init for warm start X, u =0
x1_opt=zeros(N_mpc+shift,1);
x2_opt=zeros(N_mpc+shift,1);
u_opt=zeros(N_mpc+shift,1);

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
    x1_opt=[x1_opt; zeros(shift,1)];
    x2_opt=[x2_opt; zeros(shift,1)];
    u_opt=[u_opt; zeros(shift,1)];
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
    %options.ipopt.max_iter = 3; %FIXME: really necessary?
    solver = nlpsol('solver', 'ipopt', prob, options);

    % Solve the NLP
    sol = solver('x0',w0, 'lbx',lbw, 'ubx',ubw, 'lbg',lbg, 'ubg',ubg);
    w_opt = full(sol.x);
    x1_opt=w_opt(1:3:end);
    x2_opt=w_opt(2:3:end);
    u_opt = w_opt(3:3:end);
    for k=0:shift-1
        U_applied = [U_applied, u_opt(k+1)];
        Fk = F('x0',X_applied(:,end), 'p',param, 'u',U_applied(end), 't',ts*(i+k));
        X_applied = [X_applied, full(Fk.xf)+normrnd(0,sigma,[2,1])];
    end
end
end