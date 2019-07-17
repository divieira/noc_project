%Plots control energy - MSE
clear
clc
close all
import casadi.*

%% Parameters
%many are redifined in the simulations to get a specific behavior or speed
% Simulation
fs = 120;       % Hz
T = 2.5;          % s
N = T*fs;       % steps
ts = 1/fs;      % s
x0 = [1; 0];    % initial conditions

% Model
param = [2*pi*6 .01 -1e3 0]; % [w, a, k1, k2]
tau = 1.0;      % s (arbitrary stiffness constant)
sigma = 0.0;    % disturbance on each simulation step

% Objective
k = 1000.0;     % control cost

% MPC
shift = 1;  % MPC interval
N_mpc = 10; % MPC horizon

% Reference
f_ref = 8;      % Hz
a_ref = .5;     % mV
t2_ref = 2.5;   % s
a2_ref = .3;    % mV
ref = @(t) a_ref*sin(2*pi*f_ref*t) + a2_ref*heaviside(t-t2_ref).*sin(2*pi*f_ref*t);

% Plots
FontSTitle=11;
FontSAxis=12;
FontSSGTitle=14;
FontSLabel=10;
set(0,'DefaultLineLineWidth',2)

%% Model definition
% Declare model variables
x1=SX.sym('x1');
x2=SX.sym('x2');
w=SX.sym('w');
a=SX.sym('a');
k_1=SX.sym('k1');
k_2=SX.sym('k2');

x = [x1; x2];
p = [w, a, k_1, k_2]';
u = SX.sym('u');

% Model equations
xdot = ode(x,u,[tau,w,a,k_1,k_2]);


%% Run MPC Simulation optimize k
T = 2;          % s
N = T*fs;       % steps

% Reference
f_ref = 8;      % Hz
a_ref = .5;     % mV
t2_ref = 2.5;   % s
a2_ref = .0;    % mV
ref = @(t) a_ref*sin(2*pi*f_ref*t) + a2_ref*heaviside(t-t2_ref).*sin(2*pi*f_ref*t);


rng default; % Fix RNG for reproducibility
k = logspace(-1,4,12);
MSE=zeros(size(k,2),1);
MeanControlEnergy=zeros(size(k,2),1);
time = ts*(0:N);
for jj=1:size(k,2)
    % Objective term
    t = SX.sym('t');
    L = (ref(t)-x1)^2 + k(jj)*u^2;
    % Formulate discrete time dynamics
    F = rk4integrator(x, p, u, t, xdot, L, 1/fs);
    [X_applied, U_applied] = MPC(F, x0, param, sigma, N, N_mpc, shift, ts);
    MeanControlEnergy(jj)=mean(U_applied.^2);
    MSE(jj)=mean((X_applied(1,:)-ref(time)).^2);
end
%toc

% Plot the results
figure('Renderer', 'painters', 'Position', [10 10 800 600])
subplot(2,1,1)
semilogx(k,MeanControlEnergy)
set(gca,'FontSize',FontSAxis);
xlabel('Control cost factor k','fontweight','bold','fontsize',FontSLabel)
ylabel('Mean control energy [arb. Unit^2]','fontweight','bold','fontsize',FontSLabel)

subplot(2,1,2)
semilogx(k,MSE)
set(gca,'FontSize',FontSAxis);
xlabel('Control cost factor k','fontweight','bold','fontsize',FontSLabel)
ylabel('MSE [mV^2]','fontweight','bold','fontsize',FontSLabel)
title("Noise \sigma^2=" + sigma^2 + "(mV)^2")
sgtitle('Optimize k - cost of control energy','fontweight','bold','fontsize',FontSSGTitle)


set(gcf,'Units','points')
set(gcf,'PaperUnits','points')
size = get(gcf,'Position');

size = size(3:4);
set(gcf,'PaperSize',size)
set(gcf,'PaperPosition',[0,0,size(1),size(2)])

print(gcf,'OptimizeK','-depsc','-loose'); % Save figure as .eps file

%%
%old parameters for the following simulations
T = 2.5;          % s
N = T*fs;       % steps
% Reference
f_ref = 8;      % Hz
a_ref = .5;     % mV
t2_ref = T/2;   % s
a2_ref = .3;    % mV
ref = @(t) a_ref*sin(2*pi*f_ref*t) + a2_ref*heaviside(t-t2_ref).*sin(2*pi*f_ref*t);

%% plot Multiple Shooting
%parameters
sigma = 0.0;    % disturbance on each simulation step
% MPC
shift = N;  % MPC interval
N_mpc = N; % MPC horizon

% Objective
k = 100.0;     % control cost
L = (ref(t)-x1)^2 + k*u^2;
% Formulate discrete time dynamics
F = rk4integrator(x, p, u, t, xdot, L, 1/fs);

% Run MPC Simulation without noise
rng default; % Fix RNG for reproducibility

[X_applied, U_applied] = MPC(F, x0, param, sigma, N, N_mpc, shift, ts);

% Plot the solution
figure('Renderer', 'painters', 'Position', [10 10 800 600])
subplot(2,1,1)
hold on
time = ts*(0:N);
plot(time, ref(time))
plot(time, X_applied(1,:), '-')
plot(time, X_applied(2,:), '--')
stairs(time, U_applied([1:N N]), '-.')
set(gca,'FontSize',FontSAxis);
xlabel('t [s]','fontweight','bold','fontsize',FontSLabel)
legend('x_{ref} [mV]','x1 [mV]','x2 [mV]','u [arb. Unit]', 'Location', 'none', 'Position', [0.78 0.82 0.1433 0.1560])
title({"Noise \sigma^2=" + sigma^2 + "(mV)^2", "MSE= " + num2str((mean((X_applied(1,:)-ref(time)).^2)),'%10.5e\n') + "(mV)^2"},'fontweight','bold','fontsize',FontSTitle)


% Run MPC Simulation with noise
sigma = 0.1;    % disturbance on each simulation step
[X_applied, U_applied] = MPC(F, x0, param, sigma, N, N_mpc, shift, ts);

% Plot the solution
subplot(2,1,2)
hold on
time = ts*(0:N);
plot(time, ref(time))
plot(time, X_applied(1,:), '-')
plot(time, X_applied(2,:), '--')
stairs(time, U_applied([1:N N]), '-.')
set(gca,'FontSize',FontSAxis);
xlabel('t [s]','fontweight','bold','fontsize',FontSLabel)
title({"Noise \sigma^2=" + sigma^2 + "(mV)^2", "MSE= " + num2str((mean((X_applied(1,:)-ref(time)).^2)),'%10.5e\n')+ "(mV)^2"},'fontweight','bold','fontsize',FontSTitle)


sgtitle('Multiple shooting','fontweight','bold','fontsize',FontSSGTitle)

set(gcf,'Units','points')
set(gcf,'PaperUnits','points')
size = get(gcf,'Position');

size = size(3:4);
set(gcf,'PaperSize',size)
set(gcf,'PaperPosition',[0,0,size(1),size(2)])

print(gcf,'MultipleShooting','-depsc','-loose'); % Save figure as .eps file
%% plot MPC
%parameters
sigma = 0.0;    % disturbance on each simulation step
% MPC
shift = 1;  % MPC interval
N_mpc = 10; % MPC horizon

% Objective
k = 100.0;     % control cost
L = (ref(t)-x1)^2 + k*u^2;
% Formulate discrete time dynamics
F = rk4integrator(x, p, u, t, xdot, L, 1/fs);

% Run MPC Simulation without noise
rng default; % Fix RNG for reproducibility

[X_applied, U_applied] = MPC(F, x0, param, sigma, N, N_mpc, shift, ts);

% Plot the solution
figure('Renderer', 'painters', 'Position', [10 10 800 600])
subplot(2,1,1)
hold on
time = ts*(0:N);
plot(time, ref(time))
plot(time, X_applied(1,:), '-')
plot(time, X_applied(2,:), '--')
stairs(time, U_applied([1:N N]), '-.')
set(gca,'FontSize',FontSAxis);
xlabel('t [s]','fontweight','bold','fontsize',FontSLabel)
legend('x_{ref} [mV]','x1 [mV]','x2 [mV]','u [arb. Unit]', 'Location', 'none', 'Position', [0.78 0.82 0.1433 0.1560])
title({"Noise \sigma^2=" + sigma^2 + "(mV)^2", "MSE= " + num2str((mean((X_applied(1,:)-ref(time)).^2)),'%10.5e\n') + "(mV)^2"},'fontweight','bold','fontsize',FontSTitle)


% Run MPC Simulation with noise
sigma = 0.1;    % disturbance on each simulation step
[X_applied, U_applied] = MPC(F, x0, param, sigma, N, N_mpc, shift, ts);

% Plot the solution
subplot(2,1,2)
hold on
time = ts*(0:N);
plot(time, ref(time))
plot(time, X_applied(1,:), '-')
plot(time, X_applied(2,:), '--')
stairs(time, U_applied([1:N N]), '-.')
set(gca,'FontSize',FontSAxis);
xlabel('t [s]','fontweight','bold','fontsize',FontSLabel)
title({"Noise \sigma^2=" + sigma^2 + "(mV)^2", "MSE= " + num2str((mean((X_applied(1,:)-ref(time)).^2)),'%10.5e\n')+ "(mV)^2"},'fontweight','bold','fontsize',FontSTitle)


sgtitle('Model Predicte Control','fontweight','bold','fontsize',FontSSGTitle)

set(gcf,'Units','points')
set(gcf,'PaperUnits','points')
size = get(gcf,'Position');

size = size(3:4);
set(gcf,'PaperSize',size)
set(gcf,'PaperPosition',[0,0,size(1),size(2)])

print(gcf,'MPC','-depsc','-loose'); % Save figure as .eps file




%% plot MPC with parameter change
%parameters
sigma = 0.0;    % disturbance on each simulation step
% MPC
shift = 1;  % MPC interval
N_mpc = 10; % MPC horizon

% Objective
k = 100.0;     % control cost
L = (ref(t)-x1)^2 + k*u^2;
% Formulate discrete time dynamics
F = rk4integrator(x, p, u, t, xdot, L, 1/fs);

% Run MPC Simulation without noise
rng default; % Fix RNG for reproducibility

%norm=(0.25+time./T*1.5)';
param2 = repmat(param,N/3,1);
param2=[param2;2.5*param2;0.4*param2; 0.4*param];
%param2=norm*param;

% Plot the normalized parametrs
figure('Renderer', 'painters', 'Position', [10 10 800 600])
subplot(2,1,1)
hold on
time = ts*(0:N);
for ii=1:4
    plot(time, param2(:,ii)/param(:,ii))
end
set(gca,'FontSize',FontSAxis);
xlabel('t [s]','fontweight','bold','fontsize',FontSLabel)
legend('w/w_0','a/a_0','k_1/k_{1,0}','k_2/k_{2,0}', 'Location', 'none', 'Position', [0.78 0.82 0.1433 0.1560])
title({"Normalized Parameters"},'fontweight','bold','fontsize',FontSTitle)


[X_applied, U_applied] = MPC(F, x0, param, sigma, N, N_mpc, shift, ts,param2);

% Plot the solution
subplot(2,1,2)
hold on
time = ts*(0:N);
plot(time, ref(time))
plot(time, X_applied(1,:), '-')
plot(time, X_applied(2,:), '--')
stairs(time, U_applied([1:N N]), '-.')
set(gca,'FontSize',FontSAxis);
xlabel('t [s]','fontweight','bold','fontsize',FontSLabel)
title({"Noise \sigma^2=" + sigma^2 + "(mV)^2", "MSE= " + num2str((mean((X_applied(1,:)-ref(time)).^2)),'%10.5e\n')+ "(mV)^2"},'fontweight','bold','fontsize',FontSTitle)


sgtitle('Model Predicte Control with disturbance of Model','fontweight','bold','fontsize',FontSSGTitle)

set(gcf,'Units','points')
set(gcf,'PaperUnits','points')
size = get(gcf,'Position');

size = size(3:4);
set(gcf,'PaperSize',size)
set(gcf,'PaperPosition',[0,0,size(1),size(2)])

print(gcf,'MPC','-depsc','-loose'); % Save figure as .eps file
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


