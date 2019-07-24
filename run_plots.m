%Plots control energy - MSE
clear
clc
close all
import casadi.*

% Set plot defaults
set(0,'defaultFigureRenderer','painters');  % For eps exporting
set(0,'defaultTextFontSize',11);            % title, label
set(0,'defaultTextFontWeight','bold');      % title, label
set(0,'defaultAxesFontSize',12);            % axis tick labels
set(0,'defaultLineLineWidth',2);            % plot lines
set(0,'defaultFigurePosition',[10 10 800 600]); % figure size
set(0,'defaultLegendLocation','none');          % manual legend position
set(0,'defaultLegendPosition',[0.78 0.82 0.1433 0.1560]); % upper right
figpath = "Figures/";

% Fix random seed to make noise deterministic
rng default; % Fix RNG for reproducibility

%% Parameters
%many are redifined in the simulations to get a specific behavior or speed
%Mainly relevant to multiple shooting and MPC
% Simulation
fs = 160;       % Hz
T = 2.5;        % s
N = T*fs;       % steps
ts = 1/fs;      % s
x0 = [.5; 0];    % initial conditions

%parameters for independent runs - MPC
shift_sim=[1 5 10];
N_mpc_vec=[20 20 20];

% Model for optimization
param = [2*pi*6 .01 -1e2 0]; % [w, a, k1, k2]
tau = 0.1;      % s (arbitrary stiffness constant)

% Reference for multiple shooting and MPC
f_ref = 8;      % Hz
a_ref = .5;     % mV
t2_ref = 2.5;   % s
a2_ref = 0;     % mV
ref = @(t) a_ref*cos(2*pi*f_ref*t) + a2_ref*heaviside(t-t2_ref).*cos(2*pi*f_ref*t);


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
t = SX.sym('t');

%% Multiple shooting
% Objective
alpha = .01;     % control cost
L = (ref(t)-x1)^2 + alpha*u^2;
% Formulate discrete time dynamics
F = rk4integrator(x, p, u, t, xdot, L, 1/fs);


noiseLevel=10;
%% plot Multiple Shooting
%parameters
sigma = 0.0;    % disturbance on each simulation step
% MPC
shift = N;  % MPC interval
N_mpc = N; % MPC horizon

[X_applied, U_applied] = MPC(F, x0, param, sigma, N, N_mpc, shift, ts);

% Plot the solution
time = ts*(0:N);
fig = figure();
subplot(2,1,1);
plot_trajectory(time, ref, X_applied, U_applied);
legend('x_{ref} [mV]','x1 [mV]','x2 [mV]','u [a.u.]');
title({"Noise \sigma^2 = " + sigma^2 + "(mV)^2", "MSE = " + num2str((mean((X_applied(1,:)-ref(time)).^2)),'%10.5e\n') + "(mV)^2"});

% Run MPC Simulation with noise
sigma = noiseLevel;    % disturbance on each simulation step
[X_applied, U_applied] = MPC(F, x0, param, sigma, N, N_mpc, shift, ts);

% Plot the solution
subplot(2,1,2);
plot_trajectory(time, ref, X_applied, U_applied);
title({"Noise \sigma^2 = " + sigma^2 + "(mV)^2", "MSE = " + num2str((mean((X_applied(1,:)-ref(time)).^2)),'%10.5e\n')+ "(mV)^2"});

sgtitle('Multiple shooting - Open Loop');

save_figure(fig,figpath,'MultipleShooting');


%% plot MPC
for idxShift=1:length(shift_sim)
    %% Multiple shooting and MPC
    %shorter TimeFrame to obtain better visualization
    T = 1.5;          % s
    N = T*fs;       % steps

    % Formulate discrete time dynamics
    F = rk4integrator(x, p, u, t, xdot, L, 1/fs);
    
    sigma = 0.0;    % disturbance on each simulation step
    % MPC
    shift = shift_sim(idxShift);  % MPC interval
    N_mpc = N_mpc_vec(idxShift); % MPC horizon

    [X_applied, U_applied] = MPC(F, x0, param, sigma, N, N_mpc, shift, ts);

    % Plot the solution
    time = ts*(0:N);
    fig = figure();
    subplot(2,1,1)
    plot_trajectory(time, ref, X_applied, U_applied);
    legend('x_{ref} [mV]','x1 [mV]','x2 [mV]','u [a.u.]');
    str1="Noise \sigma^2 = " + sigma^2 + "(mV)^2";
    str2="MSE = " + num2str((mean((X_applied(1,:)-ref(time)).^2)),'%10.5e\n') + "(mV)^2, Shift = " + num2str(shift) +", Horizon = "+ num2str(N_mpc);
    title({str1,str2});

    % Run MPC Simulation with noise
    sigma = noiseLevel;    % disturbance on each simulation step
    [X_applied, U_applied] = MPC(F, x0, param, sigma, N, N_mpc, shift, ts);

    % Plot the solution
    subplot(2,1,2)
    plot_trajectory(time, ref, X_applied, U_applied);
    str1="Noise \sigma^2 =" + sigma^2 + "(mV)^2";
    str2="MSE = " + num2str((mean((X_applied(1,:)-ref(time)).^2)),'%10.5e\n') + "(mV)^2, Shift = " + num2str(shift) +", Horizon = "+ num2str(N_mpc);
    title({str1,str2});

    sgtitle('Model Predictive Control - Closed Loop')

    save_figure(fig,figpath,"MPC"+int2str(shift)+"_"+int2str(N_mpc));    
end
    
    %% plot MPC with parameter change
    %parameters
    T = 5;          % s
    N = T*fs;       % steps

    %parameters
    sigma = 0.0;    % disturbance on each simulation step
    % MPC run if normal MPC was not executet before.
    %shift = 1;  % MPC interval
    %N_mpc = 10; % MPC horizon


    % Reference
    f_ref = 8;      % Hz
    a_ref = .5;     % mV
    t2_ref = 2.5;   % s
    a2_ref = .0;    % mV
    ref = @(t) a_ref*cos(2*pi*f_ref*t) + a2_ref*heaviside(t-t2_ref).*cos(2*pi*f_ref*t);

    time = ts*(0:N);

    %Parameter pertubation
    normW=(1+1*heaviside(time-0.5)-1.5*heaviside(time-1.0)+0.5*heaviside(time-1.5))';
    normA=(1+1*heaviside(time-2)-1.5*heaviside(time-2.5)+0.5*heaviside(time-3.0))';
    normK1=(1+1*heaviside(time-3.5)-1.5*heaviside(time-4.0)+0.5*heaviside(time-4.5))';
    normOnes=ones(size(time,2),1);
    param2=[normW*param(1),normA*param(2),normK1*param(3),normOnes*param(4)];
for idxShift=1:length(shift_sim)
     % MPC
    shift = shift_sim(idxShift);  % MPC interval
    N_mpc = N_mpc_vec(idxShift); % MPC horizon
    
    % Plot the normalized parameters
    fig = figure();
    subplot(2,1,1);
    plot(time, [normW normA normK1]);
    xlabel('t [s]');
    legend('w/w_0','a/a_0','k_1/k_{1,0}');
    title({"Normalized Parameters"});

    % Simulate MPC without noise and perturbed parameters
    [X_applied, U_applied] = MPC(F, x0, param, sigma, N, N_mpc, shift, ts,param2);

    % Plot the solution
    subplot(2,1,2);
    hold on
    plot_trajectory(time, ref, X_applied, U_applied);
    str1="Noise \sigma^2 = " + sigma^2 + "(mV)^2";
    str2="MSE = " + num2str((mean((X_applied(1,:)-ref(time)).^2)),'%10.5e\n') + "(mV)^2, Shift = " + num2str(shift) +", Horizon = "+ num2str(N_mpc);
    title({str1,str2});
    legend('x_{ref} [mV]','x1 [mV]','x2 [mV]','u [a.u.]');

    sgtitle('Model Predictive Control with parameter deviations');

    save_figure(fig,figpath,"MPC_ParameterDisturb"+int2str(shift)+"_"+int2str(N_mpc));
end

    %% Run MPC Simulation to optimize k
    T = 5;          % s
    N = T*fs;       % steps
    
    sigma = 10;    % disturbance on each simulation step

    % Reference
    f_ref = 8;      % Hz
    a_ref = .5;     % mV
    t2_ref = 2.5;   % s
    a2_ref = .0;    % mV
    ref = @(t) a_ref*cos(2*pi*f_ref*t) + a2_ref*heaviside(t-t2_ref).*cos(2*pi*f_ref*t);

    alpha = logspace(-4,2,12);
    fig = figure();
    
for idxShift=1:length(shift_sim)
    % MPC
    shift = shift_sim(idxShift);  % MPC interval
    N_mpc = N_mpc_vec(idxShift); % MPC horizon
    MSE=zeros(size(alpha,2),1);
    MeanControlEnergy=zeros(size(alpha,2),1);
    time = ts*(0:N);
    for jj=1:size(alpha,2)
        % Objective term
        Lk = (ref(t)-x1)^2 + alpha(jj)*u^2;
        % Formulate discrete time dynamics
        F = rk4integrator(x, p, u, t, xdot, Lk, 1/fs);
        [X_applied, U_applied] = MPC(F, x0, param, sigma, N, N_mpc, shift, ts);
        MeanControlEnergy(jj)=mean(U_applied.^2);
        MSE(jj)=mean((X_applied(1,:)-ref(time)).^2);
    end

    % Plot the results  
    subplot(2,1,1)
    legendEntry="Shift: " + num2str(shift);
    semilogx(alpha,MeanControlEnergy,'DisplayName', legendEntry)
    hold on;
    subplot(2,1,2)
    semilogx(alpha,MSE)
    hold on
end
    subplot(2,1,1);
    xlabel('Control cost factor /alpha');
    ylabel('Mean control energy [a.u.^2]');
    title("Shift = " + int2str(shift) +", Horizon = "+int2str(N_mpc));
    legend;

    subplot(2,1,2);
    xlabel('Control cost factor /alpha');
    ylabel('MSE [mV^2]');
    str1="Noise \sigma^2 = " + sigma^2 + "(mV)^2";
    title({str1});
    
    sgtitle('MSE vs control cost');

    save_figure(fig,figpath,"OptimizeK"+int2str(shift)+"_"+int2str(N_mpc));


%% Plot functions
function plot_trajectory(time, ref, X, U)
    hold on;
    plot(time, ref(time));
    plot(time, X(1,:), '-');
    plot(time, X(2,:), '--')
    stairs(time, U([1:end end]), ':','LineWidth',2);

    xlabel('t [s]');
    ylim([-1.6 1.6]);
end

function save_figure(fig, figpath, name)
    set(fig,'Units','points');
    set(fig,'PaperUnits','points');
    sizeP = get(fig,'Position');

    sizeP = sizeP(3:4);
    set(fig,'PaperSize',sizeP);
    set(fig,'PaperPosition',[0,0,sizeP(1),sizeP(2)]);

    if ~isfolder(figpath), mkdir(figpath); end
    filename = fullfile(figpath,name);

    print(fig,filename,'-depsc','-loose'); % Save figure as .eps file
end