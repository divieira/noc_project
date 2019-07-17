%Plots control energy - MSE
clear
clc
close all
import casadi.*


% Plots
FontSTitle=11;
FontSAxis=12;
FontSSGTitle=14;
FontSLabel=10;
set(0,'DefaultLineLineWidth',2)

% Fix random seed to make noise deterministic
rng default; % Fix RNG for reproducibility

%% Parameters
%many are redifined in the simulations to get a specific behavior or speed
%Mainly relevant to multiple shooting and MPC
% Simulation
fs = 120;       % Hz
T = 2.5;          % s
N = T*fs;       % steps
ts = 1/fs;      % s
x0 = [0; 0];    % initial conditions

% Model for optimization
param = [2*pi*6 .01 -1e3 0]; % [w, a, k1, k2]
tau = 1.0;      % s (arbitrary stiffness constant)

% Reference for multiple shooting and MPC
f_ref = 8;      % Hz
a_ref = .5;     % mV
t2_ref = 2.5;   % s
a2_ref = .3;    % mV
ref = @(t) a_ref*sin(2*pi*f_ref*t) + a2_ref*heaviside(t-t2_ref).*sin(2*pi*f_ref*t);


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
k = 100.0;     % control cost
L = (ref(t)-x1)^2 + k*u^2;

% Formulate discrete time dynamics
F = rk4integrator(x, p, u, t, xdot, L, 1/fs);

noiseLevel=0.1;
%% plot Multiple Shooting
%parameters
sigma = 0.0;    % disturbance on each simulation step
% MPC
shift = N;  % MPC interval
N_mpc = N; % MPC horizon

[X_applied, U_applied] = MPC(F, x0, param, sigma, N, N_mpc, shift, ts);

% Plot the solution
figure('Renderer', 'painters', 'Position', [10 10 800 600])
subplot(2,1,1)
hold on
time = ts*(0:N);
plot(time, ref(time))
plot(time, X_applied(1,:), '-')
plot(time, X_applied(2,:), '--')
stairs(time, U_applied([1:N N]), '-.','LineWidth',2)
set(gca,'FontSize',FontSAxis);
xlabel('t [s]','fontweight','bold','fontsize',FontSLabel)
legend('x_{ref} [mV]','x1 [mV]','x2 [mV]','u [arb. Unit]', 'Location', 'none', 'Position', [0.78 0.82 0.1433 0.1560])
title({"Noise \sigma^2=" + sigma^2 + "(mV)^2", "MSE= " + num2str((mean((X_applied(1,:)-ref(time)).^2)),'%10.5e\n') + "(mV)^2"},'fontweight','bold','fontsize',FontSTitle)
axis([0 T -1.6 1.6])

% Run MPC Simulation with noise
sigma = noiseLevel;    % disturbance on each simulation step
[X_applied, U_applied] = MPC(F, x0, param, sigma, N, N_mpc, shift, ts);

% Plot the solution
subplot(2,1,2)
hold on
time = ts*(0:N);
plot(time, ref(time))
plot(time, X_applied(1,:), '-')
plot(time, X_applied(2,:), '--')
stairs(time, U_applied([1:N N]), '-.','LineWidth',2)
set(gca,'FontSize',FontSAxis);
xlabel('t [s]','fontweight','bold','fontsize',FontSLabel)
title({"Noise \sigma^2=" + sigma^2 + "(mV)^2", "MSE= " + num2str((mean((X_applied(1,:)-ref(time)).^2)),'%10.5e\n')+ "(mV)^2"},'fontweight','bold','fontsize',FontSTitle)
axis([0 T -1.6 1.6])

sgtitle('Multiple shooting - Open Loop','fontweight','bold','fontsize',FontSSGTitle)

set(gcf,'Units','points')
set(gcf,'PaperUnits','points')
sizeP = get(gcf,'Position');

sizeP = sizeP(3:4);
set(gcf,'PaperSize',sizeP)
set(gcf,'PaperPosition',[0,0,sizeP(1),sizeP(2)])

print(gcf,'MultipleShooting','-depsc','-loose'); % Save figure as .eps file
%% plot MPC
%parameters
shift_sim=[1 3 5 10];
N_mpc_vec=[10 10 15 20];
for idxShift=1:size(shift_sim,2)
    %% Multiple shooting and MPC
    % Objective
    k = 100.0;     % control cost
    L = (ref(t)-x1)^2 + k*u^2;
    
    %shorter TimeFrame to obtain better visualization
    T = 1.5;          % s
    N = T*fs;       % steps

    % Formulate discrete time dynamics
    F = rk4integrator(x, p, u, t, xdot, L, 1/fs);
    noiseLevel=0.1;
    
    
    sigma = 0.0;    % disturbance on each simulation step
    % MPC
    shift = shift_sim(idxShift);  % MPC interval
    N_mpc = N_mpc_vec(idxShift); % MPC horizon

    [X_applied, U_applied] = MPC(F, x0, param, sigma, N, N_mpc, shift, ts);

    % Plot the solution
    figure('Renderer', 'painters', 'Position', [10 10 800 600])
    subplot(2,1,1)
    hold on
    time = ts*(0:N);
    plot(time, ref(time))
    plot(time, X_applied(1,:), '-')
    plot(time, X_applied(2,:), '--')
    stairs(time, U_applied([1:N N]), '-.','LineWidth',2)
    set(gca,'FontSize',FontSAxis);
    xlabel('t [s]','fontweight','bold','fontsize',FontSLabel)
    legend('x_{ref} [mV]','x1 [mV]','x2 [mV]','u [arb. Unit]', 'Location', 'none', 'Position', [0.78 0.82 0.1433 0.1560])
    str1="Noise \sigma^2=" + sigma^2 + "(mV)^2";
    str2="MSE= " + num2str((mean((X_applied(1,:)-ref(time)).^2)),'%10.5e\n') + "(mV)^2, Shift = " + num2str(shift) +", Horizon = "+ num2str(N_mpc);
    title({str1,str2},'fontweight','bold','fontsize',FontSTitle)
    axis([0 T -1 1])

    % Run MPC Simulation with noise
    sigma = noiseLevel;    % disturbance on each simulation step
    [X_applied, U_applied] = MPC(F, x0, param, sigma, N, N_mpc, shift, ts);

    % Plot the solution
    subplot(2,1,2)
    hold on
    time = ts*(0:N);
    plot(time, ref(time))
    plot(time, X_applied(1,:), '-')
    plot(time, X_applied(2,:), '--')
    stairs(time, U_applied([1:N N]), '-.','LineWidth',2)
    set(gca,'FontSize',FontSAxis);
    xlabel('t [s]','fontweight','bold','fontsize',FontSLabel)
    str1="Noise \sigma^2=" + sigma^2 + "(mV)^2";
    str2="MSE= " + num2str((mean((X_applied(1,:)-ref(time)).^2)),'%10.5e\n') + "(mV)^2, Shift = " + num2str(shift) +", Horizon = "+ num2str(N_mpc);
    title({str1,str2},'fontweight','bold','fontsize',FontSTitle)
    axis([0 T -1 1])

    sgtitle('Model Predicte Control - Closed Loop','fontweight','bold','fontsize',FontSSGTitle)

    set(gcf,'Units','points')
    set(gcf,'PaperUnits','points')
    sizeP = get(gcf,'Position');

    sizeP = sizeP(3:4);
    set(gcf,'PaperSize',sizeP)
    set(gcf,'PaperPosition',[0,0,sizeP(1),sizeP(2)])

    printstr="MPC"+int2str(shift)+"_"+int2str(N_mpc)
    print(gcf,printstr,'-depsc','-loose'); % Save figure as .eps file
    
    %% plot MPC with parameter change
    %parameters
    T = 5;          % s
    N = T*fs;       % steps

    %parameters
    sigma = 0.0;    % disturbance on each simulation step
    % MPC run if normal MPC was not executet before.
    %shift = 1;  % MPC interval
    %N_mpc = 10; % MPC horizon


    sigma = 0.0;    % disturbance on each simulation step

    % Reference
    f_ref = 8;      % Hz
    a_ref = .5;     % mV
    t2_ref = 2.5;   % s
    a2_ref = .0;    % mV
    ref = @(t) a_ref*sin(2*pi*f_ref*t) + a2_ref*heaviside(t-t2_ref).*sin(2*pi*f_ref*t);

    time = ts*(0:N);

    %Parameter pertubation
    normW=(1+1.5*heaviside(time-0.5)-2.1*heaviside(time-1.0)+0.6*heaviside(time-1.5))';
    normA=(1+1.5*heaviside(time-2)-2.1*heaviside(time-2.5)+0.6*heaviside(time-3.0))';
    normK1=(1+1.5*heaviside(time-3.5)-2.1*heaviside(time-4.0)+0.6*heaviside(time-4.5))';
    normOnes=ones(size(time,2),1);
    param2=[normW*param(1),normA*param(2),normK1*param(3),normOnes*param(4)];

    % Plot the normalized parametrs
    figure('Renderer', 'painters', 'Position', [10 10 800 600])
    subplot(2,1,1)
    hold on
    for ii=1:4
        plot(time, param2(:,ii)/param(:,ii))
    end
    set(gca,'FontSize',FontSAxis);
    xlabel('t [s]','fontweight','bold','fontsize',FontSLabel)
    legend('w/w_0','a/a_0','k_1/k_{1,0}','k_2/k_{2,0}', 'Location', 'none', 'Position', [0.78 0.82 0.1433 0.1560])
    title({"Normalized Parameters"},'fontweight','bold','fontsize',FontSTitle)

    %Simulate MPC without noise and perturbed parameters
    [X_applied, U_applied] = MPC(F, x0, param, sigma, N, N_mpc, shift, ts,param2);

    % Plot the solution
    subplot(2,1,2)
    hold on
    time = ts*(0:N);
    plot(time, ref(time))
    plot(time, X_applied(1,:), '-')
    plot(time, X_applied(2,:), '--')
    stairs(time, U_applied([1:N N]), '-.','LineWidth',2)
    set(gca,'FontSize',FontSAxis);
    xlabel('t [s]','fontweight','bold','fontsize',FontSLabel)
    str1="Noise \sigma^2=" + sigma^2 + "(mV)^2";
    str2="MSE= " + num2str((mean((X_applied(1,:)-ref(time)).^2)),'%10.5e\n') + "(mV)^2, Shift = " + num2str(shift) +", Horizon = "+ num2str(N_mpc);
    title({str1,str2},'fontweight','bold','fontsize',FontSTitle)
    legend('x_{ref} [mV]','x1 [mV]','x2 [mV]','u [arb. Unit]', 'Location', 'none', 'Position', [0.78 0.34 0.1433 0.1560])

    sgtitle('Model Predicte Control with disturbance of Model','fontweight','bold','fontsize',FontSSGTitle)

    set(gcf,'Units','points')
    set(gcf,'PaperUnits','points')
    sizeP = get(gcf,'Position');

    sizeP = sizeP(3:4);
    set(gcf,'PaperSize',sizeP)
    set(gcf,'PaperPosition',[0,0,sizeP(1),sizeP(2)])

    
    printstr="MPC_ParameterDisturb"+int2str(shift)+"_"+int2str(N_mpc)
    print(gcf,printstr,'-depsc','-loose'); % Save figure as .eps file


    %% Run MPC Simulation optimize k
    T = 5;          % s
    N = T*fs;       % steps
    sigma = 0.05;

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
    title("Shift = " + int2str(shift) +", Horizon = "+int2str(N_mpc))
    subplot(2,1,2)
    semilogx(k,MSE)
    set(gca,'FontSize',FontSAxis);
    xlabel('Control cost factor k','fontweight','bold','fontsize',FontSLabel)
    ylabel('MSE [mV^2]','fontweight','bold','fontsize',FontSLabel)
    str1="Noise \sigma^2=" + sigma^2 + "(mV)^2";
    title({str1},'fontweight','bold','fontsize',FontSTitle)
    sgtitle('Optimize k - cost of control energy in MPC','fontweight','bold','fontsize',FontSSGTitle)


    set(gcf,'Units','points')
    set(gcf,'PaperUnits','points')
    sizeP = get(gcf,'Position');

    sizeP = sizeP(3:4);
    set(gcf,'PaperSize',sizeP)
    set(gcf,'PaperPosition',[0,0,sizeP(1),sizeP(2)])
    
    printstr="OptimizeK"+int2str(shift)+"_"+int2str(N_mpc);
    print(gcf,printstr,'-depsc','-loose'); % Save figure as .eps file
end
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


