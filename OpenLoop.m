%Plots control energy - MSE
clear
clc
close all
import casadi.*


% Plot Fonts
FontSTitle=11;
FontSAxis=12;
FontSSGTitle=14;
FontSLabel=10;
set(0,'DefaultLineLineWidth',2)
figpath = "Figures/";
if ~isfolder(figpath), mkdir(figpath); end

% Fix random seed to make noise deterministic
rng default; % Fix RNG for reproducibility

%% Parameters
%many are redifined in the simulations to get a specific behavior or speed
%Mainly relevant to multiple shooting and MPC
% Simulation
fs = 160;       % Hz
T = 1;        % s
N = T*fs;       % steps
ts = 1/fs;      % s
x0 = [.5; 0];    % initial conditions

% Model for optimization
param = [2*pi*6 .01 -1e2 0]; % [w, a, k1, k2]
tau = 0.001;      % s (arbitrary stiffness constant)


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

%% Uncontrolled System
% Formulate discrete time dynamics
F = rk4integrator(x, p, u, t, xdot, 0, 1/fs);
time = ts*(0:1*N);
x_0=[0.0; 0.05];
X=[];
X=[X, x_0];
for ii=2:size(time,2)
    Fk = F('x0',X(:,ii-1), 'p',param, 'u',0, 't',time(ii));
    X = [X, full(Fk.xf)];
end

x2_0=[0.0; 0.15];
X2=[];
X2=[X2, x2_0];
for ii=2:size(time,2)
    Fk = F('x0',X2(:,ii-1), 'p',param, 'u',0, 't',time(ii));
    X2 = [X2, full(Fk.xf)];
end
[x,y] = meshgrid(-0.15:0.05:0.15,-0.15:0.05:0.15);
u=zeros(size(x));
for ii=1:size(x,1)
    for jj=1:size(x,2)
        out = ode([x(ii,jj),y(ii,jj)],0,[tau,param]);
        u(ii,jj)=out(1);
        v(ii,jj)=out(2);
    end
end

% Plot the solution
figure('Renderer', 'painters', 'Position', [10 10 800 600])
hold on;
plot(time, X(1,:), '-')
plot(time, X(2,:), '--')
set(gca,'FontSize',FontSAxis);
xlabel('t [s]','fontweight','bold','fontsize',FontSLabel)
ylabel('u [mV]','fontweight','bold','fontsize',FontSLabel)
legend('x1','x2', 'Location', 'none', 'Position', [0.78 0.82 0.1433 0.1560])
title('Plant','fontweight','bold','fontsize',FontSTitle)
axis([0 T -1.6 1.6])

set(gcf,'Units','points')
set(gcf,'PaperUnits','points')
sizeP = get(gcf,'Position');

sizeP = sizeP(3:4);
set(gcf,'PaperSize',sizeP)
set(gcf,'PaperPosition',[0,0,sizeP(1),sizeP(2)])

print(gcf,figpath+'Plant','-depsc','-loose'); % Save figure as .eps file

figure('Renderer', 'painters', 'Position', [10 10 800 600])
hold on;
plot(X(1,:), X(2,:))
plot(X2(1,:), X2(2,:))
quiver(x,y,10*u,10*v,'LineWidth',0.8,'MaxHeadSize', 0.4)
set(gca,'FontSize',FontSAxis);
xlabel('x_1 [mV]','fontweight','bold','fontsize',FontSLabel)
ylabel('x_2 [mV]','fontweight','bold','fontsize',FontSLabel)
title('Plant','fontweight','bold','fontsize',FontSTitle)
axis([-0.15 0.15 -0.15 0.15])

set(gcf,'Units','points')
set(gcf,'PaperUnits','points')
sizeP = get(gcf,'Position');

sizeP = sizeP(3:4);
set(gcf,'PaperSize',sizeP)
set(gcf,'PaperPosition',[0,0,sizeP(1),sizeP(2)])

print(gcf,figpath+'PlantPhase','-depsc','-loose'); % Save figure as .eps file