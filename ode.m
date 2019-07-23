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