function F = rk4integrator(x, p, u, t, xdot, L, T)
    % See: CasADi example direct_multiple_shooting.m
   import casadi.*
   
   % Fixed step Runge-Kutta 4 integrator
   M = 1;%4; % RK4 steps per interval
   DT = T/M;
   f = casadi.Function('f', {x, p, u, t}, {xdot, L});
   X0 = x;
   P = p;
   X = X0;
   Q = 0;
   for j=1:M
       [k1, k1_q] = f(X, P, u, t);
       [k2, k2_q] = f(X + DT/2 * k1, P, u, t + DT/2);
       [k3, k3_q] = f(X + DT/2 * k2, P, u, t + DT/2);
       [k4, k4_q] = f(X + DT   * k3, P, u, t + DT);
       X=X+DT/6*(k1 +2*k2 +2*k3 +k4);
       Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q);
    end
    F = casadi.Function('F', {X0, P, u, t}, {X, Q}, {'x0','p','u','t'}, {'xf','qf'});
end
