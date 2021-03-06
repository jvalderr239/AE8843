%% dMPC Script
clc;clear;close all;

rng('default')
% Initial State [x;y;theta]
variance = 0.0000001;

init_state = [20;
              20;
              0];
y_hat_nl(:,1) = init_state;         
y_hat = init_state(3);

goal_state = [8;
              9;
              0];
          
init_cntrl = [0,0];         

% System Mat

A = [1,0,0;
           0,1,0;
           0,0,1]; % A mat
       
sys_mat = A;

% Try to make an estimate of sys mat
param_est = randn(1,3);

sys_mat_est = diag(param_est);


% Control Variables[v;w]
       
% control_mat = [cos(theta), 0;
%                sin(theta), 0;
%                0,          1]; % B Mat

Tf = 5; % seconds
dt = 0.01;
time_series = 0:dt:Tf;


% dMPC
r = 0.5;
alpha = 0.25;
theta_hat = [0;0;0];
horizon = 3; % N = 3   
P_mat = 1000*eye(3);

xstar(:,1) = init_state;
ustar(:,1) = init_cntrl';
for i = 2:length(time_series)
    %Updated - JEV
    % ut should use ustar input
    % xt can be deviation so fmincon will drive to zero dev
    %
    xt = xstar(:,i-1)-goal_state; 
    ut = ustar(:,i-1); 
    theta = xt(3);
    v = ut(1);
    B = [cos(theta), v*sin(theta);
         sin(theta), -v*cos(theta); 
         0,          0]; %Linearized control_mat w.r.t theta, v, and w                      
    [k_star,~,~] = icare(A,B,theta_hat*theta_hat');
    %% Minimize: y_hakt^2 + r*ustar^2 + xt'*P_mat*xt... to get ustar 
    %% Need to update with Unicycle expansion and both control inputs

%     min_func = @(ustar_min)(-(y_hat'*y_hat + r*[ustar_min(1);ustar_min(2)]'*[ustar_min(1);ustar_min(2)] + xt'*P_mat*xt + ...
%                 alpha*(theta_hat'*(xt+(A*xt+B.*[ustar_min(1);ustar_min(2)])*dt) + r*[ustar_min(1);ustar_min(2)]'*[ustar_min(1);ustar_min(2)] + (xt+(A*xt+B.*[ustar_min(1);ustar_min(2)])*dt)'*(P_mat + (xt+(A*xt+B.*[ustar_min(1);ustar_min(2)])*dt)*(xt+(A*xt+B.*[ustar_min(1);ustar_min(2)])*dt)')^-1*(xt+(A*xt+B.*[ustar_min(1);ustar_min(2)])*dt)) + ...
%                 (alpha*alpha)*(theta_hat'*(xt+(A*(xt+(A*xt+B.*[ustar_min(1);ustar_min(2)])*dt)+ B*[ustar_min(1);ustar_min(2)])*dt) + r*[ustar_min(1);ustar_min(2)]'*[ustar_min(1);ustar_min(2)] + (xt+(A*(xt+(A*xt+B.*[ustar_min(1);ustar_min(2)])*dt)+ B*[ustar_min(1);ustar_min(2)])*dt)'*((P_mat + (xt+(A*xt+B.*[ustar_min(1);ustar_min(2)])*dt)*(xt+(A*xt+B.*[ustar_min(1);ustar_min(2)])*dt)'+(xt+(A*(xt+(A*xt+B.*[ustar_min(1);ustar_min(2)])*dt)+ B*[ustar_min(1);ustar_min(2)])*dt)*(xt+(A*(xt+(A*xt+B.*[ustar_min(1);ustar_min(2)])*dt)+ B*[ustar_min(1);ustar_min(2)])*dt)'))^-1*(xt+(A*(xt+(A*xt+B.*[ustar_min(1);ustar_min(2)])*dt)+ B*[ustar_min(1);ustar_min(2)])*dt)) + ...
%                   (alpha*alpha*alpha)*(xt+(A*(xt+(A*(xt+(A*xt+B.*[ustar_min(1);ustar_min(2)])*dt)+ B*[ustar_min(1);ustar_min(2)])*dt) + B*[ustar_min(1);ustar_min(2)])*dt)'*(k_star)*(xt+(A*(xt+(A*(xt+(A*xt+B.*[ustar_min(1);ustar_min(2)])*dt)+ B*[ustar_min(1);ustar_min(2)])*dt) + B*[ustar_min(1);ustar_min(2)])*dt)));

    min_func = @(ustar_min)((y_hat'*y_hat + r*[ustar_min(1);ustar_min(2)]'*[ustar_min(1);ustar_min(2)] + xt'*P_mat*xt + ...
                alpha*(theta_hat'*(xt+(A*xt+B*[ustar_min(1);ustar_min(2)])*dt) + r*[ustar_min(1);ustar_min(2)]'*[ustar_min(1);ustar_min(2)] + (xt+(A*xt+B*[ustar_min(1);ustar_min(2)])*dt)'*(P_mat + (xt+(A*xt+B*[ustar_min(1);ustar_min(2)])*dt)*(xt+(A*xt+B*[ustar_min(1);ustar_min(2)])*dt)')^-1*(xt+(A*xt+B*[ustar_min(1);ustar_min(2)])*dt)) + ...
                (alpha*alpha)*(theta_hat'*(xt+(A*(xt+(A*xt+B*[ustar_min(1);ustar_min(2)])*dt)+ B*[ustar_min(1);ustar_min(2)])*dt) + r*[ustar_min(1);ustar_min(2)]'*[ustar_min(1);ustar_min(2)] + (xt+(A*(xt+(A*xt+B*[ustar_min(1);ustar_min(2)])*dt)+ B*[ustar_min(1);ustar_min(2)])*dt)'*((P_mat + (xt+(A*xt+B*[ustar_min(1);ustar_min(2)])*dt)*(xt+(A*xt+B*[ustar_min(1);ustar_min(2)])*dt)'+(xt+(A*(xt+(A*xt+B*[ustar_min(1);ustar_min(2)])*dt)+ B*[ustar_min(1);ustar_min(2)])*dt)*(xt+(A*(xt+(A*xt+B*[ustar_min(1);ustar_min(2)])*dt)+ B*[ustar_min(1);ustar_min(2)])*dt)'))^-1*(xt+(A*(xt+(A*xt+B*[ustar_min(1);ustar_min(2)])*dt)+ B*[ustar_min(1);ustar_min(2)])*dt))));
    
    %         x = fmincon(@(ustar)y_hat*y_hat + r*ustar*ustar + xt'*P_mat*xt,[0],[],[],[],[],[-1],[1]);
    
    %Updated - JEV
    %Returns optimal input as row vector so needs input as row vector
    %
    linear_uopt = fmincon(min_func,init_cntrl ,[],[],[],[],[-1;-1],[1;1]);
    %%
    ustar(:,i) = linear_uopt';
    init_cntrl = ustar(:,i)';
%     if (i > 3) && (sign(xstar(2,i-2)) ~= sign(xstar(2,i-1)))
%         ustar(i-1) = 0;
%     end
    
    control_mat = [cos(theta), 0;
                   sin(theta), 0;
                   0,          1]; % B Mat
    
    y_hat_nl(:,i) = y_hat_nl(:,i-1)+...
        (sys_mat * y_hat_nl(:,i-1) + control_mat * ustar(:,i-1))*dt + ...
        (sqrt(variance) * randn()); % Nonlinear Model Update
    y_hat = y_hat_nl(3);
    xstar(:,i) = (A*xt+B*ustar(:,i))*dt;   %Linearized Model Update
    xt = xstar(:,i);
    % After finding min and propagating, recalculate G, theta_hat, and
    % P
    G = P_mat * xt * (variance + xt'*P_mat*xt)^-1;
    theta_hat = theta_hat + G*(y_hat - theta_hat'*xt);
    P_mat = (eye(3) - G*xt')*P_mat;
end

% Switch from deviation space to state space
iters = size(xstar,2);
 optimal_x(:, 1) = init_state;
for idx = 2:iters
    optimal_x(:, idx) = xstar(:,idx)+ goal_state;
end

figure
hold on
plot(optimal_x(1,:), optimal_x(2,:))
plot(y_hat_nl(1,:),y_hat_nl(2,:),'r')
legend('Linearized', 'Non-Linear')
title('x-y plot')

figure
hold on
plot(time_series(1:end),ustar(1,:))
title('v control plot')

figure
hold on
plot(time_series(1:end),ustar(2,:))
title('w control plot')





           
