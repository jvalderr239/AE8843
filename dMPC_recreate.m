%% dMPC Script
clc;clear;close all;

rng('default')
% Initial State [x;y;theta]
variance = 1;

init_state = [2;
              2;
              0];
% SRR - for the example in the paper, dim(y) = 1. 
% y_hat_nl dimension should also be 1. 
% From their graphs in section 4, it seems they set their initial output (y)
% to 1000 or so , I thought 10 may be a good starting point.
y_hat_nl = 10 ;         
y_hat = y_hat_nl;
goal_state = [0;
              0;
              0];
          
init_cntrl = [0];         

% System Mat

A = [2.5,-1,-0.1;
     1,   0,   0;
     0,   1   ,0]; % A mat
 
B = [1;
     0;
     0];    
 
 theta = [5;
          2;
          -1];
% SRR - the above theta variable is overwrriten below the xt, ut
% definitions. made a theta_truth variable to compute y_measured in the 
% measurement equations.
theta_truth = [5;
               2;
               -1;]
           
Tf = 5; % seconds
dt = 0.01;
time_series = 0:dt:Tf;
% SRR - Initializing k_star to all zeros, the same way the author does in 
% page 5, "Initialiation - K_t0 = 0_n" 
k_star = zeros(3);
% dMPC
r = 1;
alpha = 1;
theta_hat = [0;0;0];
horizon = 3; % N = 3   
P_mat = 1000*eye(3);
R_mat = P_mat^-1;
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
    
    %Update - SRR:
    %i believe this is DARE, not CARE. using idare now intead of icare.
    % source: https://www.mathworks.com/help/control/ref/idare.html
    % if you look at the link above, it looks exactly like the equation for
    % Kstar, with E = Identity (3x3)
    
    % Update - SRR
    % Adding observability check per page 5 , step 1 found in the paper. 
    % Kstar should only be updated if observable. Else, keep previous
    % Kstar.
    if rank(obsv(A, theta_hat')) == size(init_state, 1) 
        [k_star,~,~] = idare(A,B,theta_hat*theta_hat', r, 0, eye(3));
    end
    %% Minimize: y_hakt^2 + r*ustar^2 + xt'*P_mat*xt... to get ustar 
    %% Need to update with Unicycle expansion and both control inputs
    % SRR update - I removed dt's from min_func because A, and B are
    %               already in discrete time. 
    %              I took inverse of P, following the R_t+k equation in
    %             page 5 step 2.
    
    % SRR concern - paper says transpose(phi)*z >= 0 . How are we enforcing
    % this below? If problems persist, TODO:  Verify this actually holds true 
     
%     min_func = @(ustar_min)(-(y_hat'*y_hat + r*[ustar_min(1);ustar_min(2)]'*[ustar_min(1);ustar_min(2)] + xt'*P_mat*xt + ...
%                 alpha*(theta_hat'*(xt+(A*xt+B.*[ustar_min(1);ustar_min(2)])*dt) + r*[ustar_min(1);ustar_min(2)]'*[ustar_min(1);ustar_min(2)] + (xt+(A*xt+B.*[ustar_min(1);ustar_min(2)])*dt)'*(P_mat + (xt+(A*xt+B.*[ustar_min(1);ustar_min(2)])*dt)*(xt+(A*xt+B.*[ustar_min(1);ustar_min(2)])*dt)')^-1*(xt+(A*xt+B.*[ustar_min(1);ustar_min(2)])*dt)) + ...
%                 (alpha*alpha)*(theta_hat'*(xt+(A*(xt+(A*xt+B.*[ustar_min(1);ustar_min(2)])*dt)+ B*[ustar_min(1);ustar_min(2)])*dt) + r*[ustar_min(1);ustar_min(2)]'*[ustar_min(1);ustar_min(2)] + (xt+(A*(xt+(A*xt+B.*[ustar_min(1);ustar_min(2)])*dt)+ B*[ustar_min(1);ustar_min(2)])*dt)'*((P_mat + (xt+(A*xt+B.*[ustar_min(1);ustar_min(2)])*dt)*(xt+(A*xt+B.*[ustar_min(1);ustar_min(2)])*dt)'+(xt+(A*(xt+(A*xt+B.*[ustar_min(1);ustar_min(2)])*dt)+ B*[ustar_min(1);ustar_min(2)])*dt)*(xt+(A*(xt+(A*xt+B.*[ustar_min(1);ustar_min(2)])*dt)+ B*[ustar_min(1);ustar_min(2)])*dt)'))^-1*(xt+(A*(xt+(A*xt+B.*[ustar_min(1);ustar_min(2)])*dt)+ B*[ustar_min(1);ustar_min(2)])*dt)) + ...
%                   (alpha*alpha*alpha)*(xt+(A*(xt+(A*(xt+(A*xt+B.*[ustar_min(1);ustar_min(2)])*dt)+ B*[ustar_min(1);ustar_min(2)])*dt) + B*[ustar_min(1);ustar_min(2)])*dt)'*(k_star)*(xt+(A*(xt+(A*(xt+(A*xt+B.*[ustar_min(1);ustar_min(2)])*dt)+ B*[ustar_min(1);ustar_min(2)])*dt) + B*[ustar_min(1);ustar_min(2)])*dt)));

%     min_func = @(ustar_min)((y_hat'*y_hat + r*ustar_min'*ustar_min + xt'*P_mat*xt));
    %         x = fmincon(@(ustar)y_hat*y_hat + r*ustar*ustar + xt'*P_mat*xt,[0],[],[],[],[],[-1],[1]);
    
    % AS Defining H, Aeq, Aineq, beq, and bineq matrices and vectors for
    % quadprog
    Hmat = zeros(17);
    Hmat(1,1) = 1;
    Hmat(2,2) = r;
    Hmat(3,3) = k_star(1,1);
    Hmat(4,4) = k_star(1,2);
    Hmat(5,5) = k_star(1,3);
    Hmat(6,6) = k_star(2,1);
    Hmat(7,7) = k_star(2,2);
    Hmat(8,8) = k_star(2,3);
    Hmat(9,9) = k_star(3,1);
    Hmat(10,10) = k_star(3,2);
    Hmat(11,11) = k_star(3,3);

%     Hmat = 0.5*Hmat;
            
%     Hmat = 0.5*[1, 0, 0, 0, 0;         % y_hat       
%                 0, r, 0, 0, 0;
%                 0, 0, k_star, 0, 0;
%                 0, 0, 0, 0, 0;
%                 0, 0, 0, 0, 0];
            
    f_vec =  zeros(17,1);
    
    Aeq = zeros(17);
    Aeq(1,1) = 1;
    % Aeq row 2 is 0
    Aeq(3,2) = -B(1,1);
    Aeq(3,3) = 1;
    
    Aeq(4,2) = -B(2,1);
    Aeq(4,4) = 1;
    
    Aeq(5,2) = -B(3,1);
    Aeq(5,5) = 1;

%     Aeq(6,2) = ;
%     
%     Aeq(7,2) = ;
%     
%     Aeq(8,2) = ;
    
    Aeq(6,6) = R_mat(1,1);
    Aeq(6,7) = R_mat(1,2);
    Aeq(6,8) = R_mat(1,3);
    
    Aeq(7,6) = R_mat(2,1);
    Aeq(7,7) = R_mat(2,2);
    Aeq(7,8) = R_mat(2,3);
    
    Aeq(8,6) = R_mat(3,1);
    Aeq(8,7) = R_mat(3,2);
    Aeq(8,8) = R_mat(3,3);
    
    
    Aeq(9,9) = 1;
    Aeq(10,10) = 1;
    Aeq(11,11) = 1;
    Aeq(12,12) = 1;
    Aeq(13,13) = 1;
    Aeq(14,14) = 1;
    Aeq(15,15) = 1;
    Aeq(16,16) = 1;
    Aeq(17,17) = 1;
%     
%     Aeq = [1, 0, 0, 0, 0;
%            0, 0, 0, 0, 0;
%            0, -B,1, 0, 0;
%            0, 0, 0, P_mat^-1, 0;
%            0, 0, 0, 0, 1];

    beq = zeros(17,1);
    beq(1) = y_hat;
    beq(2) = 0;
    beq(3) = A(1,1)*xt(1,1)+A(1,2)*xt(2,1)+A(1,3)*xt(3,1);
    beq(4) = A(2,1)*xt(1,1)+A(2,2)*xt(2,1)+A(2,3)*xt(3,1);
    beq(5) = A(3,1)*xt(1,1)+A(3,2)*xt(2,1)+A(3,3)*xt(3,1);
    beq(6) = xt(1);
    beq(7) = xt(2);
    beq(8) = xt(3);
    beq(9) = R_mat(1,1);
    beq(10) = R_mat(1,2);
    beq(11) = R_mat(1,3);
    beq(12) = R_mat(2,1);
    beq(13) = R_mat(2,2);
    beq(14) = R_mat(2,3);
    beq(15) = R_mat(3,1);
    beq(16) = R_mat(3,2);
    beq(17) = R_mat(3,3);
%     beq = [y_hat; 0; A*xt; xt; P_mat^-10];
    
    Aineq = zeros(17);
    Aineq(6,6) = -xt(1,1);
    Aineq(7,7) = -xt(2,1);
    Aineq(8,8) = -xt(3,1);
    
%     Aineq = [0, 0, 0, 0,    0;
%              0, 0, 0, 0,    0;
%              0, 0, 0, 0,    0;
%              0, 0, 0, -xt', 0;
%              0, 0, 0, 0,    0];
    bineq = zeros(17,1);
    
    lb = ones(17,1)*-inf;
    lb(1) = -2000;
    lb(2) = -100;
    
    ub = lb*-1;
    linear_uopt = quadprog(0.5*Hmat,f_vec,Aineq,bineq,Aeq,beq,lb,ub);
    
    %Updated - JEV
    %Returns optimal input as row vector so needs input as row vector
    %
%     linear_uopt = fmincon(min_func,init_cntrl ,[],[],[],[],[-100],[100]);
    %%
    ustar(:,i) = linear_uopt(2);
%     init_cntrl = ustar(:,i)';
%     if (i > 3) && (sign(xstar(2,i-2)) ~= sign(xstar(2,i-1)))
%         ustar(i-1) = 0;
%     end
    % y_hat_nl is the "output measurement", from page 1 system(1) in the 
    % paper. 
    y_hat_nl(:,i) = y_hat_nl(:,i-1)+(A * y_hat_nl(:,i-1) + B * ustar(:,i))*dt + (sqrt(variance) * randn(3,1)); 
    y_hat_nl(:,i) = theta (sqrt(variance) * randn(3,1)); 
    y_hat = y_hat_nl(:,i);
    y_hat = y_hat(1);
    xstar(:,i) = xt + (A*xt+B*ustar(:,i))*dt;   %Linearized Model Update
    xt = xstar(:,i);
    % After finding min and propagating, recalculate G, theta_hat, and
    % P
    G = P_mat * xt * (variance + xt'*P_mat*xt)^-1;
    theta_hat = theta_hat + G*(y_hat - theta_hat'*xt);
    P_mat = (eye(3) - G*xt')*P_mat;
    R_mat = P_mat^-1;
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





           
