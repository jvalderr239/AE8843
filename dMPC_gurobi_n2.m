%% dMPC Script
clc;clear;close all;

rng('default')
% Initial State [x;y;theta]
variance = 1;

init_state = [2;
              2;
              0];
% SRR - for the example in the paper, dim(y) = 1. 
% y_measured dimension should also be 1. 
% From their graphs in section 4, it seems they set their initial output (y)
% to 1000 or so , I thought 10 may be a good starting point.
y_measured = 10 ;         
y_hat = y_measured;
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
               -1];
           
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
     
    % AS Defining H, Aeq, Aineq, beq, and bineq matrices and vectors for
    % quadprog
    Hmat = zeros(34);
    % y_hat
    Hmat(1,1) = 1;
    Hmat(2,2) = 1; 
    % u
    Hmat(3,3) = r;
    Hmat(4,4) = r;
    
    % Phi_N
    Hmat(8,8) = k_star(1,1);
    Hmat(8,9) = k_star(1,2);
    Hmat(8,10) = k_star(1,3);
    Hmat(9,8) = k_star(2,1);
    Hmat(9,9) = k_star(2,2);
    Hmat(9,10) = k_star(2,3);
    Hmat(10,8) = k_star(3,1);
    Hmat(10,9) = k_star(3,2);
    Hmat(10,10) = k_star(3,3);
    
    % phi'*z for k > 0
    % SRR - 5,6,7 maps to phi_t+1. 14,15,16 maps to z_t+1. 
    Hmat(5,14) = 1.0;
    Hmat(6,15) = 1.0;
    Hmat(7,16) = 1.0;

    % UPDATE AS : Updated the f_vec 
    f_vec =  zeros(34,1);
    
    f_vec(11) = xt(1,:);
    f_vec(12) = xt(2,:);
    f_vec(13) = xt(3,:);
    
    % Updating Aeq:
    Aeq = zeros(33,34);
    % SRR - confiremd to satisfy yhat_t = y_t for k =0
    Aeq(1,1) = 1;
    % SRR - Update to row 2 adding yhat(t+1) constraint.
    %           yhat_(t+1) = theta_hat*Phi_t+1 , for k =1.
    %           => y_hat(t+1) - theta_hat*Phi_(t+1) = 0
    Aeq(2, 2) = 1;
    Aeq(2, 5) = -theta_hat(1,1);
    Aeq(2, 6) = -theta_hat(2,1);
    Aeq(2, 7) = -theta_hat(3,1);
    
    % SRR - Constraint for Phi_(t+1). This is for k = 0.
    %       Phi(t+1) - B*u(t)= A*Phi(t)
    Aeq(3,3) = -B(1,1);
    Aeq(3,5) = 1;
    
    Aeq(4,3) = -B(2,1);
    Aeq(4,6) = 1;
    
    Aeq(5,3) = -B(3,1);
    Aeq(5,7) = 1;
    
    
    % SRR - Adding constraint for Phi_(t+2). This is for k =1.
    %       Phi(t+2) = A*Phi(t+1) + B*u(t+1)
    %    => Phi(t+2) - A*Phi(t+1) - B*u(t+1) = 0
    %  Example row: Phi(t+2, element1) - A(1,1)*Phi(t+1, element1) -
    %  A(1,2)*Phi(t+1, element2) - A(1,3)*Phi(t+1,element3) - B(1,1)*u(t+1)
    %  = 0
    Aeq(6,4) = -B(1,1);
    Aeq(6,5) = -A(1,1);
    Aeq(6,6) = -A(1,2);
    Aeq(6,7) = -A(1,3);
    Aeq(6,8) = 1.0;
    
    Aeq(7,4) = -B(2,1);
    Aeq(7,5) = -A(2,1);
    Aeq(7,6) = -A(2,2);
    Aeq(7,7) = -A(2,3);
    Aeq(7,9) = 1.0;
    
    Aeq(8,4) = -B(3,1);
    Aeq(8,5) = -A(3,1);
    Aeq(8,6) = -A(3,2);
    Aeq(8,7) = -A(3,3);
    Aeq(8,10) = 1.0;
    
    % SRR - OPTIONAL (gurobi) Constraint for z(t). R(t)*z(t) = Phi(t), for k =0. 
    % Example row: R(1,1)*z(t, element1) + R(1,2)*z(t, element2) + 
    % R(1,3)*z(t, element) = Phi(t, element 1)
%     Aeq(9,11) = R_mat(1,1);
%     Aeq(9,12) = R_mat(1,2);
%     Aeq(9,13) = R_mat(1,3);
%     
%     Aeq(10,11) = R_mat(2,1);
%     Aeq(10,12) = R_mat(2,2);
%     Aeq(10,13) = R_mat(2,3);
%     
%     Aeq(11,11) = R_mat(3,1);
%     Aeq(11,12) = R_mat(3,2);
%     Aeq(11,13) = R_mat(3,3);
    
    % SRR- Skipping z(t+1) because its a nonlinear constraint.
    % SRR- R(t) constraint. R(t) = P(t)^-1 for k = 0.
    % Example Row: R(t, element (1,1)) = R_mat(1,1) = element (1,1) of P(t) ^-1
    Aeq(12,17) = 1;
    Aeq(13,18) = 1;
    Aeq(14,19) = 1;
    Aeq(15,20) = 1;
    Aeq(16,21) = 1;
    Aeq(17,22) = 1;
    Aeq(18,23) = 1;
    Aeq(19,24) = 1;
    Aeq(20,25) = 1;
    % Skipping R(t+1) constraint beacuse its a nonlinear constraint..
    
    % UPDATE SRR - beq(1) = y_measured 
    beq = zeros(33,1);
    % SRR- yhat(t) constraint.
    beq(1) = y_hat;
    % SRR - yhat(t+1) constraint.
    beq(2) = 0;
    % SRR - Phi(t+1) constraint.
    beq(3) = A(1,1)*xt(1,1)+A(1,2)*xt(2,1)+A(1,3)*xt(3,1);
    beq(4) = A(2,1)*xt(1,1)+A(2,2)*xt(2,1)+A(2,3)*xt(3,1);
    beq(5) = A(3,1)*xt(1,1)+A(3,2)*xt(2,1)+A(3,3)*xt(3,1);
    % SRR - Phi(t+2) constraint
    beq(6) = 0.0;
    beq(7) = 0.0;
    beq(8) = 0.0;
    % SRR - OPTIONAL. z(t) constraint.
%     beq(9) = xt(1);
%     beq(10) = xt(2);
%     beq(11) = xt(3);
    % SRR - skipping z(t+1) nonlinear constraint.
    % SRR - R(t) constraint:
    beq(12) = R_mat(1,1);
    beq(13) = R_mat(1,2);
    beq(14) = R_mat(1,3);
    beq(15) = R_mat(2,1);
    beq(16) = R_mat(2,2);
    beq(17) = R_mat(2,3);
    beq(18) = R_mat(3,1);
    beq(19) = R_mat(3,2);
    beq(20) = R_mat(3,3);
    % SRR - Skipping R(t+1) constraint.
    
    Aineq = zeros(1,34);
    % SRR - adding z(t) inequality constraint, for k = 0
    %      -Phi(1,1)*z(t, element 1) - Phi(2,1)*z(t, element2) - ... < 0
    Aineq(1,11) = -xt(1,1);
    Aineq(1,12) = -xt(2,1);
    Aineq(1,13) = -xt(3,1);
    % SRR - skipping z(t+1) inequality constraint, for k =1 because
    % nonlinear.
    bineq = zeros(1,1);
    
    lb = ones(34,1)*-inf;
    lb(1) = -2000;
    lb(2) = -2000;
    lb(3) = -100;
    lb(4) = -100;
    
    ub = lb*-1;
    %% quadprog analysis
    %SRR update, make sure you double H matrix by 2, MATLAB will half it.
%     linear_uopt = quadprog(2*Hmat,f_vec,Aineq,bineq,Aeq,beq,lb,ub);
    
    %% Gurobi Definition
    % Trying to minimize xT*Q*x + qT*x 
    model.obj = f_vec; % linear matrix in minimize problem statement
    model.Q = sparse(Hmat);  % quadratic matrix in minimize problem statement
    model.A = sparse([Aeq;Aineq]); % LHS of linear constraints
    model.rhs = [beq;bineq]; % RHS of linear constraints
    
    % SRR - Nonlinear constraint is of the form: xT*Q*x + qT*x = b , where
    % b is const, x is state (34 dimensional in N=2 case).
    %       Notes:  Q = Q_nonlin_eq_constraint
    Q_nonlin_eq_constraint1 = zeros(34);
    Q_nonlin_eq_constraint2 = zeros(34);
    Q_nonlin_eq_constraint3 = zeros(34);
    Q_nonlin_eq_constraint4 = zeros(34);    
    Q_nonlin_eq_constraint5 = zeros(34);    
    Q_nonlin_eq_constraint6 = zeros(34);    
    Q_nonlin_eq_constraint7 = zeros(34);    
    Q_nonlin_eq_constraint8 = zeros(34);    
    Q_nonlin_eq_constraint9 = zeros(34);    
    Q_nonlin_eq_constraint10 = zeros(34);    
    Q_nonlin_eq_constraint11 = zeros(34);    
    Q_nonlin_eq_constraint12 = zeros(34);  
    Q_nonlin_eq_constraint13 = zeros(34);    
    Q_nonlin_eq_constraint14 = zeros(34); 
    Q_nonlin_eq_constraint15 = zeros(34); 
    Q_nonlin_ineq_constraint1 = zeros(34);    

    q_nonlin_eq_constraint1 = zeros(34,1);
    q_nonlin_eq_constraint2 = zeros(34,1);
    q_nonlin_eq_constraint3 = zeros(34,1);
    q_nonlin_eq_constraint4 = zeros(34,1);
    q_nonlin_eq_constraint5 = zeros(34,1);
    q_nonlin_eq_constraint6 = zeros(34,1);
    q_nonlin_eq_constraint7 = zeros(34,1);
    q_nonlin_eq_constraint8 = zeros(34,1);
    q_nonlin_eq_constraint9 = zeros(34,1);
    q_nonlin_eq_constraint10 = zeros(34,1);
    q_nonlin_eq_constraint11 = zeros(34,1);
    q_nonlin_eq_constraint12 = zeros(34,1);
    q_nonlin_eq_constraint13 = zeros(34,1);
    q_nonlin_ineq_constraint1 = zeros(34,1);
    

    % SRR - Formulating Constraint: R(t+1)*z(t+1) - Phi(t+1) = 0  
    %       Note that this is 3 by 1 matrix system 
    % This gives 3 quadratic equations. We need 3 quadcon since dim(xT*Q*x) = 1
    % SRR- first eqn R11*z1 + R12*z2 + R13*z3 - Phi1 = 0 (all are in t + 1)
    Q_nonlin_eq_constraint1(26,14) =  1.0;
    Q_nonlin_eq_constraint1(27,15) =  1.0;
    Q_nonlin_eq_constraint1(28,16) =  1.0;
    q_nonlin_eq_constraint1(5,1) = -1.0;
    model.quadcon(1).Qc = sparse(Q_nonlin_eq_constraint1);
    model.quadcon(1).q = q_nonlin_eq_constraint1;
    model.quadcon(1).rhs = 0.0;
    model.quadcon(1).sense = '=';
    % second eqn R21*z1 + R22*z2 + R23*z3 - Phi2 = 0 (all in t+1)
    Q_nonlin_eq_constraint2(29,14) =  1.0;
    Q_nonlin_eq_constraint2(30,15) =  1.0;
    Q_nonlin_eq_constraint2(31,16) =  1.0;
    q_nonlin_eq_constraint2(6,1) = -1.0;
    model.quadcon(2).Qc = sparse(Q_nonlin_eq_constraint2);
    model.quadcon(2).q = q_nonlin_eq_constraint2;
    model.quadcon(2).rhs = 0.0;
    model.quadcon(2).sense = '=';
    % third eqn R31*ze + R32*z2 + R33*z3 - Phi3 = 0 (all in t+1)
    Q_nonlin_eq_constraint2(32,14) =  1.0;
    Q_nonlin_eq_constraint2(33,15) =  1.0;
    Q_nonlin_eq_constraint2(34,16) =  1.0;
    q_nonlin_eq_constraint2(7,1) = -1.0;
    model.quadcon(3).Qc = sparse(Q_nonlin_eq_constraint3);
    model.quadcon(3).q = q_nonlin_eq_constraint3;
    model.quadcon(3).rhs = 0.0;
    model.quadcon(3).sense = '=';
    % SRR - Formulating Constraint:  R(t+1) = R(t) + Phi(t+1)*Phi'(t+1)
    %                             => R(t+1) - Phi(t+1)*Phi'(t+1) = R(t)
    %       Note that this is 3 by 3 matrix system, so there are 9
    %       quadratic equations here. Treating R(t) as known (initalized)
    % SRR- R11 -Phi11^2  = R11_t0 (LHS in t+1, RHS in t)
    Q_nonlin_eq_constraint4(5,5) = -1.0;
    q_nonlin_eq_constraint4(26,1) = 1.0;
    model.quadcon(4).Qc = sparse(Q_nonlin_eq_constraint4);
    model.quadcon(4).q = q_nonlin_eq_constraint4;
    model.quadcon(4).rhs = R_mat(1,1);
    model.quadcon(4).sense = '=';
    % SRR- R12 - Phi1*Phi2 = R12_t0 (LHS in t+1, RHS in t)
    Q_nonlin_eq_constraint5(5,6) = -1.0;
    q_nonlin_eq_constraint5(27,1) = 1.0;
    model.quadcon(5).Qc = sparse(Q_nonlin_eq_constraint5);
    model.quadcon(5).q = q_nonlin_eq_constraint5;
    model.quadcon(5).rhs = R_mat(1,2);
    model.quadcon(5).sense = '=';
    % SRR- R13 - Phi1*Phi3 = R13_t0 (LHS in t+1, RHS in t)
    Q_nonlin_eq_constraint6(5,7) = -1.0;
    q_nonlin_eq_constraint6(28,1) = 1.0;
    model.quadcon(6).Qc = sparse(Q_nonlin_eq_constraint6);
    model.quadcon(6).q = q_nonlin_eq_constraint6;
    model.quadcon(6).rhs = R_mat(1,3);
    model.quadcon(6).sense = '=';
    % SRR- R21 - Phi2*Phi1 = R21_t0 (LHS in t+1, RHS in t)
    Q_nonlin_eq_constraint7(6,5) = -1.0;
    q_nonlin_eq_constraint7(29,1) = 1.0;
    model.quadcon(7).Qc = sparse(Q_nonlin_eq_constraint7);
    model.quadcon(7).q = q_nonlin_eq_constraint7;
    model.quadcon(7).rhs = R_mat(2,1);
    model.quadcon(7).sense = '=';
    % SRR- R22 - Phi2*Phi2 = R22_t0 (LHS in t+1, RHS in t)
    Q_nonlin_eq_constraint8(6,6) = -1.0;
    q_nonlin_eq_constraint8(30,1) = 1.0;
    model.quadcon(8).Qc = sparse(Q_nonlin_eq_constraint8);
    model.quadcon(8).q = q_nonlin_eq_constraint8;
    model.quadcon(8).rhs = R_mat(2,2);
    model.quadcon(8).sense = '=';
    % SRR- R23 - Phi2*Phi3 = R23_t0 (LHS in t+1, RHS in t)
    Q_nonlin_eq_constraint9(6,7) = -1.0;
    q_nonlin_eq_constraint9(31,1) = 1.0;
    model.quadcon(9).Qc = sparse(Q_nonlin_eq_constraint9);
    model.quadcon(9).q = q_nonlin_eq_constraint9;
    model.quadcon(9).rhs = R_mat(2,3);
    model.quadcon(9).sense = '=';
    % SRR- R31 - Phi3*Phi1 = R31_t0 (LHS in t+1, RHS in t)
    Q_nonlin_eq_constraint10(7,5) = -1.0;
    q_nonlin_eq_constraint10(32,1) = 1.0;
    model.quadcon(10).Qc = sparse(Q_nonlin_eq_constraint10);
    model.quadcon(10).q = q_nonlin_eq_constraint10;
    model.quadcon(10).rhs = R_mat(3,1);
    model.quadcon(10).sense = '=';
    % SRR- R32 - Phi3*Phi2 = R32_t0 (LHS in t+1, RHS in t)
    Q_nonlin_eq_constraint11(7,6) = -1.0;
    q_nonlin_eq_constraint11(33,1) = 1.0;
    model.quadcon(11).Qc = sparse(Q_nonlin_eq_constraint11);
    model.quadcon(11).q = q_nonlin_eq_constraint11;
    model.quadcon(11).rhs = R_mat(3,2);
    model.quadcon(11).sense = '=';
    % SRR- R33 - Phi3*Phi3 = R33_t0 (LHS in t+1, RHS in t)
    Q_nonlin_eq_constraint12(7,7) = -1.0;
    q_nonlin_eq_constraint12(34,1) = 1.0;
    model.quadcon(12).Qc = sparse(Q_nonlin_eq_constraint12);
    model.quadcon(12).q = q_nonlin_eq_constraint12;
    model.quadcon(12).rhs = R_mat(3,3);
    model.quadcon(12).sense = '=';
    
    % SRR - Formulating Constraint: Phi'(t+1)*z(t+1) >= 0 
    %       Note that this is a scalar equation, so there is only 1
    %       quadratic equation here.
    % SRR- Phi1*z1 + Phi2*z2 + Phi3*z3 >= 0 (LHS in t+1)    
    Q_nonlin_ineq_constraint1(5, 14) = 1.0;
    Q_nonlin_ineq_constraint1(6, 15) = 1.0;
    Q_nonlin_ineq_constraint1(7, 16) = 1.0;
    model.quadcon(13).Qc = sparse(Q_nonlin_eq_constraint1);
    model.quadcon(13).q = q_nonlin_ineq_constraint1;
    model.quadcon(13).rhs = 0.0;
    model.quadcon(13).sense = '>';
    
    % SRR - OPTIONAL: Formulating Constraint for z(t). R(t)*z(t) = Phi(t), for k =0. 
    % Example row: R(1,1)*z(t, element1) + R(1,2)*z(t, element2) + 
    % R(1,3)*z(t, element) = Phi(t, element 1)
    Q_nonlin_eq_constraint13(17,11)= 1.0;
    Q_nonlin_eq_constraint13(18,12)= 1.0;
    Q_nonlin_eq_constraint13(19,13)= 1.0;
    model.quadcon(14).Qc = sparse(Q_nonlin_eq_constraint13);
    model.quadcon(14).q = q_nonlin_eq_constraint13;
    model.quadcon(14).rhs = xt(1);
    model.quadcon(14).sense = '=';
    
    Q_nonlin_eq_constraint14(20,11)= 1.0;
    Q_nonlin_eq_constraint14(21,12)= 1.0;
    Q_nonlin_eq_constraint14(22,13)= 1.0;
    model.quadcon(15).Qc = sparse(Q_nonlin_eq_constraint14);
    model.quadcon(15).q = q_nonlin_eq_constraint13;
    model.quadcon(15).rhs = xt(2);
    model.quadcon(15).sense = '=';
    
    Q_nonlin_eq_constraint15(23,11)= 1.0;
    Q_nonlin_eq_constraint15(24,12)= 1.0;
    Q_nonlin_eq_constraint15(25,13)= 1.0;
    model.quadcon(16).Qc = sparse(Q_nonlin_eq_constraint15);
    model.quadcon(16).q = q_nonlin_eq_constraint13;
    model.quadcon(16).rhs = xt(3);
    model.quadcon(16).sense = '=';
    

    ne = size(Aeq,1);
    ni = size(Aineq,1);
    model.sense = [repmat('=',ne,1);repmat('<',ni,1)];
    model.lb = lb; % lower bound
    model.ub = ub; %upper bound. 
    params.outputflag = 1;
    params.NonConvex = 2;
    % params.DualReductions = 0;
    result = gurobi(model, params);
    disp(result.status)
    linear_uopt = result.x;
    %%
    ustar(:,i) = linear_uopt(3);
    
    % UPDATE SRR - y_measured is the "output measurement", from page 1 
    % system(1) in the  paper. reworked this equation to match the eqns.
    y_measured(:,i) = theta_truth'*xt + (sqrt(variance) * randn(1,1)); 
    y_hat = y_measured(:,i);
%   AS Fixing propagation  
    xstar(:,i) = (A*xt+B*ustar(:,i));   %Linearized Model Update
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
xlabel('x position')
ylabel('y position')
title('x-y plot')

figure
hold on
plot(time_series(1:end), optimal_x(2,:))
xlabel('Time')
ylabel('y position')
title('y plot')

figure
hold on
plot(time_series(1:end), optimal_x(1,:))
xlabel('Time')
ylabel('x position')
title('x plot')

figure
hold on
plot(time_series(1:end),y_measured(1,:),'r')
xlabel('time')
ylabel('output y')
title ('output y  measured as a function of time')

figure
hold on
plot(time_series(1:end),ustar(1,:))
title('u star control plot')




           
