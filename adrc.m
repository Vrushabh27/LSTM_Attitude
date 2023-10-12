% Simulation parameters
clear all
Data=readtable("SCA1B_2022-12-01_C_04_losdiff_GNI1B.txt");
Data=table2array(Data(:,3:end));
q_actual=Data(:,2:5);
q_desired=Data(:,8:11);
error=[];
for i=1:length(q_desired)
error(:,i)=quatmultiply([q_desired(i,4) -q_desired(i,1:3)],[q_actual(i,4) q_actual(i,1:3)]);
end
t_N=10000;
tspan = [0, t_N];
a1=error(2:4,1);%0.2*rand(1);
q_init=[sqrt(1-3*norm(error(2:4,1))^2);a1];
omega_init=1e-3*rand(3,1);
x0 = [omega_init;q_init; 1; 0; 0; 0; 0; 0; 0;0;0];
x0 = [omega_init;q_init; 0; 0; 0; 0; 0; 0; 0;0;0];
[t_PID, x_PID] = ode45(@(t, x) spacecraft_dynamics(t, x,"PID",error), tspan, x0);
% [t_ADRC, x_ADRC] = ode45(@(t, x) spacecraft_dynamics(t, x,"ADRC",error), tspan, x0);
% [t_PID, x_PID] = ode45(@(t, x) spacecraft_dynamics(t, x,"PID"), tspan, x0);
% Plot results
eul_pid=quat2eul(x_PID(:,4:7));
eul_adrc=quat2eul(x_ADRC(:,4:7));
omega_pid=norm(x_PID(end-10:end, 1:3))
omega_adrc=norm(x_ADRC(end-10:end, 1:3))
q_pid=norm(eul_pid(end-10:end, 1:3))
q_adrc=norm(eul_adrc(end-10:end, 1:3))


figure(1)
plot(t_ADRC, x_ADRC(:, 1:1)-x_ADRC(:,7),'Linewidth',1.5);hold on;
plot(t_PID, x_PID(:, 1:1)-x_PID(:,7),'Linewidth',1.5);
title('Omega Error - ADRC','FontSize', 20);
xlabel('Time (s)','FontSize', 20);
ylabel('Error (rad/s)','FontSize', 20);
legend('$\omega_1$', '$\omega_2$', '$\omega_3$','interpreter','latex');
xlim([0 t_N])
grid on;
% ZoomPlot()

figure(2)
plot(t_PID, x_PID(:, 1:1)-x_PID(:,7),'Linewidth',1.5);
title('Omega Error - PID','FontSize', 20);
xlabel('Time (s)','interpreter','latex','FontSize', 20);
ylabel('Error (rad/s)','interpreter','latex','FontSize', 20);
legend('$\omega_1$', '$\omega_2$', '$\omega_3$','interpreter','latex');
xlim([0 t_N]);grid on;
% ZoomPlot()

figure(3)
plot(t_ADRC, eul_adrc(:, 1:3),'Linewidth',1.5);hold on;
title('Euler Error - ADRC');
xlabel('Time (s)','interpreter','latex','FontSize', 20);
ylabel('Error (rad)','interpreter','latex','FontSize', 20);
legend('$\phi$', '$\theta$', '$\psi$','interpreter','latex','FontSize', 20);
xlim([0 t_N])
grid on;
% ZoomPlot

figure(4)
plot(t_PID, eul_pid(:, 1:3),'Linewidth',1.5);
title('Euler Error - PID');
xlabel('Time (s)','interpreter','latex','FontSize', 20);
ylabel('Error (rad)','interpreter','latex','FontSize', 20);
legend('$\phi$', '$\theta$', '$\psi$','interpreter','latex','FontSize', 20);
xlim([0 t_N])
grid on;
% ZoomPlot






omega_pid=norm(x_PID(end-10:end, 1:3))
omega_adrc=norm(x_ADRC(end-10:end, 1:3))
q_pid=norm(eul_pid(end-10:end, 1:3))
q_adrc=norm(eul_adrc(end-10:end, 1:3))
diff_q=q_adrc-q_pid
function [dx] = spacecraft_dynamics(t, x,controller,error)
    % Parameters
    J = diag([1000, 2000, 3000])*1e-2; % Inertia matrix
    w = x(1:3); % Angular velocity
    q = x(4:7); % Quaternion
    x1hat=x(8:10);
    x2hat=x(11:13);
    x3hat=x(14:16);
    % Quaternion kinematics
    OMEGA = [0, -w(1), -w(2), -w(3);
             w(1), 0, w(3), -w(2);
             w(2), -w(3), 0, w(1);
             w(3), w(2), -w(1), 0];
    dq = 0.5 * OMEGA * q;
    Kp=10; Kd=10;
    w=w+error(2:4,floor(t)+1);
      q=q+error(1:4,floor(t)+1);
if controller=="ADRC"
u=-J*(x3hat+Kp*q(2:4) + Kd * w);
% u=-J*(x3hat+Kp*x2hat+ Kd * x1hat);
% cross(x(1:3), J * x(1:3))
% u=-J*(x3hat+Kp*());
else
u=-J*(Kp*q(2:4) + Kd * w);    
end
    % Spacecraft dynamics
ESO_gain = 5.8;epsilon=0.1;
 dx1hat=x2hat+3*ESO_gain*(w-x1hat)/epsilon;
 dx2hat=x3hat+3*ESO_gain*(w-x1hat)/epsilon^2+inv(J)*u;
 dx3hat=ESO_gain*(w-x1hat)/epsilon^3;
% dx1hat=dx2hat+ESO_gain*([w;q]-x1hat)/epsilon;
% dx2hat=ESO_gain*([w;q]-x1hat)/epsilon^3;
dz=[dx1hat;dx2hat;dx3hat];


dw = inv(J) * (u - cross(w, J *w))+error(2:4,floor(t)+1);%+ 0.3*sin(rand(1)*t);
    % Combined state derivatives
    dx = [dw; dq;dz];
end
