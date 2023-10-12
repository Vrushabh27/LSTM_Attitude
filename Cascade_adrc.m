% Simulation parameters
clear all
global d_lstm
tic
l=0.001;
% Data=readtable("SCA1B_2022-12-01_C_04_losdiff_GNI1B.txt");
d_lstm=load("d_predictions.mat");
d_lstm=d_lstm.y_test_pred;
d_lstm=ones(3,1)*d_lstm;
% Data=table2array(Data(:,3:end));
% q_actual=Data(:,2:5);
% q_desired=Data(:,8:11);
% error=[];
% for i=1:length(q_desired)
% error(:,i)=quatmultiply([q_desired(i,4) -q_desired(i,1:3)],[q_actual(i,4) q_actual(i,1:3)]);
% end
% error=quat2eul(error');error=error';
error=load("error1.mat");
error=error.error1;error=ones(3,1)*error;
x_init_dyn=1e-3*rand(6,1);
x0=[x_init_dyn;x_init_dyn;zeros(3,1);x_init_dyn;zeros(3,1)];
t_N=100;
tspan = [0, t_N];
[t_PID, x_PID] = ode45(@(t, x) spacecraft_dynamics(t, x,"PID",error), tspan, x0);
% [t_ADRC, x_ADRC] = ode23s(@(t, x) spacecraft_dynamics(t, x,"ADRC",error), tspan, x0);

n=min(length(x_PID));



figure(2)
figure('WindowState','maximized');
set(gca,'FontSize',30);hold on;
% n=1000;
f=20;
% n=nn-700;
% subplot(3,2,1)
xL=xlim;
yL=ylim;
% text(0.99*xL(2),0.99*yL(2),"RMS",'HorizontalAlignment','right','VerticalAlignment','top');hold on
set(gca,'FontSize',f);hold on;
plot(t_PID/3600, x_PID(:, 4)*1e6,'Linewidth',1.5);grid on;grid on;
title('Omega Error - PID (RMS=7.472e-6)','FontSize', 20);
xlabel('Time (hrs)','interpreter','latex','FontSize', 20);
ylabel('Error ($\mu$rad/s)','interpreter','latex','FontSize', 20);
legend('$\omega_1$','interpreter','latex');
% xlim([0 t_N]);grid on;


% 
% figure(3)
% figure('WindowState','maximized');
% subplot(3,2,1)
% set(gca,'FontSize',f);hold on;
% 
% plot(t_PID(end-n:end)/3600, x_PID(end-n:end, 1)*1e3,'Linewidth',1.5);grid on;
% title('Euler Error - PID (RMS=9.1481e-4)','FontSize', 20);
% xlabel('Time (hrs)','interpreter','latex','FontSize', 20);
% ylabel('Error (mrad)','interpreter','latex','FontSize', 20);
% legend('$\theta_1$','interpreter','latex');
% % xlim([0 t_N]);grid on;
% subplot(3,2,2)
% set(gca,'FontSize',f);hold on;
% 
% plot(t_ADRC(end-n:end)/3600, x_ADRC(end-n:end, 1)*1e3,'Linewidth',1.5);grid on;
% title('Euler Error - ADRC (RMS=4.969e-4)','FontSize', 20);
% xlabel('Time (hrs)','interpreter','latex','FontSize', 20);
% ylabel('Error (mrad)','interpreter','latex','FontSize', 20);
% legend('$\theta_1$','interpreter','latex');
% % xlim([0 t_N]);grid on;
% subplot(3,2,3)
% set(gca,'FontSize',f);hold on;
% 
% plot(t_PID(end-n:end)/3600, x_PID(end-n:end, 2)*1e3,'Linewidth',1.5);grid on;
% % title('Omega Error - PID','FontSize', 20);
% xlabel('Time (hrs)','interpreter','latex','FontSize', 20);
% ylabel('Error (mrad)','interpreter','latex','FontSize', 20);
% legend('$\theta_2$','interpreter','latex');
% % xlim([0 t_N]);grid on;
% subplot(3,2,4)
% set(gca,'FontSize',f);hold on;
% 
% plot(t_ADRC(end-n:end)/3600, x_ADRC(end-n:end, 2)*1e3,'Linewidth',1.5);grid on;
% % title('Omega Error - PID','FontSize', 20);
% xlabel('Time (hrs)','interpreter','latex','FontSize', 20);
% ylabel('Error (mrad)','interpreter','latex','FontSize', 20);
% legend('$\theta_2$','interpreter','latex');
% % xlim([0 t_N]);grid on;
% subplot(3,2,5)
% set(gca,'FontSize',f);hold on;
% 
% plot(t_PID(end-n:end)/3600, x_PID(end-n:end, 3)*1e3,'Linewidth',1.5);grid on;
% % title('Omega Error - PID','FontSize', 20);
% xlabel('Time (hrs)','interpreter','latex','FontSize', 20);
% ylabel('Error (mrad)','interpreter','latex','FontSize', 20);
% legend('$\theta_3$','interpreter','latex');
% % xlim([0 t_N]);grid on;
% subplot(3,2,6)
% set(gca,'FontSize',f);hold on;
% 
% plot(t_ADRC(end-n:end)/3600, x_ADRC(end-n:end, 3)*1e3,'Linewidth',1.5);grid on;
% % title('Omega Error - PID','FontSize', 20);
% xlabel('Time (hrs)','interpreter','latex','FontSize', 20);
% ylabel('Error (mrad)','interpreter','latex','FontSize', 20);
% legend('$\theta_3$','interpreter','latex');
% saveas(gcf,'euler_error','epsc')
% 
% 
% figure(4)
% figure('WindowState','maximized');
% subplot(3,2,1)
% set(gca,'FontSize',f);hold on;
% plot(t_PID(end-n:end)/3600, error(1, floor(t_PID(end-n:end))+1)'*1e6,'Linewidth',1.5);grid on;
% title('Disturbance','FontSize', 20);
% xlabel('Time (hrs)','interpreter','latex','FontSize', 20);
% ylabel('$d_1(\mu$ rad)','interpreter','latex','FontSize', 20);
% legend('$d_1(\mu$ rad)','interpreter','latex');
% % xlim([0 t_N]);grid on;
% subplot(3,2,2)
% set(gca,'FontSize',f);hold on;
% 
% % plot(t_ADRC(end-n:end), error(1, floor(t_ADRC(end-n:end))+1).*(1+sin(t_ADRC(end-n:end))'),'Linewidth',1.5);grid on;
% plot(t_ADRC(end-n:end)/3600, x_ADRC(end-n:end,13)*1e6,'Linewidth',1.5);grid on;
% 
% title('Estimated disturbance','FontSize', 20);
% xlabel('Time (hrs)','interpreter','latex','FontSize', 20);
% ylabel('$d_1(\mu$ rad)','interpreter','latex','FontSize', 20);
% legend('$d_1(\mu$ rad)','interpreter','latex');
% % xlim([0 t_N]);grid on;
% subplot(3,2,3)
% set(gca,'FontSize',f);hold on;
% 
% plot(t_PID(end-n:end)/3600, error(2, floor(t_PID(end-n:end))+1)'*1e6,'Linewidth',1.5);grid on;
% % title('Omega Error - PID','FontSize', 20);
% xlabel('Time (hrs)','interpreter','latex','FontSize', 20);
% ylabel('$d_2(\mu$ rad)','interpreter','latex','FontSize', 20);
% legend('$d_2(\mu$ rad)','interpreter','latex');
% % xlim([0 t_N]);grid on;
% subplot(3,2,4)
% set(gca,'FontSize',f);hold on;
% 
% plot(t_ADRC(end-n:end)/3600, x_ADRC(end-n:end, 14)*1e6,'Linewidth',1.5);grid on;
% % title('Omega Error - PID','FontSize', 20);
% xlabel('Time (hrs)','interpreter','latex','FontSize', 20);
% ylabel('$d_2(\mu$ rad)','interpreter','latex','FontSize', 20);
% legend('$d_2(\mu$ rad)','interpreter','latex');
% % xlim([0 t_N]);grid on;
% subplot(3,2,5)
% set(gca,'FontSize',f);hold on;
% 
% plot(t_PID(end-n:end)/3600, error(3, floor(t_PID(end-n:end))+1)'*1e6,'Linewidth',1.5);grid on;
% % title('Omega Error - PID','FontSize', 20);
% xlabel('Time (hrs)','interpreter','latex','FontSize', 20);
% ylabel('$d_3(\mu$ rad)','interpreter','latex','FontSize', 20);
% legend('$d_3(\mu$ rad)','interpreter','latex');
% % xlim([0 t_N]);grid on;
% subplot(3,2,6)
% set(gca,'FontSize',f);hold on;
% 
% plot(t_ADRC(end-n:end)/3600, x_ADRC(end-n:end, 15)*1e6,'Linewidth',1.5);grid on;
% % title('Omega Error - PID','FontSize', 20);
% xlabel('Time (hrs)','interpreter','latex','FontSize', 20);
% ylabel('$d_3(\mu$ rad)','interpreter','latex','FontSize', 20);
% legend('$d_3(\mu$ rad)','interpreter','latex');
% % xlim([0 t_N]);grid on;
% saveas(gcf,'d_error','epsc')


toc

function [dx] = spacecraft_dynamics(t, x,controller,error)
global d_lstm
    % Parameters
xdyn=x(1:6);
eta1=x(7:15);
eta2=x(16:24);
    Kp=2; Kd=2;
I=eye(3);
sigma1=(I(2,2)-I(3,3))/I(1,1);
sigma2=(I(3,3)-I(1,1))/I(2,2);
sigma3=(I(1,1)-I(2,2))/I(3,3);
omega0=1;l=0.001;
zhat=eta1(7:9)+eta2(7:9);
y=xdyn(1:3)+error(1:3,floor(t)+1).*(exp(l*t)./exp(l*t));%+0.00001*rand(3,1);%cplus*xdyn;%xdyn(1:3)+0.000001*rand(3,1);

% else
% u=-I*(Kp*y + Kd * eta1(4:6)); 
u=-I*(Kp*y(1:3) + Kd * xdyn(4:6))+I*Kp*d_lstm(floor(t)+1)*ones(3,1); 

Adyn=[zeros(3,3) eye(3);-4*sigma1*omega0^2 0 0 0 0 omega0*(1-sigma1);0 3*sigma1*omega0^2 zeros(1,4);0 0 sigma3*omega0^2 -omega0*(1-sigma2) 0 0 ];
Adyn=[zeros(3,3) eye(3);Adyn(4:6,1:6)];
B=[zeros(3,3);inv(I)];
% w=x(4:6);
% q=x(1:4);
%     OMEGA = [0, -w(1), -w(2), -w(3);
%              w(1), 0, w(3), -w(2);
%              w(2), -w(3), 0, w(1);
%              w(3), w(2), -w(1), 0];
%     dq = 0.5 * OMEGA * q;
% dw = inv(I) * (u - cross(w, I *w));


epsilon=1;
gain1=2;gain2=0.5*gain1^2;gain3=0.5*gain1^3;
gain1=15;gain2=2;gain3=0.1;
Aplus=blkdiag(Adyn,zeros(3,3));%[Adyn zeros(6,3);zeros(3,3) zeros(3,3)];
dplus=[zeros(3,3);inv(I);zeros(3,3)];
alpha=3;
gain=[100;20;740;alpha*150;alpha*29;alpha*0.05];
gain=[300;30;304;alpha*150;alpha*29;alpha*0.05];
lplus=[gain(1)*eye(3);gain(2)*eye(3);gain(3)*eye(3)];
cplus=[eye(3);zeros(3,3)];

xdyndot=Adyn*xdyn+B*u;%+[0;0;0;0.000001*rand(3,1)];%0.0001;%[0;0;0;error(2:4,floor(t)+1)];%+[0;0;0;0.00001*rand(3,1)];
eta1dot=Aplus*eta1+dplus*u+lplus*(y-eta1(1:3));
lplus2=[gain(4)*eye(3);gain(5)*eye(3);gain(6)*eye(3)];
eta2dot=Aplus*eta2+dplus*u+[zeros(3,1);eta1(4:6);zeros(3,1)]+lplus2*(eta1(1:3)-eta2(1:3));


dx=[xdyndot;eta1dot;eta2dot];
end





% 
%     J = diag([1000, 2000, 3000])*1e-2; % Inertia matrix
%     w = x(1:3); % Angular velocity
%     q = x(4:7); % Quaternion
%     x1hat=x(8:10);
%     x2hat=x(11:13);
%     x3hat=x(14:16);
%     % Quaternion kinematics
%     OMEGA = [0, -w(1), -w(2), -w(3);
%              w(1), 0, w(3), -w(2);
%              w(2), -w(3), 0, w(1);
%              w(3), w(2), -w(1), 0];
%     dq = 0.5 * OMEGA * q;
%     Kp=10; Kd=10;
%     w=w+error(2:4,floor(t)+1);
%       q=q+error(1:4,floor(t)+1);
% if controller=="ADRC"
% u=-J*(x3hat+Kp*q(2:4) + Kd * w);
% % u=-J*(x3hat+Kp*x2hat+ Kd * x1hat);
% % cross(x(1:3), J * x(1:3))
% % u=-J*(x3hat+Kp*());
% else
% u=-J*(Kp*q(2:4) + Kd * w);    
% end
%     % Spacecraft dynamics
% ESO_gain = 0.8;epsilon=0.1;
%  dx1hat=x2hat+3*ESO_gain*(w-x1hat)/epsilon;
%  dx2hat=x3hat+3*ESO_gain*(w-x1hat)/epsilon^2+inv(J)*u;
%  dx3hat=ESO_gain*(w-x1hat)/epsilon^3;
% % dx1hat=dx2hat+ESO_gain*([w;q]-x1hat)/epsilon;
% % dx2hat=ESO_gain*([w;q]-x1hat)/epsilon^3;
% dz=[dx1hat;dx2hat;dx3hat];
% 
% 
% dw = inv(J) * (u - cross(w, J *w))+error(2:4,floor(t)+1);%+ 0.3*sin(rand(1)*t);
%     % Combined state derivatives
%     dx = [dw; dq;dz];

