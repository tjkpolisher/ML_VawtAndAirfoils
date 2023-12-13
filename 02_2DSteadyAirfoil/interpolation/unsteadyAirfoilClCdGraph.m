clear all

% filename = 'report-def-0-rfile_4500.out';
% fileID = fopen(filename, 'w');
% formatSpec = '%4.9f';
% C = textscan(fileID,formatSpec);
C = load('C:\Users\tjk\OneDrive\문서\머신러닝\AirfoilClCdFiles\unsteadyNACA0018cl-1-history.txt');
lift = C(:,2);
liftReference = 0.9*ones(1000,1);
timeStep1 = C(:,1);

D = load('C:\Users\tjk\OneDrive\문서\머신러닝\AirfoilClCdFiles\unsteadyNACA0018cd-1-history.txt');
drag = D(:,2);
dragReference = 0.02*ones(1000,1);
timeStep2 = C(:,1);

close all
figure(1)
set(gcf,'Position',[430 100 1200 900])
set(gcf,'DefaultLineLineWidth',2)
set(gcf,'DefaultAxesFontName','Times new roman')
set(gcf,'DefaultAxesFontSize', 25)
set(gcf,'DefaultTextFontName','Times new roman')
plot(timeStep1, lift,'Color','b')
hold on
plot(timeStep1, liftReference, 'k--')
% plot(ang_x03,Ft1_xd03_l25/(0.5*1.255*9^2*0.00103),'Color','[0.6350 0.0780 0.1840]') % brown RGB: [0.6350 0.0780 0.1840]
% plot(ang_x05,Ft1_xd05_l25/(0.5*1.255*9^2*0.00103), 'Color','m')
% plot(ang_x07,Ft1_xd07_l25/(0.5*1.255*9^2*0.00103),'Color','g')
% plot(ang_x09,Ft1_xd09_l25/(0.5*1.255*9^2*0.00103),'Color','r')
% plot(ang_no_dimple,Ft1_no_dimple/(0.5*1.255*9^2*0.00103),'Color','k','LineStyle','--')
title('Time - $C_l$ ($\alpha$ = $10^\circ$)', 'Interpreter', 'LaTeX')
legend('Fluent (unsteady)','Experimental data','Interpreter', 'LaTeX','fontsize', 30)
%xlim([0 16])
ylim([0 1.2])
% xticks([0:45:360])
xlabel('Time (sec)', 'Interpreter', 'LaTeX')
ylabel('$C_l$', 'Interpreter', 'LaTeX')
grid on
f1 = gcf;
exportgraphics(f1, '20220725Time-Cl(unsteady, 6second, alpha10).jpg','Resolution',300)
%exportgraphics(f1, 'Time-Cl(unsteady, 5second, alpha20).eps','Resolution',300)

figure(2)
set(gcf,'Position',[430 100 1200 900])
set(gcf,'DefaultLineLineWidth',2)
set(gcf,'DefaultAxesFontName','Times new roman')
set(gcf,'DefaultAxesFontSize', 25)
set(gcf,'DefaultTextFontName','Times new roman')
plot(timeStep2, drag,'Color','b')
hold on
plot(timeStep2, dragReference, 'k--')
% plot(ang_x03,Ft1_xd03_l25/(0.5*1.255*9^2*0.00103),'Color','[0.6350 0.0780 0.1840]') % brown RGB: [0.6350 0.0780 0.1840]
% plot(ang_x05,Ft1_xd05_l25/(0.5*1.255*9^2*0.00103), 'Color','m')
% plot(ang_x07,Ft1_xd07_l25/(0.5*1.255*9^2*0.00103),'Color','g')
% plot(ang_x09,Ft1_xd09_l25/(0.5*1.255*9^2*0.00103),'Color','r')
% plot(ang_no_dimple,Ft1_no_dimple/(0.5*1.255*9^2*0.00103),'Color','k','LineStyle','--')
title('Time - $C_d$ ($\alpha$ = $10^\circ$)', 'Interpreter', 'LaTeX')
legend('Fluent (unsteady)','Experimental data','Interpreter', 'LaTeX','fontsize', 30)
%xlim([0 16])
ylim([0 0.2])
% xticks([0:45:360])
xlabel('Time (sec)', 'Interpreter', 'LaTeX')
ylabel('$C_d$', 'Interpreter', 'LaTeX')
grid on
f2 = gcf;
exportgraphics(f2, '20220725Time-Cd(unsteady, 6second, alpha10).jpg','Resolution',300)
%exportgraphics(f2, 'Time-Cd(unsteady, 5second, alpha20).eps','Resolution',300)