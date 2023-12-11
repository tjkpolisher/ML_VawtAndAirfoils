clear all
close all
TSR = [2.0, 2.5, 3.0];
filename ={'report-def-0-rfile_TSR2.0Cd.out','report-def-1-rfile_TSR2.0Cl.out',...
    'report-def-0-rfile_TSR3.0Cd.out','report-def-1-rfile_TSR3.0Cl.out'};
data_path = 'C:\Users\cfdML\OneDrive\문서\머신러닝\TSR2랑3\';
for i=1:1

    data_d_path1='D:\airfoilFluent\airfoilSimulations\NACA0021(TSR2.5)(10s)\NACA0021(TSR2.5)(10s)_files\dp0\FFF-15\Fluent\report-def-0-rfile.txt';
    data_d1=readmatrix(data_d_path1);
    data_d_path2='D:\airfoilFluent\airfoilSimulations\NACA0021(TSR2.5)(10s)\NACA0021(TSR2.5)(10s)_files\dp0\FFF-15\Fluent\report-def-1-rfile.txt';
    data_d2=readmatrix(data_d_path2);

    Cd = data_d1(:,2);
    Cl = data_d2(:,2);
%     iteration = data_d1(:,1);
    time = data_d1(:,3);
    figure(i)
    plot(time, Cd, "Color",'b')
    hold on
    grid on
    plot(time, Cl, "Color",'r')
    legend('$C_d$', '$C_l$','interpreter','latex')
    title(char(["TSR 2.5, alpha 30, unsteady"]))
    xlabel('Time [s]')
    ylabel('$C_d$, $C_l$','interpreter','latex')
    %xlim([1.8 2.0])
    ylim([0 3])


end