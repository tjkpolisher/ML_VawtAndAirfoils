clear all

x=[-15 -13 -11 -9 -7 -5 -3 -1 1 3 5 7 9 11 13 15];
cl = [-0.83927438 -0.7927162 -0.71376611 -0.57897818 -0.37853146 -0.15189397 0.062265541 0.30396389 0.52564567 0.7482871 0.92587521 1.0999253 1.2413888 1.329638 1.3674946 1.3632917];
cd = [0.061852953 0.046767259 0.033480603 0.02463768 0.018364832 0.015317716 0.014267085 0.013890308 0.015223818 0.016577271 0.022332372 0.02686945 0.033729821 0.046135507 0.063351656 0.10981444];
%cpboth = [0.2709 0.3231 0.3514 0.3628];
%ref = [0.3528 0.3528 0.3528 0.3528 0.3528];

close all
figure(1)
set(gcf,'Position',[430 100 1200 900])
set(gcf,'DefaultLineLineWidth',2)
set(gcf,'DefaultAxesFontName','Times new roman')
set(gcf,'DefaultAxesFontSize', 25)
set(gcf,'DefaultTextFontName','Times new roman')
plot(x, cl,'Color','b','Marker','o')
hold on
%plot(xboth, cpboth,'Color','g', 'Marker', 'diamond')
grid on
xlim([-15 15])
ylim([-1.5 2])
%ylim([0.3 0.38])
xticks([-15:5:15])
yticks([-1:0.5:2])
title('AoA - $C_l$', 'interpreter', 'LaTeX')
xlabel('$AoA$', 'Interpreter', 'LaTeX')
ylabel('$C_l$', 'Interpreter', 'LaTeX')
grid on
f = gcf;
exportgraphics(f, 'AoA_Cl.jpg','Resolution',300)
exportgraphics(f, 'AoA_Cl.eps','Resolution',300)

figure(2)
set(gcf,'Position',[430 100 1200 900])
set(gcf,'DefaultLineLineWidth',2)
set(gcf,'DefaultAxesFontName','Times new roman')
set(gcf,'DefaultAxesFontSize', 25)
set(gcf,'DefaultTextFontName','Times new roman')
plot(x, cd,'Color','b','Marker','o')
hold on
%plot(xboth, cpboth,'Color','g', 'Marker', 'diamond')
grid on
xlim([-15 15])
ylim([0 0.12])
%ylim([0.3 0.38])
xticks([-15:5:15])
yticks([0:0.02:0.12])
title('AoA - $C_d$', 'interpreter', 'LaTeX')
xlabel('$AoA$', 'Interpreter', 'LaTeX')
ylabel('$C_d$', 'Interpreter', 'LaTeX')
grid on
f = gcf;
exportgraphics(f, 'AoA_Cd.jpg','Resolution',300)
exportgraphics(f, 'AoA_Cd.eps','Resolution',300)
