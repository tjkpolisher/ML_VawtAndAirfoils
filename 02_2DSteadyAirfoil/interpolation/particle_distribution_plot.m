function particle_distribution_plot

close all;
clear all;

for i=1:19
   
    dir_path = '/Users/sahuckoh/Downloads/seaWaterMATLABCodes/Particle_';    
    fileName = char([dir_path,num2str(i),'.txt']);


    fid = fopen(fileName,'r');
    formatSpec = '%f%f%f%f%f%f%f';
    A = textscan(fid,formatSpec);
    fclose(fid);
    A = horzcat(A{1},A{2},A{3},A{4},A{5},A{6},A{7});
    xx = A(:,1); % 실제 particle의 x좌표
    yy = A(:,2); % 실제 particle의 y좌표
    zz = A(:,3); % 실제 particle의 z좌표

    figure(1)
    plot(zz(:),yy(:),'.');
    axis equal;    hold on
    xlabel('z');   ylabel('y')

    figure(2)
    plot(xx(:),yy(:),'.');
    axis equal;    hold on
    xlabel('x');   ylabel('y')

    figure(3)
    plot(xx(:),zz(:),'.');
    axis equal;    hold on
    xlabel('x');   ylabel('z')


    1;
end
hold off





