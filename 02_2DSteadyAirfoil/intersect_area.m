function test_intersect_area

 
clear all;
clc
close all;

nx_grid = 101;
ny_grid = 101;

airfoilDirectory = 'D:\AirfoilClCdCoordinates_out\AirfoilClCdCoordinates_out\airfoil1';
airfoilCoordinateName = 'airfoilOut1.txt';
fileName = char([airfoilDirectory,'\',airfoilCoordinateName]);

fid = fopen(fileName,'r');
formatSpec = '%f';
A_raw = textscan(fid,formatSpec,100,'Delimiter',',');
A = A_raw{:}; 
fclose(fid);

dat1 = horzcat(A(1:50),A(51:100));
yUp = dat1(:,1);
yDown = dat1(:,2);
x = linspace(0,1,length(yUp));
y = [yUp; flipud(yDown)];
xx = [x'; flipud(x')]; % x-coordinates for plotting airfoil

% x_grid = [0.1 0.2 0.2 0.1];
% y_grid = [0.02 0.02, 0.03 0.03];

xg = linspace(0,1,nx_grid);
yg = linspace(-0.5,0.5,ny_grid);

plot(xx,y,'o-','Color','b')

hold on

%plot(yyg,zzg,'ro')

grid on

artificialImage = zeros(nx_grid-1,ny_grid-1);
dx = 2/nx_grid;
dy = 2/ny_grid;
for i=1:ny_grid-1
    y_grid = [yg(i) yg(i) yg(i+1) yg(i+1)];
    for j=1:nx_grid-1
        x_grid = [xg(j) xg(j+1) xg(j+1) xg(j)];
        poly0 = polyshape(xx,y); % Airfoil shape
        
        poly1 = polyshape(x_grid,y_grid); % Grid point in the entire domain
        poly2 = intersect(poly0, poly1); % Area intersected
        
        rpd = area(poly2)/(dx*dy)*100; % Raw Pixel Density
        fpd = (1-rpd/100)*100; % Final Pixel Density (출처: AIAA Application of Convolutional Neural Network to Predict Airfoil Lift Coefficient)
        artificialImage(j,i) = fpd;
    end
end
xg2 = linspace(0,1,nx_grid-1);
yg2 = linspace(-0.5,0.5,ny_grid-1);
[xxg,yyg] = meshgrid(xg2,yg2);
artificialImage = artificialImage(1:nx_grid-1, 1:ny_grid-1);

figure(2)
contourf(xxg,yyg,artificialImage',100,'edgecolor','none');   
colorbar

outputPath = 'D:\';
filename_out = char([outputPath,'airfoil1.txt']);
writematrix(out,filename_out);
1;

 

 

 

 

% x = 0; y = 0; r = 0.5; 

% th = 0:pi/50:2*pi;

% xunit = r * cos(th) + x;

% yunit = r * sin(th) + y;

% poly0 = polyshape(xunit(1:end-1),yunit(1:end-1));

% poly1 = polyshape([0 0 1 1],[1 0 0 1]);

% poly2 = intersect(poly0,poly1);

% area(poly2)

 

 

1;