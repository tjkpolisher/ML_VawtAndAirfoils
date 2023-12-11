function [res1,res2] = aeroCNN2mask(xx, yy, alpha, beta, h)

close all;
%% Selecting mode
% If mode is 1, the mask of the entire domain is created.
% If mode is 2, the mask of the magnified domain is created.
% Details for the magnified region is described below.
%mode = 2;

set(0,'DefaultLineLineWidth',2)

% Change default axes fonts.
set(0,'DefaultAxesFontName', 'Times New Roman')
set(0,'DefaultAxesFontSize', 20)

% Change default text fonts.
set(0,'DefaultTextFontname', 'Times New Roman')
set(0,'DefaultTextFontSize', 20)
%% Setting parameters for generating meshgrid

N = 100; % Number of grids
maxVal = 100; % A grid completly inside the airfoil has value of 100, completly outside the airfoil has 0.
xmin = -1; xmax = 1;
ymin = -1; ymax = 1;
xOffset = 0.03; yOffset = -0.17;
xminMag = 0.5-xOffset; xmaxMag = 0.5+xOffset;
yminMag = yOffset; ymaxMag = 0.012;

x = linspace(xmin,xmax,N+1)'; dx = x(2) - x(1);
y = linspace(ymin,ymax,N+1)'; dy = y(2) - y(1);
xM = linspace(xminMag,xmaxMag,N+1)';
yM = linspace(yminMag,ymaxMag,N+1)';
dxM = xM(2) - xM(1); dyM = yM(2) -yM(1);
area_grid = dx*dy; area_gridMag = dxM*dyM;


% The 'if' state here is for independent execution of aeroCNN2mask.m file.
if nargin == 0
    coord_path = 'D:\AirfoilClCdCoordinates_out\AirfoilClCdCoordinates_out\airfoil';
    coord_name = char([coord_path,num2str(15),'\airfoil',num2str(15),'coordinates.txt']);
    fid = fopen(coord_name,'r');
    formatSpec = '%d%d%f%f%f';
    A = textscan(fid, formatSpec, 'HeaderLines', 1);
    A = horzcat(A{3}, A{4}, A{5});
    A = rmmissing(A,1);
    xx = A(:,1); yy = A(:,2);
    xx = xx/1000-0.5; yy = yy/1000;
    fclose(fid);

    alpha = 16;
    beta = 0;
    h = 0;
end
% Meshgrids of the entire domain and the magnified region, respectively.

[ym,xm] = meshgrid(x,y); [xMm, yMm] = meshgrid(xM, yM);
% Results
res1 = zeros(N,N); res2 = zeros(N,N);
%% Creating the coordinates of Gurney flaps
% yLeft = linspace(-h/5, -h, 5);
% yRight = linspace(-h/5, -h, 5);
% xLeft = 0.5*ones(1,5) - 0.02*h;
% xRight = 0.5*ones(1,5);
% 
% betaValue = 90-beta;
%% Rotating transformation - Phase 1.Gurney flap
% For AeroCNN-II based model, we have to use not only y-coordinates
% but also x-coordinates of the Gurney flap
% rotateBeta = [cosd(betaValue) -sind(betaValue); sind(betaValue) cosd(betaValue)];
% LeftImp = [xLeft-0.5; yLeft]; % Coordinates of the left side of Gurney flap
% RightImp = [xRight-0.5; yRight]; % Coordinates of the right side of Gurney flap
% 
% rotatedFlapLeft = rotateBeta * LeftImp;
% rotatedFlapRight = rotateBeta * RightImp;
%                                                                                                                                                                                                                                                 
% rotatedFlapLeftx = rotatedFlapLeft(1,:); rotatedFlapRightx = rotatedFlapRight(1,:);
% rotatedFlapLefty = rotatedFlapLeft(2,:); rotatedFlapRighty = rotatedFlapRight(2,:);
% 
% rotatedFlapLeftx = rotatedFlapLeftx'+0.5; rotatedFlapRightx = rotatedFlapRightx'+0.5;
% rotatedFlapLefty = rotatedFlapLefty'; rotatedFlapRighty = rotatedFlapRighty';
% 
% % Final coordinates (Airfoil + Gurney flap)
% xi = [xx;rotatedFlapLeftx;flipud(rotatedFlapRightx);max(xx)];
% yi = [yy;rotatedFlapLefty;flipud(rotatedFlapRighty);0];

xi=[xx;max(xx)];
yi=[yy;0];
%% Rotating transformation - Phase 2. Entire airfoil

rotateAlpha = [cosd(-alpha) -sind(-alpha); sind(-alpha) cosd(-alpha)];
rotated = rotateAlpha * [xi yi]';

xi = rotated(1,:); yi = rotated(2,:);
xi = xi'; yi = yi';

p = [xi yi]; 
poly_airfoil = polyshape(p, 'SolidBoundaryOrientation','ccw','Simplify',false);

% Tips and tricks: You may change the 'SolidBoundaryOrientation' keyword 
% depending on the way your geometry is aligned.
% If the keyword 'Simplify' is not designated as 'false',
% the 'polyshape' method might automatically deletes some of the points of Gurney
% flaps or airfoil itself. 


for i=1:N
    for j=1:N
        %if mode==1
            x_grid = [xm(i,j) xm(i+1,j) xm(i+1,j+1) xm(i,j+1)];        
            y_grid = [ym(i,j) ym(i+1,j) ym(i+1,j+1) ym(i,j+1)];
        
            poly_grid = polyshape(x_grid,y_grid);
    
            poly_intersect = intersect(poly_airfoil,poly_grid);
            A = area(poly_intersect);
            
            res1(i,j) =  maxVal*A/area_grid;
            1;
        
        %elseif mode==2
            x_gridMag = [xMm(i,j) xMm(i+1,j) xMm(i+1,j+1) xMm(i,j+1)];        
            y_gridMag = [yMm(i,j) yMm(i+1,j) yMm(i+1,j+1) yMm(i,j+1)];
        
            poly_gridMag = polyshape(x_gridMag,y_gridMag);
   
            poly_intersectM = intersect(poly_airfoil,poly_gridMag);
            AM = area(poly_intersectM);
            
            res2(i,j) =  maxVal*AM/area_gridMag;
            1;
        %end
    end
end
%% Checking by ploting images
%if mode==1
    figure(1)
    image(res1'/maxVal*255,"XData",[mean(x(1:2)) mean(x(end-1:end))],...
        "YData",[mean(y(1:2)) mean(y(end-1:end))]);
    % image(res',"XData",[mean(x(1:2)) mean(x(end-1:end))],...
    %     "YData",[mean(y(1:2)) mean(y(end-1:end))]);
    colorbar
    hold on
    plot(xm,ym,'g')
    plot(xm',ym','g')
    plot(xi,yi,'r')
    hold off
    axis equal
    1;

%elseif mode==2
    figure(2)
    image(res2/maxVal*255,"XData",[mean(xM(1:2)) mean(xM(end-1:end))],...
        "YData",[mean(yM(1:2)) mean(yM(end-1:end))]);
    % image(res',"XData",[mean(x(1:2)) mean(x(end-1:end))],...
    %     "YData",[mean(y(1:2)) mean(y(end-1:end))]);
    colorbar
    hold on
    plot(xMm,yMm,'g')
    plot(xMm',yMm','g')
    plot(xi,yi,'r')
    hold off
    axis ([xminMag xmaxMag yminMag ymaxMag])
    1;
%end
exportname1 = char(['D:\VAWT_data\flap_steady\flap_steady\aeroCNN2Mask\h_',num2str(h),'_beta',...
num2str(beta),'_alpha',num2str(alpha),'.csv']);
writematrix(res1, exportname1)
exportname2 = char(['D:\VAWT_data\flap_steady\flap_steady\aeroCNN2Mask\h_',num2str(h),'_beta',...
num2str(beta),'_alpha',num2str(alpha),'_magnified.csv']);
writematrix(res2, exportname2)