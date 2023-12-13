function [nMat,zzg,yyg,xg] = particleEntireNumber2(fileName)

close all;

idx_cube = 0; % set to 1 to make the grid look like cube. Otherwise set to 0 and define nx_grid value.
idx_plot = 0; % set to 1 when you want to plot contours

if nargin == 0
    dir_path = '/Users/sahuckoh/Downloads/seaWaterMATLABCodes/';
    data = 'Particle_10.txt';
    fileName = char([dir_path,data]);
end

fid = fopen(fileName,'r');
formatSpec = '%f%f%f%f%f%f%f';
A = textscan(fid,formatSpec);
A = horzcat(A{1},A{2},A{3},A{4},A{5},A{6},A{7});
xx = A(:,1); 
yy = A(:,2); 
zz = A(:,3); 
fclose(fid);

% xmax = max(xx);  xmin = min(xx);
% ymax = max(yy);  ymin = min(yy);
% zmax = max(zz);  zmin = min(zz);

% The following min and max values are defined from the plots of the
% "particle_distribution_plot.m" file

xmin = 0.0; xmax = 0.04;
ymin = 0; ymax = 0.014;
zmin = 0.01; zmax = 0.03;


lx = xmax - xmin;
ly = ymax - ymin;
lz = zmax - zmin;


xNor = (xx-xmin)/lx; 
yNor = (yy-ymin)/lx; 
zNor = (zz-zmin)/lx; 


ny_grid = 51;
nz_grid = floor(ny_grid*lz/ly);
if idx_cube == 1
    nx_grid = floor(ny_grid*lx/ly);
else
    nx_grid = 51; 
end



xg = linspace(0,1,nx_grid);
yg = linspace(0,max(yNor),ny_grid);
zg = linspace(0,max(zNor),nz_grid);

dx = xg(2)-xg(1);
dy = yg(2)-yg(1);
dz = zg(2)-zg(1);

nMat = zeros(nz_grid-1,ny_grid-1,nx_grid-1);


[yyg,zzg] = meshgrid(yg,zg);



for i=1:nx_grid 
    ix = xNor >= (i-3/2)*dx & xNor < (i-1/2)*dx;         
    yLmt = yNor(ix); zLmt = zNor(ix);
 
    for j=1:ny_grid 

        iy = yLmt >= (j-3/2)*dy & yLmt < (j-1/2)*dy; 
        zLmt2 = zLmt(iy);

        for k=1:nz_grid 
            iz = zLmt2 >= (k-3/2)*dz & zLmt2 < (k-1/2)*dz;
            n = sum(iz);
            nMat(k,j,i) = n;    
 
        end
    end
end


if idx_plot == 1
    figure(1)
    for i=1:nx_grid
        contourf(zzg,yyg,nMat(:,:,i),30,'edgecolor','none');
        title(['xNor = ',num2str(xg(i))])
        xlabel('zNor')
        ylabel('yNor')    
        colorbar
        pause(0.02)
        
        1;
    end
end
1;