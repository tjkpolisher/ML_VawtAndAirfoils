function yplot_3D

clear all;
close all;
tn = 22;
%ycut = 0.06;

xmin = 0.03; xmax = 0.14;
zmin = 0.035; zmax = 0.065;
Nz = 30;
dy = 0.002;

allyCut = 0.03:dy:0.12;

dir_path = 'D:\newparticleStorage\particle_';
save_path = 'D:\newparticleStorage\numberOfParticles\20221210\nz30\';

for i=1:tn
     
    xxx = []; yyy = []; zzz = []; nnn = [];
    fileName = char([dir_path,num2str(i),'.txt']);

    fid = fopen(fileName,'r');
    formatSpec = '%f%f%f%f%f%f%f';
    A = textscan(fid,formatSpec);
    A = horzcat(A{1},A{2},A{3},A{4},A{5},A{6},A{7});
    xx = A(:,1); 
    yy = A(:,2); 
    zz = A(:,3); 
    fclose(fid);
   

    for j=1:length(allyCut)

        ycut = allyCut(j);
        iy = yy<ycut+dy/2 & yy>ycut-dy/2 & xx<0.14 & zz>0.01;

        [xxg,zzg,nMat] = grid_interpolation(xx(iy),zz(iy),xmin,xmax,zmin,zmax,Nz);
           
        xxx = [xxx;xxg(:)];
        yyy = [yyy;0*xxg(:) + ycut];
        zzz = [zzz;zzg(:)];
        nnn = [nnn;nMat(:)];

%         figure(2)
%         contourf(xxg,zzg,nMat,30,'edgecolor','none'); axis equal
%         colorbar



    end

    res = [xxx yyy zzz nnn];
    fileName = char([save_path, num2str(['out_',num2str(i),'.txt'])]);
    writematrix(res,fileName ,'Delimiter','tab')
    1;

end



1;



function [xxg,zzg,nMat] = grid_interpolation(x,z,xmin,xmax,zmin,zmax,Nz)


dz = (zmax-zmin)/Nz;
Nx = round((xmax-xmin)/dz);
dx = (xmax-xmin)/Nx;

nMat = zeros(Nx,Nz);

for i=1:Nx
    for j=1:Nz
        ix = x > xmin + (i-3/2)*dx & x < xmin + (i-1/2)*dx & z > zmin + (j-3/2)*dz & z < zmin + (j-1/2)*dz;  
        nMat(i,j) = sum(ix);
    end
end

xg = linspace(xmin,xmax,Nx);
zg = linspace(zmin,zmax,Nz);

[zzg,xxg] = meshgrid(zg,xg);

1;





