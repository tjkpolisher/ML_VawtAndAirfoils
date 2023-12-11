function yplot1

clear all;
close all;
tn = 22;
ycuts = 0.075;

%ycut = 0.06;
%z=;

xmin = 0.03; xmax = 0.12;
zmin = 0.035; zmax = 0.065;
Nz = 20; % 20 or 30 (1st injector only) / 52 (2nd injector included, calculated based on 30 when 1st injector only)
save_path = 'D:\newparticleStorage\numberOfParticles\20221210\nz20';

dy = 0.002;
S = 0;
for ii=1:length(ycuts)
    ycut = ycuts(ii);

    for i=1:tn
        %dir_path = 'D:\newparticleStorage\particle_';
        fileName = char([dir_path,num2str(i),'.txt']);
    
        fid = fopen(fileName,'r');
        formatSpec = '%f%f%f%f%f%f%f';
        A = textscan(fid,formatSpec);
        A = horzcat(A{1},A{2},A{3},A{4},A{5},A{6},A{7});
        xx = A(:,1); 
        yy = A(:,2); 
        zz = A(:,3); 
        fclose(fid);
       
        iy = yy<ycut+dy/2 & yy>ycut-dy/2 & xx<0.14 & zz>0.01;
        %iy = yy<ycut+dy/2 & yy>ycut-dy/2 & xx<0.2;
        %plot3(xx,yy,zz,'k.',xx(ix),yy(ix),zz(ix),'r.',xx(iy),yy(iy),zz(iy),'g.'); axis equal
        figure(1) % Particles on xz plane(y-coordinate is fixed at ycut) at the time step i.
        plot(xx(iy),zz(iy),'.')
        axis equal
        axis([xmin xmax zmin zmax]);
        title(['time = ', num2str(i)])
        
        [xxg,zzg,nMat] = grid_interpolation(xx(iy),zz(iy),xmin,xmax,zmin,zmax,Nz);
    
        figure(2) % Contourf version of figure 1.
        contourf(xxg,zzg,nMat,30,'edgecolor','none'); axis equal
        colorbar
        %caxis([0 25])
        
        outputname = sprintf('numberOfParticles(transient)_y_%0.3f_time_%d_Nz_%d.csv', ycut,i,Nz);
        exportname = char([save_path,'\',outputname]);
        writematrix(nMat, exportname)
        
        S = S+nMat;
    
        figure(3)
        plot3(xx,yy,zz,'.',xx(iy),yy(iy),zz(iy),'g.'); axis equal; xlabel('x'); ylabel('y'); zlabel('z')
    
        1;
    end
    figure(4)
    contourf(xxg,zzg,S/tn,30,'edgecolor','none'); axis equal
    title('y=',ycut)
    colorbar
    
    outputname = sprintf('numberOfParticles_TimeAveraged_y_%0.3f_Nz_%d.csv', ycut,Nz);
    exportname = char([save_path,'\',outputname]);
    writematrix(nMat, exportname)
    1;

end

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





