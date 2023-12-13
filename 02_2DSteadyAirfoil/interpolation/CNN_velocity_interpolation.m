


function CNN_velocity_interpolation


close all;

% %%%%%%%%%%%%%%%%%%%%% input %%%%%%%%%%%%%%%%%%%%%%%%%%
theta = 0;
data_load_idx = 1;
D = 1.03;
x1 = 2*D; x2 = 5*D;
y1 = -D;  y2 = D;
nx = 100;
ny = 100;

path = 'C:\Users\tjk\Downloads\interpolation/';

for i=3.973816:0.005256:4.163032



formatSpec = '%.6f';
s = num2str(i,formatSpec);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pathMesh = char([path,s,'/constant/polyMesh']);
pathData = char([path,s]);

if data_load_idx == 0
    [~,faces,startFace,CNT] = load_pos_faces_boundary(pathMesh);
elseif data_load_idx == 1
    load('all.mat')  
end



%%%%%%% load position file at each time
scriptName= char([pathData, '/script_point.sh']);
if ~exist(scriptName, 'file')
    copyFile = '/media/com3/hard1/vawt_added/initial/script_point.sh';
    cmd = char(['cp ',copyFile,' ',pathData]); 
    system(cmd);
end
cmd = char(['source ',scriptName,' ',pathData]);
system(cmd);

filePoints = char([pathData,'/polyMesh/points1']);
pos = importdata(filePoints);


%%%%%%% calculate velcotiy components
[xx,yy,uu,vv,magV] = implementationData(pos,faces,startFace,pathData,...
    CNT,x1,x2,y1,y2,nx,ny);        

figure()
contourf(xx/D,yy/D,magV,100,'edgecolor','none'); colorbar
saveas(gcf,s,'epsc')


%%%%%%% write data file %%%%%%%%


outFilename_u = char([path,num2str(theta),'_data_u.txt']);
fileID = fopen(outFilename_u,'w');
fprintf(fileID,'%f %f %f\n',xx,yy,uu);
fclose(fileID);

outFilename_v = char([path,num2str(theta),'_data_v.txt']);
fileID = fopen(outFilename_v,'w');
fprintf(fileID,'%f %f %f\n',xx,yy,vv);
fclose(fileID);


outFilename_magV = char([path,num2str(theta),'_data_magV.txt']);
fileID = fopen(outFilename_magV,'w');
fprintf(fileID,'%f %f %f\n',xx,yy,magV);
fclose(fileID);
theta=theta+10;

1;

end


    



function [xx,yy,uu,vv,magV] = implementationData(pos,faces,startFace,...
    pathData,CNT,x1,x2,y1,y2,nx,ny)

idxTM = 'SA'; % dummy
fileIdx = 1; % for velocity

res = readSolutionFile(fileIdx,pathData,pos,faces,startFace,CNT,idxTM);

pos = res.allPc{1};
ix = pos(:,3)>0;
pos = pos(ix,:);

vel = res.allVal{1};
vel = vel(ix,:);

Fx = scatteredInterpolant(pos(:,1),pos(:,2),vel(:,1));
Fy = scatteredInterpolant(pos(:,1),pos(:,2),vel(:,2));



x = linspace(x1,x2,nx);
y = linspace(y1,y2,ny);


[yy,xx] = meshgrid(y,x);


uu = Fx(xx,yy);
vv = Fy(xx,yy);
magV = sqrt(uu.^2 + vv.^2);



1;







