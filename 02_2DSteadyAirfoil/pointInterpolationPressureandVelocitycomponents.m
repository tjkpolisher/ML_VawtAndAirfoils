clc;
close all;
clear all;

path_origin = 'D:\airfoilFluent\airfoilSimulations\'; % Directory for original exported file from Fluent
path2 = 'D:\airfoilFlowField\'; % Directory for interpolated data

dir_path = 'D:\AirfoilClCdCoordinates_out\AirfoilClCdCoordinates_out\';
data = 'AirfoilIndexList.txt';
fileName = char([dir_path,data]);

fid = fopen(fileName,'r');
formatSpec = '%s%s';
A = textscan(fid,formatSpec);
A = horzcat(A{1},A{2});
nameList1 = A(:,1); 
nameList2 = A(:,2); 
fclose(fid);

alpha_list = linspace(-10,20,16); % angles of attack except 0 degree

for j=1:length(nameList1)
    name1 = char(nameList2(j));
    name2 = char(nameList1(j));
    path = char([path_origin,char(name1),'\',char(name1),'_files\user_files']);
    if length(dir(path)) < 16
        disp('%s (%s) is skipped.', name2, name1)
        continue
    end
    
    for i=1:16
        
        outFilename = char([path,'\',name1,'alpha',num2str(alpha_list(i)),'.csv']);
        %original_raw = readmatrix(char(outFilename));
        original = readmatrix(char(outFilename));
        original = rmmissing(original);
        
        x = original(:, 1);
        y = original(:, 2);

        % data to be sampled
        p = original(:, 4); % pressure
        vu = original(:, 6); % x-velocity component
        vv = original(:, 7); % y-velocity component
        
        [xq,yq] = meshgrid(linspace(-1,1,100), linspace(-1,1,100));
        F1 = scatteredInterpolant(x, y, p);
        F2 = scatteredInterpolant(x, y, vu);
        F3 = scatteredInterpolant(x, y, vv);
        pq = F1(xq, yq);
        vuq = F2(xq, yq);
        vvq = F3(xq, yq);

        fileName1 = char([path2,'pressureField\',name2,'alpha',num2str(alpha_list(i)),'_pressureInterpolated.csv']);
        writematrix(pq, char(fileName1))
        fileName2 = char([path2,'velocityUField\',name2,'alpha',num2str(alpha_list(i)),'_velocityUInterpolated.csv']);
        writematrix(vuq, char(fileName2))
        fileName3 = char([path2,'velocityVField\',name2,'alpha',num2str(alpha_list(i)),'_velocityVInterpolated.csv']);
        writematrix(vvq, char(fileName3))
    end
end