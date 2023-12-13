clc;
close all;
clear all;

path_origin = 'D:\airfoilFluent\airfoilSimulations\'; % Directory for original exported file from Fluent
path2 = 'D:\airfoilFlowField\'; % Directory for interpolated data
% airfoilpath = 'C:\Users\tjk\OneDrive\문서\머신러닝\AirfoilClCdFiles\';
% airfoilname = char([airfoilpath,'naca0018coordinates.txt']);
% airfoil = load(airfoilname)
% airfoil = importdata(airfoilname);

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
        % original = [];
        % for j=1:104485
        %     if original_raw(j, 1)>=-5 & original_raw(j, 1)<=5
        %             if original_raw(j:2)>=-5 & original_raw(j:2)<=5
        %                 original = cat(1, original, original_raw(j, :));
        %             end
        %     end
        % end
        
        x = original(:, 1);
        y = original(:, 2);
        v = original(:, 5); % data to be sampled
        
        [xq,yq] = meshgrid(-2:0.04:2, -2:0.04:2);
        F = scatteredInterpolant(x, y, v);
        vq = F(xq, yq);       

        fileName = char([path2,name2,'alpha',num2str(alpha_list(i)),'_interpolated.csv']);
        writematrix(vq, char(fileName))
    end
end