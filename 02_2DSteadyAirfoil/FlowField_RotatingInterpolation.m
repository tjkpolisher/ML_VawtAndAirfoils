function FlowField_RotatingInterpolation()

% if nargin==0
% 
% end

path_origin = 'D:\airfoilFluent\airfoilSimulations\'; % Directory for original exported file from Fluent
path2 = 'D:\rotatedInterpolation_pow2\n_grid128(-1to1)offset0.25\'; % Directory for interpolated data

dir_path = 'D:\AirfoilClCdCoordinates_out\AirfoilClCdCoordinates_out\';
data = 'AirfoilIndexList.txt';
fileName = char([dir_path,data]);

fid = fopen(fileName,'r');
formatSpec = '%s%s';
A = textscan(fid,formatSpec);
A = horzcat(A{1},A{2});
nameList1 = A(:,1); % Airfoil indexes
nameList2 = A(:,2); % Real airfoil name
fclose(fid);

n_grid = 128;
alpha_list = linspace(-10,20,16); % angles of attack except 0 degree

for j=1:length(nameList1)    
    name1 = char(nameList2(j)); % Real airfoil name
    name2 = char(nameList1(j)); % Airfoil indexed
    path = char([path_origin,char(name1),'\',char(name1),'_files\user_files']);
    if length(dir(path)) < 16
        disp('%s (%s) is skipped.', name2, name1)
        continue
    end
    
    for i=1:length(alpha_list)
        
        outFilename = char([path,'\',name1,'alpha',num2str(alpha_list(i)),'.csv']);
        original = readmatrix(char(outFilename));
        original = rmmissing(original);
        
        alpha = alpha_list(i);
        xx = original(:, 1);
        yy = original(:, 2);
        if max(xx) < 900
            %xx = xx-0.25;
            xx = xx;
        else
            %xx = xx/1000-0.25; yy = yy/1000; % Coordinate scale adjustment (mm -> m)
            xx = xx/1000; yy = yy/1000;
        end
        AA = [xx'; yy']; % Coordinates of the original data nodes
        transf = [cosd(-alpha) -sind(-alpha); sind(-alpha) cosd(-alpha)];
        coordT = transf*AA;
        coordT = coordT';
        xi = coordT(:,1); yi = coordT(:,2); % 회전된 노드 좌표

        % data to be sampled
        p = original(:, 4); % pressure
        v = original(:, 5); % velocity magnitude
        vu = original(:, 6); % x-velocity component
        vv = original(:, 7); % y-velocity component
        vc = transf*[vu'; vv']; % 속도 성분 회전

        vc = vc';
        vu = vc(:,1);
        vv = vc(:,2);

        xmrange = linspace(-1, 1, n_grid);
        ymrange = linspace(-1, 1, n_grid);

        [xq,yq] = meshgrid(xmrange, ymrange);
        F1 = scatteredInterpolant(xi, yi, p);
        F2 = scatteredInterpolant(xi, yi, v);
        F3 = scatteredInterpolant(xi, yi, vu);
        F4 = scatteredInterpolant(xi, yi, vv);
        pq = F1(xq, yq);
        vq = F2(xq, yq);
        vuq = F3(xq, yq);
        vvq = F4(xq, yq);

        if alpha_list(i)==-10||alpha_list(i)==0||alpha_list(i)==10||alpha_list(i)==20
            figure(i)
            set(gcf,'Position',[430 100 1200 900])
            set(gcf,'DefaultLineLineWidth',2)
            set(gcf,'DefaultAxesFontName','Times new roman')
            set(gcf,'DefaultAxesFontSize', 25)
            set(gcf,'DefaultTextFontName','Times new roman')
            contourf(xq,yq,vq,100,'edgecolor','none'); colorbar
            imageTitle = char(['Airfoil ',num2str(j), '(',char(nameList2(j)),')',', $\alpha$=',num2str(alpha)]);
            imageTitle2 = char(['Airfoil',num2str(j), '(',char(nameList2(j)),')','_alpha',num2str(alpha)]);
            title(imageTitle,'Interpreter', 'LaTeX')

            f = gcf;
            graphicsName = char([path2,'images\',imageTitle2,'.jpg']);
            exportgraphics(f, graphicsName,'Resolution',300)
            pause(0.05)
            close all
        end

        fileName1 = char([path2,'pressureField\',name2,'alpha',num2str(alpha_list(i)),'_pressureInterpolated.csv']);
        writematrix(pq, char(fileName1))
        fileName2 = char([path2,'velocityMagnitudeField\',name2,'alpha',num2str(alpha_list(i)),'_velocityMagnitudeInterpolated.csv']);
        writematrix(vq, char(fileName2))
        fileName3 = char([path2,'velocityUField\',name2,'alpha',num2str(alpha_list(i)),'_velocityUInterpolated.csv']);
        writematrix(vuq, char(fileName3))
        fileName4 = char([path2,'velocityVField\',name2,'alpha',num2str(alpha_list(i)),'_velocityVInterpolated.csv']);
        writematrix(vvq, char(fileName4))
    end
end