clc;
close all;
clear all;

k=133; % number of files
path = 'C:\Users\cfdML\Documents\AirfoilClCdCoordinates\airfoil';
airfoilpath = 'C:\Users\tjk\OneDrive\문서\머신러닝\AirfoilClCdFiles\';
airfoilname = char([airfoilpath,'naca0018coordinates.txt']);
%airfoil = load(airfoilname)
airfoil = importdata(airfoilname);
xi = airfoil(:,3); xi = xi/1000 -0.25; % airfoil x-coordinates
yi = airfoil(:,4); yi = yi/1000; % airfoil y-coordinates

alpha_list = linspace(-10,20,16); % angles of attack except 0 degree

for i=1:133 % Importing coordinates of the airfoils
    coord_path = 'D:\AirfoilClCdCoordinates_out\AirfoilClCdCoordinates_out\airfoil';
    coord_name = char([coord_path,num2str(i),'\airfoil',num2str(i),'coordinates.txt']);
    
    fid = fopen(coord_name,'r');
    formatSpec = '%d%d%f%f%f';
    A = textscan(fid, formatSpec, 'HeaderLines', 1);
    A = horzcat(A{3}, A{4}, A{5});
    A = rmmissing(A,1);
    xx = A(:,1); yy = A(:,2); % x- and y-coordinates, respectively
    
    if max(xx) < 900
        xx = xx-0.25;
    else
        xx = xx/1000-0.25; yy = yy/1000; % Coordinate scale adjustment (mm -> m)
    end

    A = horzcat(xx, yy);
    fclose(fid);
end

nodes = char([airfoilpath,'nodeNumberAndVelocityMagnitude.txt']);
faces = char([airfoilpath,'faces.txt']);
nodes = importdata(nodes); % all nodes of each cases
faces = importdata(faces) + 1; % all faces of each cells

ix = isnan(faces(:,4 )); % face의 개수가 4개인지(즉, 사각형 cell인지)를 판단
faces1 = faces(~ix,:); % 삼각형 cell의 faces
faces2 = faces(ix,1:3); % 사각형 cell의 faces

pos = nodes(:,2:3); % node의 x좌표와 y좌표
v = nodes(:,5); % 해당 노드에서의 velocity magnitude

figure(1)
tplot(pos,faces1,v) % 삼각형 cell plot
hold on
tplot(pos,faces2,v) % 사각형 cell plot
hold off
axis([-1 1 -1 1])

for i=1:1
outFilename = char([path,'kOmegaSSTvelocityMagnitude(alpha',num2str(i),').csv']);
%original_raw = readmatrix(char(outFilename));
original = readmatrix(char(outFilename));

x = original(:, 1);
y = original(:, 2);
v = original(:, 4); % data to be sampled

[xq,yq] = meshgrid(-2:0.02:2, -2:0.02:2); % interpolation할 점

%%%%%

in = inpolygon(xq,yq,xi,yi ); % interpolation point가 에어포일 외부인지 내부인지를 판단
xGrid_in = xq(in); % 에어포일 내부에 있는 점의 x좌표
yGrid_in = yq(in); % 에어포일 내부에 있는 점의 y좌표
xGrid_out = xq(~in); % 에어포일 내부에 있는 점의 x좌표
yGrid_out = yq(~in); % 에어포일 내부에 있는 점의 y좌표

figure(2)
plot(xi,yi,'k',xGrid_in,yGrid_in,'o') % 에어포일 표면의 점과 에어포일 내부에 있는 점 plot

%%%%%
xE = [x;xGrid_in];
yE = [y;yGrid_in];
vE = [v;0*xGrid_in];

F = scatteredInterpolant(xE, yE, vE);
vq = F(xq, yq);

%k = dsearchn([xGrid_out yGrid_out],[xi yi]);
x_in = unique(xGrid_in); y_in = unique(yGrid_in);
% y_in_d = -y_in;
% xp_u=ones(length(x_in), 1); yp_u=ones(length(x_in), 1);
% xp_d=ones(length(x_in), 1); yp_d=ones(length(x_in), 1);
% ii=1; jj=1;
% while ii<length(x_in)
%     if y_in(ii)<y_in(ii+1)
%         xp_u(ii)=x_in(ii); yp_u(ii) = y_in(ii+1);
%     elseif y_in(ii)>y_in(ii+1)
%         xp_u(ii)=x_in(ii); yp_u(ii) = y_in(ii+1);
%     else
%         xp_u(ii)=x_in(ii); yp_u(ii) = y_in(ii);
%     end
%     ii = ii+1;
% end
% 
% while jj<length(x_in)
%     if y_in(jj)<y_in(jj+1)
%         xp_d(jj)=x_in(jj); yp_d(jj) = y_in(jj+1);
%     elseif y_in(jj)>y_in(jj+1)
%         xp_d(jj)=x_in(jj); yp_d(jj) = y_in(jj+1);
%     else
%         xp_d(jj)=x_in(jj); yp_d(jj) = y_in(jj);
%     end
%     jj = jj+1;
% end

figure(3)
set(gcf,'DefaultLineLineWidth', 2)
set(gcf,'DefaultAxesFontName', 'Times new roman')
set(gcf,'DefaultAxesFontSize', 25)
set(gcf,'DefaultTextFontName', 'Times new roman')
%set(gcf,'Position',[400 100 1200 900])


contourf(xq, yq, vq, 50)

set(gca,'XTick',-2:0.5:2)
set(gca,'YTick',-2:0.5:2)
hold on

for j=1:length(xq)
    plot(xq(:,j),yq(:,j) ,  'r','linewidth',0.5)
    plot(xq(j,:),yq(j,:) ,  'r','linewidth',0.5)
    1;
end
plot(airfoil(:,3)/1000-0.5,airfoil(:,4)/1000,'b')
1;
for jj=1:length(k)
    plot(xGrid_out(k(jj)), yGrid_out(k(jj)),'o', 'Markersize', 10, 'Color','b')
    1;
end
for kk = 1:length(x_in)
    plot(xp_u(k(kk)), yp_u(k(kk)),'o', 'Markersize', 10, 'Color','b')
    plot(xp_d(k(kk)), yp_d(k(kk)),'o', 'Markersize', 10, 'Color','b')
    1;
end
hold off
axis([-1 1 -1 1])
1;

%fileName = char([path,'kOmegaSSTvelocityMagnitude(alpha',num2str(i),')_interpolated.csv']);
%writematrix(vq, char(fileName))
end