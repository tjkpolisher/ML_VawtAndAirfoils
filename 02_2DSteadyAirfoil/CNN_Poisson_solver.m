function [xx,yy,uu] = CNN_Poisson_solver(x_airfoil,y_airfoil)

% alpha : degree
%% Data preparation
close all;
plotidx = 'off';
hmax = 0.1;
nref = 1;
N = 100;
xg = linspace(-2,2,N)';
yg = linspace(-2,2,N)';
count = 0;
%% Determine the governing equation
% If the variable 'GoverningEquation' is set 1, the governing equation of
% the field will set to Laplace equation.
% Otherwise, the field will be built with respect to Poisson equation, with
% the constant value of (-alpha/10).
GoverningEquationMode = 1;
if GoverningEquationMode==1
    GoverningEquation = char('laplace'); 
elseif GoverningEquationMode==2
    GoverningEquation = char('poisson');
end
BC_case = 2; % determine the type of boundary condition of the domain (details are in the 'for' loop below)

dir_path = 'D:\AirfoilClCdCoordinates_out\AirfoilClCdCoordinates_out\';
data = 'AirfoilIndexList.txt';
coord_path = 'D:\AirfoilClCdCoordinates_out\AirfoilClCdCoordinates_out\airfoil';
path_save = char(['D:\CNN_poisson_files\', GoverningEquation, '\']);

fileName = char([dir_path,data]);

fid = fopen(fileName,'r');
formatSpec = '%s%s';
A = textscan(fid,formatSpec);
A = horzcat(A{1},A{2});
nameList1 = A(:,1); % Airfoil indexes
nameList2 = A(:,2); % Real airfoil name
fclose(fid);

alpha_list = linspace(-10,20,16); % angles of attack except 0 degree

%% Determine the governing equation
% If fval is 0, the govering equation is Laplace eq.
% On the other hand, if fval is 1, the equation will be Poisson eq.

xout = [-2 2 2 -2]'; % 경계점의 x좌표
yout = [-2 -2 2 2]'; % 경계점의 y좌표
pv_bdr = [xout yout]; % grid 경계점
pv_bdr = [pv_bdr;pv_bdr(1,:)];

%% Field generation
for i=1:length(nameList1) % Importing coordinates of the airfoils
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

    A = [xx';yy'];
    fclose(fid);

    for j=1:length(alpha_list) % Rotating coordinates to each angles of attack
        alpha = alpha_list(j);
        if GoverningEquationMode==1
            fval = 0; %laplace
        elseif GoverningEquationMode==2
            fval = -alpha/10; % poisson (CAUTION: alpha is degree)
        end
        
        
        transf = [cosd(-alpha) -sind(-alpha); sind(-alpha) cosd(-alpha)];
        coordT = transf*A;
        coordT = coordT';
        x_airfoil = coordT(:,1); y_airfoil = coordT(:,2);
        
        %%% 

        pv_in1 = [x_airfoil y_airfoil];
        pv_in1 = [pv_in1;pv_in1(1,:)];
        
        %%%
        
        
        [p,t,e,n1,n2]=pmesh_inside_1_fast(pv_bdr,pv_in1,hmax,nref,plotidx);
        
        e_domain =  e(1:n1);
        e_airfoil = e(n1+1:n2);
        
        %%%%% boundary conditions
        if BC_case==1
            ix = abs(p(e_domain,1) + 2)<1e-6; e_inlet = e_domain(ix); %B.C. case1 (오른쪽 면만 상수)
        elseif BC_case==2
            ix = abs(p(e_domain,1) + 2)<1e-6 | abs(p(e_domain,2) + 2)<1e-6; e_inlet = e_domain(ix); %B.C. case2(오른쪽 면, 아랫면 상수)
        elseif BC_case==3
            ix = abs(p(e_domain,1) - 2)<1e-6; e_inlet = e_domain(~ix); %B.C. case3 (오른쪽 면, 윗면, 아랫면 상수)
        else
            warning('The argument "BC_case" must be an integer of 1, 2, or 3.')
        end
        
        
        [Amat,b] = matrix_poisson(p,t,fval);
        
        
        Amat(e_airfoil,:) = 0; b(e_airfoil) = 0;
        Amat(e_inlet,:) = 0; b(e_inlet) = 1;
        
        Amat(e_airfoil,e_airfoil)=speye(length(e_airfoil),length(e_airfoil));
        Amat(e_inlet,e_inlet)=speye(length(e_inlet),length(e_inlet));
        
        % Solve
        u = Amat\b;
        
        %tplot(p,t,u)
        %%%%%%%%%%%%
        
        [p2,t2] = pmesh_fast(pv_in1,hmax/10,nref);
        u2 = 0*p2(:,1);
        
        pp = [p;p2];
        uu = [u;u2];
        
        [pp,ix] = unique(pp,'rows');
        uu = uu(ix);
        
        [yy,xx] = meshgrid(yg,xg);
        F = scatteredInterpolant(pp(:,1),pp(:,2),uu);
        uu = F(xx,yy);
        
%         uumax=2.902; uumin=-5.26392124908248;
%         uuNor = (uu-uumin)/(uumax-uumin);
%         disp(min(uuNor))
%         disp(max(uuNor))
% 
%         figure(2)
%         contourf(xx,yy,uuNor,100,'edgecolor','none');
%         hold on
%         c=colorbar;
%         c.Limits = [-5 5];
%         plot(x_airfoil, y_airfoil,'Marker','o','MarkerSize',5,'MarkerFaceColor','k', 'MarkerEdgeColor','k','Color','k')
%         imageTitle = char(['Airfoil ',num2str(i), '(',char(nameList2(j)),')',', $\alpha$=',num2str(alpha)]);
%         title(imageTitle, 'Interpreter','latex')
%         hold off
        
       

        fileName = char([path_save, 'bc',num2str(BC_case), '\', char(nameList1(i)),'_alpha',num2str(alpha_list(j)),'.csv']);
        writematrix(uu, char(fileName))


%         [i, j, sum(sum(isnan(uu))) ]    
% 
% 
%         if sum(sum(isnan(uu)))  ~= 0
%             disp(char([nameList1(i),'(',nameList2(i),')',num2str(alpha_list(j)),', ', num2str(count)]))
%             count = count + 1;
%         end
        %clear uu;
    end
    1;
    %clear A;
end




%close all





function [A,b] = matrix_poisson(p,t,fval)

% Assemble K and F
N=size(p,1);
A=sparse(N,N);
b=zeros(N,1);
for ielem=1:size(t,1)
  el=t(ielem,:);
  
  Q=[ones(3,1),p(el,:)];
  Area=abs(det(Q))/2;
  c=inv(Q);
    
  Ah=Area*(c(2,:)'*c(2,:)+c(3,:)'*c(3,:));
  fh=Area/3*fval;
  
  A(el,el)=A(el,el)+Ah;
  b(el)=b(el)+fh;
end

