function airfoil_coord_interpolation

close all;
pathDir = '/Users/sahuckoh/Downloads/AirfoilClCdCoordinates';
epsilon = 1e-4;
n = 133;
ng = 50;

allXgu = zeros(ng,n);
allYgu = zeros(ng,n);
allXgl = zeros(ng,n);
allYgl = zeros(ng,n);

for i = 1:n 
    outputPath = char([pathDir,'/airfoil',num2str(i)]);
    filename = char([outputPath,'/airfoil',num2str(i),'coordinates.txt']);
        
    res = importdata(filename);

    pos = res.data;
    pos = pos(:,3:4);

    if abs(max(pos(:,1)) - 1)>epsilon || abs(min(pos(:,1)))>epsilon
      a = max(pos(:,1)) - min(pos(:,1)); 
      pos(:,1) = (pos(:,1))/a;
      pos(:,2) = pos(:,2)/a;
    end



    ixnan = isnan(pos(:,1)) | isnan(pos(:,2));
    pos  = pos(~ixnan,:);
    [~, ix] = min(pos(:,1));

    d = norm(pos(1,:) - pos(end,:));    
    if d>epsilon

        posE = [pos;pos(1,:)];
    else

        posE = pos;
    end

    
    xu = posE(1:ix,1); xu = flipud(xu);
    yu = posE(1:ix,2); yu = flipud(yu);
    
    xl = posE(ix:end,1);
    yl = posE(ix:end,2);

    plot(xu,yu,'k-',xl,yl,'k-'); 
    tau = linspace(0,1,ng+2)'; tau = tau(2:end-1);

    
    [xgu, ygu] = generate_data(xu,yu,tau);
    [xgl, ygl] = generate_data(xl,yl,tau);
    
    hold on
    plot(xgu,ygu,'bo-',xgl,ygl,'rd-'); 
    hold off
    axis equal;
    
    allXgu(:,i) = xgu; allYgu(:,i) = ygu;
    allXgl(:,i) = xgl; allYgl(:,i) = ygl;
    1;
end


mean_xgu = mean(allXgu,2);
mean_xgl = mean(allXgl,2);


xgu = mean_xgu;
xgl = mean_xgl;

xg = (xgu+xgl)/2;




for i = 1:n 
    outputPath = char([pathDir,'/airfoil',num2str(i)]);
    filename = char([outputPath,'/airfoil',num2str(i),'coordinates.txt']);
    res = importdata(filename);

    pos = res.data;
    pos = pos(:,3:4);


    if abs(max(pos(:,1)) - 1)>epsilon || abs(min(pos(:,1)))>epsilon
      a = max(pos(:,1)) - min(pos(:,1)); 
      pos(:,1) = (pos(:,1))/a;
      pos(:,2) = pos(:,2)/a;
    end


    ixnan = isnan(pos(:,1)) | isnan(pos(:,2));
    pos  = pos(~ixnan,:);
    [~, ix] = min(pos(:,1));

    d = abs(pos(1,1) - pos(end,1));    
    if d>epsilon

        posE = [pos;pos(1,:)];
    else

        posE = pos;
    end

    
    xu = posE(1:ix,1); xu = flipud(xu);
    yu = posE(1:ix,2); yu = flipud(yu);
    
    xl = posE(ix:end,1);
    yl = posE(ix:end,2);

  
    plot(xu,yu,'k-',xl,yl,'k-'); 

    ygu = interp1(xu,yu,xg);
    ygl = interp1(xl,yl,xg);

    
    hold on
    plot(xg,ygu,'bo-',xg,ygl,'rd-'); 
    hold off
    axis equal;

    out = [ygu(:) ygl(:)]';
    filename_out = char([outputPath,'/airfoilOut',num2str(i),'.txt']);
    writematrix(out,filename_out);
    1;
end


1;





function [xg,yg] = generate_data(xu,yu,tau)

pos = [xu yu];
n = size(pos,1);

allL = zeros(n,1);

for i = 1: n-1
    allL(i+1) = allL(i) + norm(pos(i+1,:) - pos(i,:));
end
allL_nor = allL/allL(end);


xg = interp1(allL_nor,xu,tau);
yg = interp1(allL_nor,yu,tau);
1;














