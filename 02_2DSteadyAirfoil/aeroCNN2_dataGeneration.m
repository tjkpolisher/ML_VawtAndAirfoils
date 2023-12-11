function aeroCNN2_dataGeneration

close all;

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
        xx = xx-0.5;
    else
        xx = xx/1000-0.5; yy = yy/1000; % Coordinate scale adjustment (mm -> m)
    end

    %A = horzcat(xx, yy);
    fclose(fid);

    for j=1:length(alpha_list) % Rotating coordinates to each angles of attack
        alpha = alpha_list(j);
        
        res = airfoilData_aeroCNN2(xi,yi);
        exportname = char(['D:\20221103aeroCNNII(-1to1)\offset0.5\airfoil',num2str(i),'_alpha', num2str(alpha), '_.csv']);
        writematrix(res, exportname)
    end
    
    clear A;
end