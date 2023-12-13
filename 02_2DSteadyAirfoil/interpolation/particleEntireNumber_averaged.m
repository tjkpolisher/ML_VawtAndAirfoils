function particleEntireNumber_averaged

tn = 19;
S = 0;
for i=1:tn
   
    dir_path = '/Users/sahuckoh/Downloads/seaWaterMATLABCodes/Particle_';    
    fileName = char([dir_path,num2str(i),'.txt']);

    [nMat,zzg,yyg,xg]=particleEntireNumber2(fileName);
    S = S + nMat;
    1;
end

S = S/tn;
maxS = max(S(:));
figure(1)
nx_grid = size(S,3);
for i=1:nx_grid
    contourf(zzg,yyg,nMat(:,:,i),30,'edgecolor','none');
    title(['xNor = ',num2str(xg(i))])
    caxis([0 maxS])
    xlabel('zNor')
    ylabel('yNor')    
    colorbar
    pause(0.1)  
    1;   
end




