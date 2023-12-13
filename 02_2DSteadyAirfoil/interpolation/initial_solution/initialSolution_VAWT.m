function initialSolution_VAWT

%disp(' mesh (for computation) connectivity loading ...')

%%%%%%%%%%%%%%%%%%%%% input %%%%%%%%%%%%%%%%%%%%%%%%%%
path = '/media/cfd3/HDD1/mesh_layer_sim8/sim48_l21';
pathRef = '/media/cfd3/HDD1/dimple_simulation3/sim40';
%pathRef = '/media/cfd3/HDD1/dimple_simulation3/sim40';
dummyResultFolder = '/0.05';
ResultFolderRef = '/2.8763';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pathDataForSimulation = char([path,'/0']);
pathData = char([path,dummyResultFolder]);
pathMesh = char([path,'/constant/polyMesh']);

pathDataRef =  char([pathRef,ResultFolderRef]);
pathMeshRef = char([pathRef,'/constant/polyMesh']);


 [pos,faces,startFace,CNT] = load_pos_faces_boundary(pathMesh);
 [posRef,facesRef,startFaceRef,CNTRef] =...
     load_pos_faces_boundary(pathMeshRef,pathDataRef);
 
%  load('all.mat')
%  
%  close all;
 
 
 for fileIdx = 1:15
     disp(['================== fileIDx : ',num2str(fileIdx),' start... =================='])
     implementationData(fileIdx,pos,faces,startFace,...
         posRef,facesRef,startFaceRef,pathData,pathDataRef,CNT,CNTRef)
     1;
 end

cmd = char(['cp ', pathData,'/*  ',pathDataForSimulation]);
system(cmd);

cmd = char(['cp -r ', pathMesh,' ',pathDataForSimulation]);
system(cmd);

1;

function implementationData(fileIdx,pos,faces,startFace,...
    posRef,facesRef,startFaceRef,pathData,pathDataRef,CNT,CNTRef)

[res,validIdx,allnFaces,nLine,nout] = ...
    readSolutionFile(fileIdx,pathData,pos,faces,startFace,CNT);


[resRef,validIdxRef] = readSolutionFile...
    (fileIdx,pathDataRef,posRef,facesRef,startFaceRef,CNTRef);

%%%%%
allVal = zeros(allnFaces,nout);
nStart = 0;
for iter=1:length(validIdx)

    allPcRef = resRef.allPc{validIdxRef(iter)};
    allValRef = resRef.allVal{validIdxRef(iter)};    
    if abs(max(allPcRef(:,3)) - min(allPcRef(:,3))) > 1e-6
        disp('zRef direction reconsideration required')
    end
   
    allPc = res.allPc{validIdx(iter)};
    %allVal = res.allVal{validIdx(iter)};    
    if abs(max(allPc(:,3)) - min(allPc(:,3))) > 1e-6
        disp('z direction reconsideration required')
    end
    [allVal,nStart] = ...
    interpolationValue(allPcRef,allPc,allValRef,allVal,nStart,nout);

   1; 
end

writeSolutionFile(fileIdx,pathData,allVal,nLine)







function [allVal,nStart] = ...
    interpolationValue(allPcRef,allPc,allValRef,allVal,nStart,nout)

xmax = max(allPcRef(:,1)); xmin = min(allPcRef(:,1));
ymax = max(allPcRef(:,2)); ymin = min(allPcRef(:,2));
n = size(allPc,1); ix = (1:n) + nStart;

if nout == 1    
   val = cal_allVal(allValRef,allPcRef,allPc,xmax,xmin,ymax,ymin);
   allVal(ix) = val;
elseif nout == 3
   val = cal_allVal(allValRef(:,1),allPcRef,allPc,xmax,xmin,ymax,ymin);
   allVal(ix,1) = val; 

   val = cal_allVal(allValRef(:,2),allPcRef,allPc,xmax,xmin,ymax,ymin);
   allVal(ix,2) = val; 

   val = cal_allVal(allValRef(:,3),allPcRef,allPc,xmax,xmin,ymax,ymin);
   allVal(ix,3) = val; 

end

nStart = nStart + n;


function val = cal_allVal(allValRef,allPcRef,allPc,xmax,xmin,ymax,ymin)

if xmax-xmin<1e-6
    val = interp1(allPcRef(:,2),allValRef,allPc(:,2));
elseif ymax-ymin<1e-6
    val = interp1(allPcRef(:,1),allValRef,allPc(:,1));
else
    F = scatteredInterpolant(allPcRef(:,1),allPcRef(:,2),allValRef);
    val = F(allPc(:,1), allPc(:,2));    
end

% dataFileName = char([pathData,'nuTilda']);
%                 %/Users/sahucko/Documents/MATLAB/opt4_5s/5
% [pos,faces,startFace] = load_pos_faces_boundary(scriptName,pathMesh);
% [res,validIdx,allnFaces,nLine] = ...
%     readSolutionFile(dataFileName,pathMesh,pos,faces,startFace);
% 
% %%%%%
% pathMeshRef = '/Users/sahucko/Documents/MATLAB/set24_10s/constant/polyMesh/';
% scriptNameRef= char([pathMeshRef, 'script.sh']);
% pathDataRef =  '/Users/sahucko/Documents/MATLAB/set24_10s/10/';
% dataFileNameRef =  char([pathDataRef,'nuTilda']);
% 
% [posRef,facesRef,startFaceRef] = load_pos_faces_boundary(scriptNameRef,pathMeshRef);
% [resRef,validIdxRef] = readSolutionFile...
%     (dataFileNameRef,pathMeshRef,posRef,facesRef,startFaceRef);







