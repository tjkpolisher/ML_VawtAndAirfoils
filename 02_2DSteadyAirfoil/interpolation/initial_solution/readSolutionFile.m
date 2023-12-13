function [res,validIdx,allnFaces,nline,nout] = ...
    readSolutionFile(fileIdx,path,pos,faces,allStartFace,CNT)

[out,~,nout] = dataSelection(fileIdx);
out2 = createFileName(out,fileIdx);
fileName = char([path,out]);
fileName2 = char([path,out2]);


nlineFile = char([path,'/nline']);
cmd = char(['wc -l ',fileName2,' > ', nlineFile ]);
%system('wc -l p.txt > nline')
system(cmd);
fileID = fopen(nlineFile,'r');
nline = fscanf(fileID, '%d');
fclose(fileID);

fid = fopen(fileName,'r');
words = createWords;

part = [];
validIdx = [];
startIdx = '('; endIdx1 = ')'; endIdx2 = '}';  %or use tline(find(~isspace(tline)))
endIdx = endIdx1; % initial idx
valIdx = 0; % if this value is 1, do interpolation
count = 0; idxPart = 1; allnFaces = 0; volQuantityIdx = 0;

for line=1:nline
    tline = fgetl(fid);
    tline_noSpace = strrep(tline, ' ', '');
    nWords = length(tline_noSpace);
    if valIdx == 1 && nout == 3 && nWords>2
        tline = tline(2:end-1);
    end
    
    if line == 12 && ~isempty(strfind(tline,'vol'))
        volQuantityIdx = 1;
        1;
    end
    
    if isempty(part) && idxPart<15
        
        k = strfind(tline,words{idxPart});
        if ~isempty(k)
            part = words{idxPart};
            endIdx = choose_endIndicator(endIdx1,endIdx2,idxPart);
            allPc = [];  allVal = [];   countPart = 0; 
            startFace = allStartFace(idxPart);
        end        
       
    end
   
    if ~isempty(part)&& nWords == 1 && tline_noSpace(1) == startIdx 
        valIdx = 1;
    end
    
    %%% this is to read number of n values at each part
    if ~isempty(part) && valIdx == 0 && ~isempty( str2num(tline) )
        nFaces = str2double(tline);   
        allPc = zeros(nFaces,3);
        allVal = zeros(nFaces,nout);
        allnFaces = allnFaces + nFaces; 
    end

    %%% it terminates caluations inside the part
    if  nWords == 1 && tline_noSpace(1) == endIdx
       res.allPc{idxPart} = allPc;
       res.allVal{idxPart} = allVal;
       if ~isempty(allPc)
           validIdx = [validIdx;idxPart];
       end
       valIdx = 0;  part = []; 
       idxPart = idxPart + 1; 
    end
    
    
    if  valIdx == 1 && ~isempty( str2num(tline) )
        count = count + 1;
        countPart = countPart + 1;
        val = str2num(tline);
        f = startFace + countPart;
        if volQuantityIdx == 1 && idxPart == 1
            t = faces(nonzeros(CNT(f,:)),:);
            t = t(~isnan(t));
            t = unique(t);
        else
            t = faces(f,:);
        end
        

        pc = mean(pos(t,:));
        allPc(countPart,:) = pc;
        allVal(countPart,:) = val;
        1;
    
    end
    1;
end
fclose(fid);


function endIdx = choose_endIndicator(endIdx1,endIdx2,iter)
if iter == 1
    endIdx = endIdx1;
else
    endIdx = endIdx2;
end

function words = createWords



words{1} = 'internalField';
words{2} = 'outlet';
words{3} = 'up';
words{4} = 'bottom';
words{5} = 'inlet';
words{6} = 'ami_stator';

words{7} = 'back_stator';
words{8} ='front_stator';
words{9} = 'face1';
words{10} = 'blade1_extruded';
words{11} = 'blade2_extruded';
words{12} = 'blade3_extruded';
words{13} = 'ami_rotor_extruded';
words{14} = 'face1_top';



