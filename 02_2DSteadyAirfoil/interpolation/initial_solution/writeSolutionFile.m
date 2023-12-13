function  writeSolutionFile(fileIdx,path,val,nline)

%copyFile = char([directory,'/nuTilda_orig']);

[out,out_orig,nout] = dataSelection(fileIdx);
out2 = createFileName(out,fileIdx);
fileName = char([path,out]);
copyFile = char([path,out_orig]);
fileName2 = char([path,out2]);
copyFile2 = char([fileName2,'_orig']);


if ~exist(copyFile, 'file')
    cmd = char(['cp ',fileName2,' ',copyFile2]); 
    system(cmd);
end

fid = fopen(copyFile,'r');
fo = fopen(fileName,'w');
startIdx = '('; endIdx1 = ')'; endIdx2 = '}';  %or use tline(find(~isspace(tline)))
endIdx = endIdx1; % initial idx
valIdx = 0; % if this value is 1, do interpolation
count = 0; 

n = length(val);
for line=1:nline
    tline = fgetl(fid);
    tline_noSpace = strrep(tline, ' ', '');
    nWords = length(tline_noSpace);
    
    if nout >1 && nWords > 2 && valIdx == 1 
        tline = tline(2:end-1); 
    end
    
    if nWords == 1 && tline_noSpace(1) == startIdx 
        valIdx = 1;
    elseif nWords == 1 && tline_noSpace(1) == endIdx 
        valIdx = 0; endIdx = endIdx2;
    end
    
    
    if valIdx == 1 && ~isempty( str2num(tline) )
        count = count + 1;
        if count>n
            disp('writing error');
        end
        if nout == 1
            fprintf(fo, '%10.6e\n', val(count,:));
        elseif nout == 3
            fprintf(fo, '(%f %f %f)\n', val(count,1),val(count,2),val(count,3) );
        end
    else
        fprintf(fo,'%s\n',tline);        
    end
    1;
end

fclose(fid);
fclose(fo);

