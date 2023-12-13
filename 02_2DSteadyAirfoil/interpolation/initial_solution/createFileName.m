function [out, out_orig ] = createFileName(fileName,fileIdx)

if fileIdx == 12
    out = '/ddt0\(U\)';           out_orig = '/ddt0\(U\)_orig';     
elseif fileIdx == 13
    out = '/ddt0\(nuTilda\)';     out_orig = '/ddt0\(nuTilda\)_orig'; 
elseif fileIdx == 14
    out = '/ddtCorrDdt0\(U\)';    out_orig = '/ddtCorrDdt0\(U\)_orig';  
elseif fileIdx == 15
    out= '/ddtCorrDdt0\(Uf\)';   out_orig = '/ddtCorrDdt0\(Uf\)_orig'; 
else
    out=fileName;
end
