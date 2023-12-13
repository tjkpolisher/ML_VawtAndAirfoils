function [out,out_orig,nout] = dataSelection(ix)

if ix == 1
    out = '/U';                 out_orig = '/U_orig';            nout = 3;
elseif ix == 2
    out = '/U_0';               out_orig = '/U_0_orig';         nout = 3;
elseif ix == 3
    out = '/Uf';                out_orig = '/Uf_orig';          nout = 3;
elseif ix == 4
    out = '/Uf_0';              out_orig = '/Uf_0_orig';        nout = 3;
elseif ix == 5
    out = '/p';                 out_orig = '/p_orig';           nout = 1;
elseif ix == 6
    out = '/phi';               out_orig = '/phi_orig';         nout = 1;
elseif ix == 7
    out = '/nuTilda';           out_orig = '/nuTilda_orig';     nout = 1;
elseif ix == 8
    out = '/nuTilda_0';         out_orig = '/nuTilda_0_orig';   nout = 1;
elseif ix == 9
    out = '/nut';               out_orig = '/nut_orig';         nout = 1;
elseif ix == 10
    out = '/meshPhi';           out_orig = '/meshPhi_orig';     nout = 1;
elseif ix == 11
    out = '/meshPhiCN_0';       out_orig = '/meshPhiCN_0_orig'; nout = 1;
elseif ix == 12
    out = '/ddt0(U)';           out_orig = '/ddt0(U)_orig';     nout = 3;
elseif ix == 13
    out = '/ddt0(nuTilda)';     out_orig = '/ddt0(nuTilda)_orig';   nout = 1;
elseif ix == 14
    out = '/ddtCorrDdt0(U)';    out_orig = '/ddtCorrDdt0(U)_orig';  nout = 3;
elseif ix == 15
    out = '/ddtCorrDdt0(Uf)';   out_orig = '/ddtCorrDdt0(Uf)_orig'; nout = 3;
end