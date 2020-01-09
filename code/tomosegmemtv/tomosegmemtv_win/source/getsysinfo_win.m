function [c,m,l] = getsysinfo_win()
% GETSYSINFO_WIN Return system information on Windows Computers
%   OUTPUT: If error in some output it will be set ot -1
%       c - number of available cores
%       m - memory size
%       l - Cache size
%
%   See also:
%
%   AUTHOR: Antonio Martinez-Sanchez (an.martinez.s.sw@gmail.com)
%   REFERENCES:
%       [1] Martinez-Sanchez A., et al. Robust membrane detection based on tensor voting 
%       for electron tomography. J Struct Biol. 186 (2014) 49-61.

[c,m,l] = getsysinfo_win_kernel();

end