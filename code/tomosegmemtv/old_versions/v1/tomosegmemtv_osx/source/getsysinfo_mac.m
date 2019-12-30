function [c,m,l] = getsysinfo_mac()
% GETSYSINFO_MAC Return system information on MacOS Computers
%   OUTPUT: If error in some output it will be set ot -1
%       c - number of available cores
%       m - memory size
%       l - Cache L2 size
%
%   See also:
%
%   AUTHOR: Antonio Martinez-Sanchez (an.martinez.s.sw@gmail.com)
%   REFERENCES:
%       [1] Martinez-Sanchez A., et al. Robust membrane detection based on tensor voting 
%       for electron tomography. sent to J Struct Biol. (2013)

[c,m,l] = getsysinfo_mac_kernel();

end