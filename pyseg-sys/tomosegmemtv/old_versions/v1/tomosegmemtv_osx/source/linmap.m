function [m c] = linmap( li, ui, lo, uo )
% LINMAP  Calcultes the parameters for remaping data linearly 
%   INPUT:  
%       li - Lower bound input range for independent variable
%       ui - Upper bound input range for independent variable
%       lo - Lower bound input range for dependent variable
%       uo - Upper bound input range for dependent variable
%   OUTPUT:
%       m - Linear slope
%       c - Linear offset
%
%   See also: 
%   
%   AUTHOR: Antonio Martinez-Sanchez (an.martinez.s.sw@gmail.com)
%   REFERENCES:
%       [1] Martinez-Sanchez A., et al. Robust membrane detection based on tensor voting 
%       for electron tomography. sent to J Struct Biol. (2013)

m = (uo-lo) / (ui-li);
c = uo - m*ui;