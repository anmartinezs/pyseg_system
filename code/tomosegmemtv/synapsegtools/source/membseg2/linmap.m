function [m,c] = linmap( li, ui, lo, uo )
% DIFF2D  Differentiation in 2D images
%   INPUT:  
%       li,ui - minimum and maximum input variable values (X)
%       lo,uo - minimum and maximum output variable values (Y)
%   OUTPUT: Linear conversion can be done with Y = X*m + c;
%       m - linear slope parameter
%		c - offset parameter
%
%   See also: dtvoting, dtvotinge
%   
%   AUTHOR: Antonio Martinez-Sanchez (an.martinez.s.sw@gmail.com)
%   REFERENCES:
%       [1] Martinez-Sanchez, A., et al. A differential structure approach to membrane segmentation 
%       in electron tomography. J. Struct. Biol. (2011), doi:10.1016/j.jsb.2011.05.010
%       [2] Martinez-Sanchez, A., et al. A ridge-based framework for segmentation of 3D electron 
%       microscopy datasets. J. Struct. Biol. (2012), http://dx.doi.org/10.1016/j.jsb.2012.10.002

m = (uo-lo) / (ui-li);
c = uo - m*ui;