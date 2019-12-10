function C = cropt( T, x, y, z )
% CROPT Crop a tomogram
%   INPUT:  
%		T - input tomogram
%		i = [li hi] - range for coodinate i 
%   OUTPUT:
%		C - output cropped tomogram
%
%   See also:
%   
%   AUTHOR: Antonio Martinez-Sanchez (an.martinez.s.sw@gmail.com)
%   REFERENCES:
%       [1] Martinez-Sanchez, A., et al. A differential structure approach to membrane segmentation 
%       in electron tomography. J. Struct. Biol. (2011), doi:10.1016/j.jsb.2011.05.010
%       [2] Martinez-Sanchez, A., et al. A ridge-based framework for segmentation of 3D electron 
%       microscopy datasets. J. Struct. Biol. (2012), http://dx.doi.org/10.1016/j.jsb.2012.10.002

C = zeros( size(T) );
C(x(1):x(2),y(1):y(2),z(1):z(2)) = T(x(1):x(2),y(1):y(2),z(1):z(2));

end