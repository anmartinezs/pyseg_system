function C = cropt( T, x, y, z )
% CROPT Crop a tomogram, set to zero voxels in the border of the tomogram
%   INPUT:  
%		T - input tomogram
%		i = [li hi] - range for coodinate i 
%   OUTPUT:
%		C - output cropped tomogram
%
%   See also: global_analysis
%   
%   AUTHOR: Antonio Martinez-Sanchez (an.martinez.s.sw@gmail.com)
%   REFERENCES:
%       [1] Martinez-Sanchez A., et al. Robust membrane detection based on tensor voting 
%       for electron tomography. J Struct Biol. 186 (2014) 49-61.

C = zeros( size(T) );
C(x(1):x(2),y(1):y(2),z(1):z(2)) = T(x(1):x(2),y(1):y(2),z(1):z(2));

end