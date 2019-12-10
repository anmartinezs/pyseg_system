function [C,Nx,Ny,Nz] = mask2coord( T )
% MASK2COORD Get sparse coordinates vector from mask
%   INPUT:  
%       I - Input tomogram (binary)
%   OUTPUT:
%       C - 3 columns vector with the three coordinates [X,Y,Z]
%       Ni - Tomgram sizes
%
%   See also: dirfiltersparse
%   
%   AUTHOR: Antonio Martinez-Sanchez (an.martinez.s.sw@gmail.com)
%   REFERENCES:
%       [1] Martinez-Sanchez, A., et al. A differential structure approach to membrane segmentation 
%       in electron tomography. J. Struct. Biol. (2011), doi:10.1016/j.jsb.2011.05.010
%       [2] Martinez-Sanchez, A., et al. A ridge-based framework for segmentation of 3D electron 
%       microscopy datasets. J. Struct. Biol. (2012), http://dx.doi.org/10.1016/j.jsb.2012.10.002

%% Initialization
[Nx,Ny,Nz] = size( T );
Id = T > 0;
ld = sum(sum(sum( Id )));
C = zeros( ld, 3 );
[X,Y,Z] = meshgrid( 1:Ny, 1:Nx, 1:Nz );

%% Convertion
C(:,1) = Y(Id);
C(:,2) = X(Id);
C(:,3) = Z(Id);

end