function D = diff2d( I, k, h )
% DIFF2D  Differentiation in 2D images
%   INPUT:  
%       Is - Input image
%       k - Dimension 1: x-dimension, otherwise: y-dimension
%       h - Step size
%   OUTPUT:
%       D - Output tomogram
%
%   See also: diff3d, dtvoting, dtvotinge
%   
%   AUTHOR: Antonio Martinez-Sanchez (an.martinez.s.sw@gmail.com)
%   REFERENCES:
%       [1] Martinez-Sanchez A., et al. Robust membrane detection based on tensor voting 
%       for electron tomography. J Struct Biol. 186 (2014) 49-61.

[Ny Nx] = size(I);
Idp = zeros( [Ny Nx] ); 
Idn = zeros( [Ny Nx] );

if k == 1
    Idp(:,1:Nx-1) = I(:,2:Nx);
    Idn(:,2:Nx) = I(:,1:Nx-1);
    % Pad extremes
    Idp(:,Nx) = Idp(:,Nx-1);
    Idn(:,1) = Idn(:,2);
else
    Idp(1:Ny-1,:) = I(2:Ny,:);
    Idn(2:Ny,:) = I(1:Ny-1,:);
    % Pad extremes
    Idp(Ny,:) = Idp(Ny-1,:);
    Idn(1,:) = Idn(2,:);
end

D = (Idp-Idn) / (2*h);

end