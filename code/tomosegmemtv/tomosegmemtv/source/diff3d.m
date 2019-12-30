function [D] = diff3d( T, k )
% DIFF3D  Calculates partial derivative along any dimension in a tomogram.
%   INPUT:  
%       Is - Input tomogram
%       k - 1: x-dimension, 2: y-dimension and otherwise: z-dimension
%   OUTPUT:
%       D - Output tomgram
%
%   See also: angauss, eig3dkmex, diff2d
%   
%   AUTHOR: Antonio Martinez-Sanchez (an.martinez.s.sw@gmail.com)
%   REFERENCES:
%       [1] Martinez-Sanchez A., et al. Robust membrane detection based on tensor voting 
%       for electron tomography. J Struct Biol. 186 (2014) 49-61.

[Nx Ny Nz] = size( T );
if isa(T,'double')
    Idp = zeros( Nx, Ny, Nz );
    Idn = zeros( Nx, Ny, Nz );
else
    Idp = single( zeros(Nx,Ny,Nz) );
    Idn = single( zeros(Nx,Ny,Nz) );
end

if k == 1
    Idp(1:Nx-1,:,:) = T(2:Nx,:,:);
    Idn(2:Nx,:,:) = T(1:Nx-1,:,:);
    % Pad extremes
    Idp(Nx,:,:) = Idp(Nx-1,:,:);
    Idn(1,:,:) = Idn(2,:,:);
elseif k == 2
    Idp(:,1:Ny-1,:) = T(:,2:Ny,:);
    Idn(:,2:Ny,:) = T(:,1:Ny-1,:);
    % Pad extremes
    Idp(:,Ny,:) = Idp(:,Ny-1,:);
    Idn(:,1,:) = Idn(:,2,:);
else
    Idp(:,:,1:Nz-1) = T(:,:,2:Nz);
    Idn(:,:,2:Nz) = T(:,:,1:Nz-1);
    % Pad extremes
    Idp(:,:,Nz) = Idp(:,:,Nz-1);
    Idn(:,:,1) = Idn(:,:,2);
end
    
D = (Idp-Idn) * 0.5;