% Calculates partial derivate along k dimension of tomogram T
% Author: Antonio Martínez Sánchez
% T -> tomogram (3D image)
% k -> 1: x-dimension, 2: y-dimension and otherwise: z-dimension
function [D] = diff3d( T, k )

[Nx Ny Nz] = size( T );
Idp = zeros( Nx, Ny, Nz );
Idn = zeros( Nx, Ny, Nz );

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