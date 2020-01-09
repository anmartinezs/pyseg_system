function [P,V] = surfaceness( I, s, m, e, v )
% SURFACENESS Enhances local ridges (or edges) with surface shape
%   INPUT:  
%       I - Input tomogram (foreground bright), input data must be single and double
%       s - Scale factor 
%       m - If equal to 1 ridge detection (default), otherwise edge detection
%       e - Mode for resolving eigenproblem; 1- Fast (default), 2- Intermediate,
%       otherwise- Accurate
%       v - If equal to 1 verbose mode activated (disabled by default)
%   OUTPUT:
%       F - filtered tomogram with the membranes enhanced, -1 if error
%       Vi - Coordinate i of the normal to membrane
%
%   See also: tomosegmemtv
%   
%   AUTHOR: Antonio Martinez-Sanchez (an.martinez.s.sw@gmail.com)
%   REFERENCES:
%       [1] Martinez-Sanchez A., et al. Robust membrane detection based on tensor voting 
%       for electron tomography. J Struct Biol. 186 (2014) 49-61.

%% Get Tensor (the result is better if the smoothing is done after the derivatives, the opposite
%% would be faster)
Ix = diff3d( I, 1 );
Iy = diff3d( I, 2 );
Iz = diff3d( I, 3 );
if m == 1
    if v
        fprintf( 1, 'Hessian tensor...\n' );
    end
    % Hessian tensor
    Ixx = diff3d( Ix, 1 );
    Iyy = diff3d( Iy, 2 );
    Izz = diff3d( Iz, 3 );
    Ixy = diff3d( Ix, 2 );
    Ixz = diff3d( Ix, 3 );
    Iyz = diff3d( Iz, 3 );
else
    if v
        fprintf( 1, 'Structure tensor...\n' );
    end
    % Structure tensor
    Ixx = Ix .* Ix;
    Iyy = Iy .* Iy;
    Izz = Iz .* Iz;
    Ixy = Ix .* Iy;
    Ixz = Ix .* Iz;
    Iyz = Iy .* Iz;
end
% Smoothing
Ix = angauss( Ix, s, 1 );
Iy = angauss( Iy, s, 1 );
Iz = angauss( Iz, s, 1 );
Ixx = angauss( Ixx, s, 1 );
Iyy = angauss( Iyy, s, 1 );
Izz = angauss( Izz, s, 1 );
Ixy = angauss( Ixy, s, 1 );
Ixz = angauss( Ixz, s, 1 );
Iyz = angauss( Iyz, s, 1 );

%% Eigen problem
if v
    fprintf( 1, 'Eigenproblem...\n' );
end
[L1,L2,~,V] = eig3dkmex( Ixx, Iyy, Izz, Ixy, Ixz, Iyz, e );
clear Ixx;
clear Iyy;
clear Izz;
clear Ixy;
clear Ixz;
clear Iyz;

%% Non-maximum suppression
if v
    fprintf( 1, 'Non-maximum suppression...\n' );
end
% Halo cropping
[Nx,Ny,Nz] = size( I );
off = ceil( 3*s );
M = zeros( Nx, Ny, Nz );
M(off:Nx-off-1,off:Ny-off-1,off:Nz-off-1) = 1;
% Manifoldless estimation
L1 = M .* L1;
M = L1.^2 ./ (Ix.*Ix+Iy.*Iy+Iz.*Iz);
M = M .* (L1<0);
clear Ix;
clear Iy;
clear Iz;
% Non-maximum suppresion
L1 = abs( L1 );
P = nonmaxsup( L1, M>.03, V(:,:,:,1), V(:,:,:,2), V(:,:,:,3) );
clear M;

%% Local properties
if v
    fprintf( 1, 'Local properties...\n' );
end
P = ( 1 - exp(-(1/mean(L1(P>0)))*L1) ) .* P; % Sharpness
L2 = 1 - abs(L2) ./ L1;
L2(L1==0) = 0;
clear L1;
P = ( 1 - exp(-(1/mean(L2(P>0)))*L2) ) .* P; % Planeless
clear L2;

%% Finishing
Pm = P > 0;
V(:,:,:,1) = V(:,:,:,1) .* Pm;
V(:,:,:,2) = V(:,:,:,2) .* Pm;
V(:,:,:,3) = V(:,:,:,3) .* Pm;

end