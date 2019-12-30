function [L1,L2,L3,V1,V2,V3] = eig3dkmex( Ixx, Iyy, Izz, Ixy, Ixz, Iyz, mode )
% EIG3DKMEX  Multicore eigenvalues and eigenvectors computation in 3x3 symmetric real tensors 
%            of tomograms
%   INPUT:  
%       Iij - Tomograms with the i, j discrete diferentials, input data must be single and double
%       mode: 1-> analytical (fastest), 2-> hybird (intermediate) ohterwise-> Jacobi 
%       (the most precise)
%   OUTPUT:
%       Lk: the k eigen value ordered by its absolute value (descendant)
%       Vk: the corresponding eigenvector
%
%   See also: diff3d, surfaceness
%   
%   AUTHOR: Antonio Martinez-Sanchez (an.martinez.s.sw@gmail.com)
%   REFERENCES:
%       [1] Martinez-Sanchez A., et al. Robust membrane detection based on tensor voting 
%       for electron tomography. sent to J Struct Biol. (2013)

%% Preparing input data
[Nx,Ny,Nz] = size( Ixx );
N = Nx * Ny * Nz;
L1 = reshape( Ixx, N, 1 );
L2 = reshape( Iyy, N, 1 );
L3 = reshape( Izz, N, 1 );
V1h = reshape( Ixy, N, 1 );
V2h = reshape( Ixz, N, 1 );
V3h = reshape( Iyz, N, 1 );
if isa(Ixx,'double')
    moded = 1;
elseif isa(Ixx,'single')
    moded = 0;
else
    fprintf( 1, 'eig3dkmex.m: Input data must be double or single.\n' );
    return;
end

%% Computation
if moded == 1
    if mode == 1
        [L1,L2,L3,V1xm,V1ym,V1zm,V2xm,V2ym,V2zm,V3xm,V3ym,V3zm] = desyevvmex( L1, L2, L3, V1h, V2h, V3h );
    elseif mode == 2
        [L1,L2,L3,V1xm,V1ym,V1zm,V2xm,V2ym,V2zm,V3xm,V3ym,V3zm] = desyevhmex( L1, L2, L3, V1h, V2h, V3h );
    else
        [L1,L2,L3,V1xm,V1ym,V1zm,V2xm,V2ym,V2zm,V3xm,V3ym,V3zm] = desyevjmex( L1, L2, L3, V1h, V2h, V3h );
    end
else
    if mode == 1
        [L1,L2,L3,V1xm,V1ym,V1zm,V2xm,V2ym,V2zm,V3xm,V3ym,V3zm] = desyevvmex_single( L1, L2, L3, V1h, V2h, V3h );
    elseif mode == 2
        [L1,L2,L3,V1xm,V1ym,V1zm,V2xm,V2ym,V2zm,V3xm,V3ym,V3zm] = desyevhmex_single( L1, L2, L3, V1h, V2h, V3h );
    else
        [L1,L2,L3,V1xm,V1ym,V1zm,V2xm,V2ym,V2zm,V3xm,V3ym,V3zm] = desyevjmex_single( L1, L2, L3, V1h, V2h, V3h );
    end
end
clear V1h;
clear V2h;
clear V3h;

%% Preparing output data
L1 = reshape( L1, Nx, Ny, Nz );
L2 = reshape( L2, Nx, Ny, Nz );
L3 = reshape( L3, Nx, Ny, Nz );
if moded == 1
    V1 = zeros( Nx, Ny, Nz, 3 );
else
    V1 = single( zeros(Nx,Ny,Nz,3) );
end
V1(:,:,:,1) = reshape( V1xm, Nx, Ny, Nz );
clear V1xm;
V1(:,:,:,2) = reshape( V1ym, Nx, Ny, Nz );
clear V1ym;
V1(:,:,:,3) = reshape( V1zm, Nx, Ny, Nz );
clear V1zm;
if moded == 1
    V2 = zeros( Nx, Ny, Nz, 3 );
else
    V2 = single( zeros(Nx,Ny,Nz,3) );
end
V2(:,:,:,1) = reshape( V2xm, Nx, Ny, Nz );
clear V2xm;
V2(:,:,:,2) = reshape( V2ym, Nx, Ny, Nz );
clear V2ym;
V2(:,:,:,3) = reshape( V2zm, Nx, Ny, Nz );
clear V2zm;
if moded == 1
    V3 = zeros( Nx, Ny, Nz, 3 );
else
    V3 = single( zeros(Nx,Ny,Nz,3) );
end
V3(:,:,:,1) = reshape( V3xm, Nx, Ny, Nz );
clear V3xm;
V3(:,:,:,2) = reshape( V3ym, Nx, Ny, Nz );
clear V3ym;
V3(:,:,:,3) = reshape( V3zm, Nx, Ny, Nz );
clear V3zm;

end