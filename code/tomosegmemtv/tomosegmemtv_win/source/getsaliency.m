function [S,Vx,Vy,Vz] = getsaliency( I, s, mode )
% GETSALIENCY  Get ridge saliency and its normal
%   INPUT:  
%       E - Input tomogram (foreground bright), input data must be single and double
%       s - smoothing scale
%       mode: 1-> analytical (fastest), 2-> hybird (intermediate) ohterwise-> Jacobi 
%       (the most precise)
%   OUTPUT:
%       S - normal vector field
%       Vi - the i coordinate of the normal vector field
%
%   See also: eig3dkmex, nonmaxsup, diff3d, angauss
%   
%   AUTHOR: Antonio Martinez-Sanchez (an.martinez.s.sw@gmail.com)
%   REFERENCES:
%       [1] Martinez-Sanchez A., et al. Robust membrane detection based on tensor voting 
%       for electron tomography. J Struct Biol. 186 (2014) 49-61.

%% Derivatives
Ih = angauss( I, s, 1 );
Ix = diff3d( Ih, 1 );
Iy = diff3d( Ih, 2 );
Iz = diff3d( Ih, 3 );
clear Ih;
Ixx = diff3d( Ix, 1 );
Iyy = diff3d( Iy, 2 );
Izz = diff3d( Iz, 3 );
Ixy = diff3d( Ix, 2 );
Ixz = diff3d( Ix, 3 );
Iyz = diff3d( Iy, 3 );

%% Eigenproblem
[S,~,~,V] = eig3dkmex( Ixx, Iyy, Izz, Ixy, Ixz, Iyz, mode );
clear Ixx;
clear Iyy;
clear Izz;
clear Ixy;
clear Ixz;
clear Iyz;
S = S .* (S<0);
Vx = V(:,:,:,1);
Vy = V(:,:,:,2);
Vz = V(:,:,:,3);

end