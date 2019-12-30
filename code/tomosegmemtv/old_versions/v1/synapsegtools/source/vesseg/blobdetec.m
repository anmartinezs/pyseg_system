%% 3D Monge-Ampere operator for blob detection
% Author: Antonio Martinez-Sanchez
% T: intput tomogram
% s: scale for anistropic filtering
% r: aspect ratio for anistropic filtering
% B: blob detection
function B = blobdetec( T, s, r )

%% Anistropic filtering
I = angauss( T, s, r );

%% Directional derivatives
Ix = diff3d( I, 1 ); 
Iy = diff3d( I, 2 );
Iz = diff3d( I, 3 );
Ixx = diff3d( Ix, 1 ); 
Iyy = diff3d( Iy, 2 );
Izz = diff3d( Iz, 3 );
Ixy = diff3d( Ix, 2 ); 
Ixz = diff3d( Ix, 3 );
Iyz = diff3d( Iy, 3 );
clear Ix;
clear Iy;
clear Iz;
B = Ixx.*Iyy.*Izz + 2*Ixy.*Ixz.*Iyz - Ixx.*Iyz.*Iyz - Iyy.*Ixz.*Ixz - Izz.*Ixy.*Ixy;

end