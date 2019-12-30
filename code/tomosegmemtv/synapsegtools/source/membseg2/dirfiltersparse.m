function T = dirfiltersparse( I, C, Nx, Ny, Nz, F )
% DIRFILTERSPARSE Applies a directional filter over a scalar field in sparse mode
%   INPUT:  
%       I - Intput tomogram (scalar field where foreground is bright)
%       C - Coordinates of the sparse mask
%       Ni - Corrdinate i of the scalar field (normalized) for setting the direction of the 
%           filter
%       F - filter, all dimensions must have the same size and they must be odd 
%   OUTPUT:
%       T - Output filtered tomogram
%
%   See also: eig3dkmex, surfaceness
%   
%   AUTHOR: Antonio Martinez-Sanchez (an.martinez.s.sw@gmail.com)
%   REFERENCES:
%       [1] Martinez-Sanchez, A., et al. A differential structure approach to membrane segmentation 
%       in electron tomography. J. Struct. Biol. (2011), doi:10.1016/j.jsb.2011.05.010
%       [2] Martinez-Sanchez, A., et al. A ridge-based framework for segmentation of 3D electron 
%       microscopy datasets. J. Struct. Biol. (2012), http://dx.doi.org/10.1016/j.jsb.2012.10.002

%% Initialization
h = waitbar( 0, 'Applying 3D directional filter...' );

%% Creating halos
[Sx,Sy,Sz] = size( I );
w = floor( .5*length(F) );
w2 = 2 * w; 
Ph = zeros( w2+Sx, w2+Sy, w2+Sz ); 
T = zeros( Sx, Sy, Sz );
[Nhx,Nhy,Nhz] = size( Ph );

%% Rotation angles
P = ceil( acosd(-Nz) );
% R = Ny ./ Nx;
R = atan2( Ny, Nx );
R = R * (180/pi) - 90;

%% Loop for sparse coordinates
li = w + 1;
hx = Nhx - w;
hy = Nhy - w;
hz = Nhz - w;
Ph(li:hx,li:hy,li:hz) = I;
lc = length( C(:,1) );
lp = ceil( lc * .1 );
cont = 1;
contb = 1;
Cw = w + C;
for k = 1:lc
    
    % Get coordinates from vector
    x = C(k,1);
    y = C(k,2);
    z = C(k,3);
    xw = Cw(k,1);
    yw = Cw(k,2);
    zw = Cw(k,3);
            
    % Crop and rotation
    wx = xw-w:xw+w;
    wy = yw-w:yw+w;
    wz = zw-w:zw+w;
    r = R(x,y,z);
    p = P(x,y,z);
    Fr = tom_rotate( F, [0,r,p] );
    Tc = Ph(wx,wy,wz);

    % Filter weight
    T(x,y,z) = abs( sum(sum(sum( Tc .* Fr ))) );   
    
    % Update progress bar
    if cont >= lp
        waitbar( contb*.1, h );
        contb = contb + 1;
        cont = 1;
    end
    cont = cont + 1;
end

close( h );

end