function S = dtvoting( E, s, sg, m, d, v )
% STVOTING  Applies dense 2D tensor voting for ridge detection by steerable filters over 
%           every slice of a tomogram at X, Y and Z axes.
%   INPUT:  
%       E - Input tomogram (foreground dark)
%       s - Scale factor 
%       sg - variance for gaussian prefiltering
%       m - Missing wedge semiangle in Z axis, it tries to prevent missing wedge effects 
%       if less than 0 is disabled 
%       d - Input data; 1- foreground black, otherwise- forecast bright
%       v - If equal to 1 verbose mode activated (disabled by default)
%   OUTPUT:
%       S - Output saliency
%
%   See also: steer2dtvoting, dtvotinge
%   
%   AUTHOR: Antonio Martinez-Sanchez (an.martinez.s.sw@gmail.com)
%   REFERENCES:
%       [1] Martinez-Sanchez A., et al. Robust membrane detection based on tensor voting 
%       for electron tomography. J Struct Biol. 186 (2014) 49-61.

%% Initialization
if m > 0
    m = (90-m) * (pi/180);
end
[Nx,Ny,Nz] = size( E );
if isa(E,'double')
    S = zeros( Nx, Ny, Nz );
else
    S = single( zeros(Nx,Ny,Nz) );
end
if mod(Nx,2)
    Nx2 = floor( .5*Nx );
    Vnx = -Nx2:Nx2;
else
    Nx2 = .5*Nx;
    Vnx = -Nx2:Nx2-1;
end
if mod(Ny,2)
    Ny2 = floor( .5*Ny );
    Vny = -Ny2:Ny2;
else
    Ny2 = .5*Ny;
    Vny = -Ny2:Ny2-1;
end
if mod(Nz,2)
    Nz2 = floor( .5*Nz );
    Vnz = -Nz2:Nz2;
else
    Nz2 = .5*Nz;
    Vnz = -Nz2:Nz2-1;
end

%% Local smooth
J = angauss( E, sg, 1 );
if d==1
    [p,c] = linmap( min(min(min(J))), max(max(max(J))), 1, 0 );
else
    [p,c] = linmap( min(min(min(J))), max(max(max(J))), 0, 1 );
end
J = J*p + c;

%% XY plane
[X,Y] = meshgrid( Vny, Vnx );
H1 = X.*X + Y.*Y;
H2 = exp( -H1/(2*s*s) );
H3 = (X+1i*Y) ./ sqrt(H1);
W0 = H2 .* (H3).^0;
W0(Nx2+1,Ny2+1) = 1;
W0 = fft2( W0 ); % TODO: probably here we do trivial calculations
W2 = H2 .* (H3).^2;
W2(Nx2+1,Ny2+1) = 1;
W2 = fft2( W2 );
W4 = H2 .* (H3).^4;
W4(Nx2+1,Ny2+1) = 1;
W4 = fft2( W4 );
W6 = H2 .* (H3).^6;
W6(Nx2+1,Ny2+1) = 1;
W6 = fft2( W6 );
W8 = H2 .* (H3).^8;
W8(Nx2+1,Ny2+1) = 1;
W8 = fft2( W8 );
for z = 1:Nz
    I = J(:,:,z);
    Ax = diff2d( I, 1, 1 );
    Axx = diff2d( Ax, 1, 1 );
    Ay = diff2d( I, 2, 1 );
    Ayy = diff2d( Ay, 2, 1 );
    Axy = diff2d( Ax, 2, 1 );
    O = .5 * angle( Axx-Ayy + 1i*2*Axy ); % Orientation
    I = ((Axx+Ayy)<0) .* sqrt( Axx.*Axx+Ayy.*Ayy-2*Axx.*Ayy+4*Axy.*Axy );
    Sh = steer2dtvoting( I, O, W0, W2, W4, W6, W8 );
    S(:,:,z) = Sh;
    if v
        fprintf( 1, 'Progress Phase 1: %.2f%%...\n', (z*100)/Nz );
    end
end

%% XZ plane
[X,Z] = meshgrid( Vnz, Vnx );
H1 = X.*X + Z.*Z;
H2 = exp( -H1/(2*s*s) );
H3 = (X+1i*Z) ./ sqrt(H1);
W0 = H2 .* (H3).^0;
W0(Nx2+1,Nz2+1) = 1;
W0 = fft2( W0 ); % TODO: probably here we do trivial calculations
W2 = H2 .* (H3).^2;
W2(Nx2+1,Nz2+1) = 1;
W2 = fft2( W2 );
W4 = H2 .* (H3).^4;
W4(Nx2+1,Nz2+1) = 1;
W4 = fft2( W4 );
W6 = H2 .* (H3).^6;
W6(Nx2+1,Nz2+1) = 1;
W6 = fft2( W6 );
W8 = H2 .* (H3).^8;
W8(Nx2+1,Nz2+1) = 1;
W8 = fft2( W8 );
if m > 0
    for y = 1:Ny
% if y == 167
%     jol = 1;
% end
        I = reshape( J(:,y,:), Nx, Nz ); 
        Ax = diff2d( I, 1, 1 );
        Axx = diff2d( Ax, 1, 1 );
        Ay = diff2d( I, 2, 1 );
        Ayy = diff2d( Ay, 2, 1 );
        Axy = diff2d( Ax, 2, 1 );
        O = .5 * angle( Axx-Ayy + 1i*2*Axy ); % Orientation
        I = ((Axx+Ayy)<0) .* (abs(O)<m) .* sqrt( Axx.*Axx+Ayy.*Ayy-2*Axx.*Ayy+4*Axy.*Axy );
        Sh = steer2dtvoting( I, O, W0, W2, W4, W6, W8 );
        S(:,y,:) = S(:,y,:) + reshape(Sh,Nx,1,Nz);
        if v
            fprintf( 1, 'Progress Phase 2: %.2f%%...\n', (y*100)/Ny );
        end
    end
else
    for y = 1:Ny
        I = reshape( J(:,y,:), Nx, Nz ); 
        Ax = diff2d( I, 1, 1 );
        Axx = diff2d( Ax, 1, 1 );
        Ay = diff2d( I, 2, 1 );
        Ayy = diff2d( Ay, 2, 1 );
        Axy = diff2d( Ax, 2, 1 );
        O = .5 * angle( Axx-Ayy + 1i*2*Axy ); % Orientation
        I = ((Axx+Ayy)<0) .* sqrt( Axx.*Axx+Ayy.*Ayy-2*Axx.*Ayy+4*Axy.*Axy );
        Sh = steer2dtvoting( I, O, W0, W2, W4, W6, W8 );
        S(:,y,:) = S(:,y,:) + reshape(Sh,Nx,1,Nz);
        if v
            fprintf( 1, 'Progress Phase 2: %.2f%%...\n', (y*100)/Ny );
        end
    end
end

%% YZ plane
[Y,Z] = meshgrid( Vnz, Vny );
H1 = Y.*Y + Z.*Z;
H2 = exp( -H1/(2*s*s) );
H3 = (Y+1i*Z) ./ sqrt(H1);
W0 = H2 .* (H3).^0;
W0(Ny2+1,Nz2+1) = 1;
W0 = fft2( W0 ); % TODO: probably here we do trivial calculations
W2 = H2 .* (H3).^2;
W2(Ny2+1,Nz2+1) = 1;
W2 = fft2( W2 );
W4 = H2 .* (H3).^4;
W4(Ny2+1,Nz2+1) = 1;
W4 = fft2( W4 );
W6 = H2 .* (H3).^6;
W6(Ny2+1,Nz2+1) = 1;
W6 = fft2( W6 );
W8 = H2 .* (H3).^8;
W8(Ny2+1,Nz2+1) = 1;
W8 = fft2( W8 );
if m > 0
    for x = 1:Nx
        I = reshape( J(x,:,:), Ny, Nz ); 
        Ax = diff2d( I, 1, 1 );
        Axx = diff2d( Ax, 1, 1 );
        Ay = diff2d( I, 2, 1 );
        Ayy = diff2d( Ay, 2, 1 );
        Axy = diff2d( Ax, 2, 1 );
        O = .5 * angle( Axx-Ayy + 1i*2*Axy ); % Orientation
        I = ((Axx+Ayy)<0) .* (abs(O)<m) .* sqrt( Axx.*Axx+Ayy.*Ayy-2*Axx.*Ayy+4*Axy.*Axy );
        Sh = steer2dtvoting( I, O, W0, W2, W4, W6, W8 );
        S(x,:,:) = S(x,:,:) + reshape(Sh,1,Ny,Nz);
        if v
            fprintf( 1, 'Progress Phase 3: %.2f%%...\n', (x*100)/Nx );
        end
    end
else
    for x = 1:Nx
        I = reshape( J(x,:,:), Ny, Nz ); 
        Ax = diff2d( I, 1, 1 );
        Axx = diff2d( Ax, 1, 1 );
        Ay = diff2d( I, 2, 1 );
        Ayy = diff2d( Ay, 2, 1 );
        Axy = diff2d( Ax, 2, 1 );
        O = .5 * angle( Axx-Ayy + 1i*2*Axy ); % Orientation
        I = ((Axx+Ayy)<0) .* sqrt( Axx.*Axx+Ayy.*Ayy-2*Axx.*Ayy+4*Axy.*Axy );
        Sh = steer2dtvoting( I, O, W0, W2, W4, W6, W8 );
        S(x,:,:) = S(x,:,:) + reshape(Sh,1,Ny,Nz);
        if v
            fprintf( 1, 'Progress Phase 3: %.2f%%...\n', (x*100)/Nx );
        end
    end
end

%% Average
it = 1 / 3;
S = it * S;

end