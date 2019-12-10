%% Multiscale vesicles detector by template matching
% Author: Antonio Martinez-Sanchez
% I: input tomogram
% r: radius range [min,max]
% t: iteration step in r
% s: membrane thickness
% aw: wedge semiangle in degress
% rw: angle rotation for the wedge
% V: coeficients scale integration
% R: estimated radii for the coeficients
function [V,R] = ves3dtm( I, r, t, s, aw, rw )

%% Initialization
h = waitbar( 0, 'Vesicles multiscale template matching...' );
[Nx,Ny,Nz] = size( I );
Rg = r(1):t:r(2);
V = zeros( Nx, Ny, Nz ) - Inf;
R = zeros( Nx, Ny, Nz );
lr = length( Rg );

%% Radius loop
cont = 0;
for ro = Rg
    
    % Update template
    T = getvestemp( ro, s, ceil(2*ro+4*s), aw, rw, 1 );
    
    % Template matching
    H = tom_os3_corr( I, T );
    
    % Update maximum
    Id = H > V;
    V(Id) = H(Id);
    R(Id) = ro;
    
    cont = cont + 1;
    waitbar( cont/lr, h );
    
end

close( h );

end