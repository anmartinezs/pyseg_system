%% Create a 3D vesicle template
% Author: Antonio Martinez-Sanchez
% r: radius 
% t: thickness
% s: size of the template (should be odd)
% w: wedge semiangle in degress
% rw: angle rotation for the wedge
% mode: 1-> hard in range [-1,1], 2-> hard in range [0,1], otherwise -> soft
% T: output 3D template
function [T] = getvestemp( r, t, s, aw, rw, mode )

%% Grid
w = .5 * (s-1); 
[X,Y,Z] = meshgrid( -w:w, -w:w, -w:w );
R = sqrt( X.*X + Y.*Y + Z.*Z );
T = zeros( s, s, s );

t2 = .5 * t;
M1 = logical( R < (r-t2) );
M2 = logical( (R<=(r+t2)) .* (~M1) );

if mode == 1
    T(M2) = -1;
    T(~M2) = 1;
elseif mode == 2
    T(M2) = 0;
    T(~M2) = 1;
else
    R1 = (pi*R) ./ (2*r-t);
    T(M1) = cos(R1(M1));
    R1 = ((pi/t)*(R-r));
    T(M2) = -cos(R1(M2));
end

% M = logical( R <= (r+2*t2+1) );

if aw > 0
    W = tom_wedge( T, aw, rw );
    T = real( ifftn( (fftn(T).*fftshift(W)) ) );
end


end