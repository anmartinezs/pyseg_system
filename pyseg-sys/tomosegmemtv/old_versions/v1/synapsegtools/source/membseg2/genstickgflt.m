function J = genstickgflt( ro, s )
% GENSTICKGFLT Generates (unrotated) stickness kernel with gaussian profile
%   INPUT:  
%       ro: Radius
%       s - Scale
%   OUTPUT:
%       J - Output kernel (unrotated) with size s+ceil(s/2)
%
%   See also: dirfiltersparse
%   
%   AUTHOR: Antonio Martinez-Sanchez (an.martinez.s.sw@gmail.com)
%   REFERENCES:
%       [1] Martinez-Sanchez, A., et al. A differential structure approach to membrane segmentation 
%       in electron tomography. J. Struct. Biol. (2011), doi:10.1016/j.jsb.2011.05.010
%       [2] Martinez-Sanchez, A., et al. A ridge-based framework for segmentation of 3D electron 
%       microscopy datasets. J. Struct. Biol. (2012), http://dx.doi.org/10.1016/j.jsb.2012.10.002

%% Initialization for Plate kernel
w = ceil( 2*ro );
[X,Y,Z] = meshgrid( -w:w, -w:w, -w:w );
R = sqrt( X.*X + Y.*Y + Z.*Z );
w1 = w + 1;
R(w1,w1,w1) = .001 * min(R(R>0)); % For avoiding division by zero
Rxy = sqrt( X.*X + Y.*Y );
Rxy(w+1,w+1,:) = .001 * min(R(R>0)); % For avoiding division by zero
P = acos( Rxy ./ R ) - pi/2;
P(P==0) = .001; % For avoiding division by zero
M = (P<=(pi/4)) .* (P>=(-pi/4));
M = M .* (Z>=0);

%% Stick kernel
s2 = 1 / (2*s*s);
G = 1/16;
J = G * exp(-s2*((abs(Z)-ro).^2+X.*X+Y.*Y)) .* (cos(P)).^2 .* (1+cos(4*P));  
J = J .* M;
J(w1,w1,w1) = 0;