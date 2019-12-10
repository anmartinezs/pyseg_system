function [F,Vx,Vy,Vz] = membflt_kernel( I, s, t, v, m, w, e, d )
% MEMBFLT_KERNEL Kernel code for TOMOSEGMEMTV. Do not call this function directly, call tomosegmemtv instead.
%   INPUT:  
%       I - Input tomogram (foreground dark), input data must be single and double
%       s - Scale factor 
%       t - Membrane thickness factor
%       v - If equal to 1 verbose mode activated (disabled by default)
%       m - If equal to 1 ridge detection (default), otherwise edge detection
%       w - Missing wedge semiangle in Z axis, it tries to prevent missing wedge effects 
%       if less than 0 is disabled 
%       e - Mode for resolving eigenproblem; 1- Fast (default), 2- Intermediate,
%       otherwise- Accurate
%       d - Densification scale factor
%   OUTPUT:
%       F - filtered tomogram with the membranes enhanced, -1 if error
%       Vi - Coordinate i of the normal to membrane
%
%   See also: tomosegmemtv
%   
%   AUTHOR: Antonio Martinez-Sanchez (an.martinez.s.sw@gmail.com)
%   REFERENCES:
%       [1] Martinez-Sanchez A., et al. Robust membrane detection based on tensor voting 
%       for electron tomography. sent to J Struct Biol. (2013)

%% Enhance membranes by tensor voting
if v
    fprintf( 1, '- Dense tensor voting\n' );
end
if m == 1
    F = dtvoting( I, s, t, w, 1, 0 );
else
    F = dtvotinge( I, s, t, w, 0 );
end

%% Local properties
if v
    fprintf( 1, '- Surfaceness\n' );
end
[F] = surfaceness( F, 1, 1, e, 0 );

%% Refine result with tensor voting
if v
    fprintf( 1, '- Refining\n' );
end
F = dtvoting( F, d, .75, w, 2, 0 );
[F,Vx,Vy,Vz] = getsaliency( F, 1, 1 );

%% Segmentation and labeling (by size)
if v
    fprintf( 1, '- Segmentation\n' );
end
M = F<0;
F = -F;
B = nonmaxsup( F, M, Vx, Vy, Vz );
F = F .* B;
Vx = Vx .* B;
Vy = Vy .* B;
Vz = Vz .* B;

end