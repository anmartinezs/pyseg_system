function S = mdtvoting( E, s, sg, m, d, v )
% MDTVOTING  Applies multiscale dense 2D tensor voting for ridge detection
%            in a range of input thickness (gaussian prefiltering) and 
%            scale factor
%   INPUT:  
%       E - Input tomogram (foreground dark)
%       s - Scale factor range [s_min s_max s_step]
%       sg - variance for gaussian prefiltering range [sg_min sg_max sg_step]
%       m - Missing wedge semiangle in Z axis, it tries to prevent missing wedge effects 
%       if less than 0 is disabled 
%       d - Input data; 1- foreground black, otherwise- foreground bright
%       v - If equal to 1 verbose mode activated (disabled by default)
%   OUTPUT:
%       S - Output saliency
%
%   See also: steer2dtvoting, dtvotinge
%   
%   AUTHOR: Antonio Martinez-Sanchez (an.martinez.s.sw@gmail.com)
%   REFERENCES:
%       [1] Martinez-Sanchez A., et al (2014). Robust membrane detection based on tensor voting 
%       for electron tomography.  186(1), 49-61.

%% Initialization
sv = s(1):s(3):s(2);
sgv = sg(1):sg(3):sg(2);
S = zeros( size( E ) );
if d ~= 1
    [p,c] = linmap( min(min(min(E))), max(max(max(E))), 1, 0 );
    J = E*p + c;
else
    J = E;
end

%% Loop
for k = sv
    for kk = sgv
        
        Sh = dtvoting( J, k, kk, m, 2, 0 );
        Id = Sh > S;
        S(Id) = Sh(Id);
        
    end
    if v == 1
        fprintf( 1, 'Progress...%.2f%%\n', (100*sv(k))/s(2) );
    end
end
