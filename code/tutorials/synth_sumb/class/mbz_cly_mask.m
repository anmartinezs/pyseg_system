function [ M ] = mbz_cly_mask(sz, s, h, c, r, g)
%mbz_cly_mask Generates a cylindric mask for membrane subvolumes whose
%                      membrane is on plane XY
%   sz: subvolume size (all parameters are in voxels)
%   s: starting Z axis point for the cylinder
%   h: cylinder heigh in Z axis
%   c: cylinder center on XY plane
%   r: cylinder radius
%   g: sigma for smoothing Gaussian filter

% Unbounded cylinder
M = tom_cylindermask(ones(sz), r, 0, c);

% Bounding
if s > 0
    M(:,:,1:s-1) = 0;
end
if h < (sz(3) - 1)
    M(:,:,h+1:sz(3)) = 0;
end

% Smoothing
if g > 0
    M = imgaussfilt3(M, g);
end

end
