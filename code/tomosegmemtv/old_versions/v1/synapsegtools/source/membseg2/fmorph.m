function P = fmorph( I, it, mode )
% FMORPH Dilation/Erotion over a forground
%   INPUT:  
%       I - Input tomogram where 2 is foreground and 1 background surface
%       it: number iterations
%       I: if 1 erode, otherwise dilate
%   OUTPUT:
%       P: output foreground (2) and background (1)
%
%   See also: memblabel, mask2coord
%   
%   AUTHOR: Antonio Martinez-Sanchez (an.martinez.s.sw@gmail.com)
%   REFERENCES:
%       [1] Martinez-Sanchez, A., et al. A differential structure approach to membrane segmentation 
%       in electron tomography. J. Struct. Biol. (2011), doi:10.1016/j.jsb.2011.05.010
%       [2] Martinez-Sanchez, A., et al. A ridge-based framework for segmentation of 3D electron 
%       microscopy datasets. J. Struct. Biol. (2012), http://dx.doi.org/10.1016/j.jsb.2012.10.002

[Nx,Ny,Nz] = size( I );
if mode == 1
    H = I;
else
    H = (I==2) + 2*(I==1);
end

for kk = 1:it
    H2 = H;
    C = mask2coord( H==2 );
    lc = length( C(:,1) );

    for k = 1:lc

        % Get coordinates from vector
        x = C(k,1);
        y = C(k,2);
        z = C(k,3);

        % Get neighbourhood
        if (x>=2) && (x<=Nx-1) && (y>=2) && (y<=Ny-1) && (z>=2) && (z<=Nz-1)
            W = H2(x-1:x+1,y-1:y+1,z-1:z+1);
            c = sum( W(W>0)~=2 );
            if c > 0
                H(x,y,z) = 1;
            end
        end

    end
end

if mode == 1
    P = H;
else
    P = (H==2) + 2*(H==1);
end

end