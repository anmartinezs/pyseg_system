function [C,H] = vol3d( M, ax, c )
% VOL3D Measures the size of connected objects along 2D slices
%   INPUT:  
%       M - Binary input tomogram
%       ax: Reference axis (1->x, 2->y, otherwise->z)
%       c: 2D connectivity, either 4 or 8
%   OUTPUT:
%       C: Tomograms with components classfied by their volume
%       H: Components volume histogram
%
%   See also: memblabel
%   
%   AUTHOR: Antonio Martinez-Sanchez (an.martinez.s.sw@gmail.com)
%   REFERENCES:
%       [1] Martinez-Sanchez, A., et al. A differential structure approach to membrane segmentation 
%       in electron tomography. J. Struct. Biol. (2011), doi:10.1016/j.jsb.2011.05.010
%       [2] Martinez-Sanchez, A., et al. A ridge-based framework for segmentation of 3D electron 
%       microscopy datasets. J. Struct. Biol. (2012), http://dx.doi.org/10.1016/j.jsb.2012.10.002

[Nx,Ny,Nz] = size( M );
L = zeros( Nx, Ny, Nz );
C = zeros( Nx, Ny, Nz );
if ax == 1
    K = zeros( Nx );
elseif ax == 2 
    K = zeros( Ny );
else
    K = zeros( Nz );
end
cont = 1;

% Components identification
if ax == 1
    
    for k = 1:Nx
        Mh = reshape( M(k,:,:), Ny, Nz );
        [L(k,:,:),K(k)] = bwlabel( Mh, c );
        H = zeros( K(k), 1 );

        for kk = 1:K(k)
            Id = reshape( L(k,:,:)==kk, Ny, Nz);
            H(cont) = sum(sum( Id.*Mh ));
            C(k,:,:) = C(k,:,:) + reshape(Id*H(cont),1,Ny,Nz);
            cont = cont + 1;
        end
    end
    
elseif ax == 2
    
    for k = 1:Ny
        Mh = reshape( M(:,k,:), Nx, Nz );
        [L(:,k,:),K(k)] = bwlabel( Mh, c );
        H = zeros( K(k), 1 );
        
        for kk = 1:K(k)
            Id = reshape( L(:,k,:)==kk, Nx, Nz );
            H(cont) = sum(sum( Id.*Mh ));
            C(:,k,:) = C(:,k,:) + reshape(Id*H(cont),Nx,1,Nz);
            cont = cont + 1;
        end
    end
    
else
    
    for k = 1:Nz
        [L(:,:,k),K(k)] = bwlabel( M(:,:,k), c );
        H = zeros( K(k), 1 );
        
        for kk = 1:K(k)
            Id = L(:,:,k)==kk;
            H(cont) = sum(sum( Id.*M(:,:,k) ));
            C(:,:,k) = C(:,:,k) + Id*H(cont);
            cont = cont + 1;
        end
    end
    
end
