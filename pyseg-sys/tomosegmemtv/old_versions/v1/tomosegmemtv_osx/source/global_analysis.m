function C = global_analysis( M, tv, c, v )
% GLOBAL_ANALYSIS Perform a simple global analysis for tomogram segmentation, firstly binarizes 
%                 and secondly labels tomogram structures according to their size
%
%   INPUT:  
%       M - Input tomogram
%       tv - Binarization threshold
%       c - Voxel connectivity: 6, 8 (default) or 26
%       v - if 1 verbose mode activated
%   OUTPUT:
%       C - Output tomogram with the structures labeled according their size
%
%   See also: tomosegmemtv, cropt
%   
%   AUTHOR: Antonio Martinez-Sanchez (an.martinez.s.sw@gmail.com)
%   REFERENCES:
%       [1] Martinez-Sanchez A., et al. Robust membrane detection based on tensor voting 
%       for electron tomography. sent to J Struct Biol. (2013)

%% Thesholding and Volume analysis
[Nx,Ny,Nz] = size( M );
if (c~=6) && (c~=18) && (c~=26) 
    fprintf( 1, 'warning: connectivity must be to 6, 18 or 26, now it is set from %d to 18\n', c ); 
    c = 18;
end
[L,K] = bwlabeln( M>tv, c );
C = zeros( Nx, Ny, Nz );
Hk = zeros( K, 1 );

if v == 1
    fprintf( 1, '- Thresholding and Volume analysis\n' );  
end

%% Count loop
for x = 1:Nx
    for y = 1:Ny
        for z = 1:Nz             
            id = L(x,y,z);
            if id > 0
                Hk(id) = Hk(id) + 1;
            end
        end
    end
end

if v == 1
    fprintf( 1, '- Size measured\n' );  
end  

%% Write loop
for x = 1:Nx
    for y = 1:Ny
        for z = 1:Nz            
            id = L(x,y,z);
            if id > 0
                C(x,y,z) = Hk(id);
            end
        end
    end   
end

if v == 1
    fprintf( 1, '- Labels written\n' );  
end  
