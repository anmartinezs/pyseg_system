%% Fill inner region for vesseg graphic usage
% Author: Antonio Martinez-Sanchez
% N: oriented normals (norm == 1)
% F: output filled tomogram
function F = filling( N )

%% Initilialization
[Nx,Ny,Nz,~] = size( N );
F = zeros( Nx, Ny, Nz );
szi = Nx .* Ny;
h = waitbar( 0, 'Size measuring and labeling...' );

%% Distance transform
[~,I] = bwdist( abs(N(:,:,:,1))>0 );

%% Loop for tomogram
for z = 1:Nz
    for y = 1:Ny
        for x = 1:Nx
            
            % Get nearest point coordinates
            id = double( I(x,y,z) );
            iz = ceil( id / szi );
            ih = id - (iz-1)*szi;
            iy = ceil( ih / Nx );
            ix = ih - (iy-1)*Nx;
            
            % Test if the point is in or out
            n = reshape( N(ix,iy,iz,:), 3, 1 );
            v = [x; y; z] - [ix; iy; iz];
            o = acos( dot(n,v) );
            if o < pi/2
                F(x,y,z) = 1;
            end
            
        end
    end
    
    waitbar( z/Nz, h );
    
end

close(h);

end