% Clear particles smaller than a threshold (fast version with graphic bar)
% Author: Antonio Martinez-Sanchez
% M: binary tomogram
% c: 3D connectivity => 6, 18 or 26 
% C: cleared tomogram
function C = clrpartf2( M, c )

[Nx Ny Nz] = size( M );
[L K] = bwlabeln( M, c );
C = zeros( Nx, Ny, Nz );
Hk = zeros( K, 1 );

h = waitbar( 0, 'Size measuring and labeling...' );
mx = 2*Nz;

%% Count loop
for z = 1:Nz
    for y = 1:Ny
        for x = 1:Nx             
            id = L(x,y,z);
            if id > 0
                Hk(id) = Hk(id) + 1;
            end
        end
    end
    waitbar( z/mx, h );
end

%% Write loop
for z = 1:Nz
    for y = 1:Ny
        for x = 1:Nx            
            id = L(x,y,z);
            if id > 0
                C(x,y,z) = Hk(id);
            end
        end
    end
    waitbar( (Nz+z)/mx, h );
end

close( h );
