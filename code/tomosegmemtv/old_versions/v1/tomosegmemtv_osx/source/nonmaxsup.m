function B = nonmaxsup( I, M, Vx, Vy, Vz )
% NONMAXSUP  Ridge centreline detection by non-maximum suppresion criteria
%   INPUT:  
%       I: Input tomogram, input data must be single or double 
%       M: Binary (logical format) mask for improve the speed 
%       Vi: The i coordinate of the major eigenvector
%   OUTPUT:
%       B: binary output
%
%   See also: eig3dkmex, surfaceness, getsaliency
%   
%   AUTHOR: Antonio Martinez-Sanchez (an.martinez.s.sw@gmail.com)
%   REFERENCES:
%       [1] Martinez-Sanchez A., et al. Robust membrane detection based on tensor voting 
%       for electron tomography. sent to J Struct Biol. (2013)

% Format data for calling the stub
[Nx,Ny,Nz] = size( I );
H = zeros( Nx, Ny, Nz );
H(2:Nx-1,2:Ny-1,2:Nz-1) = 1;
M = M .* H;
clear H;
m = Nx * Ny * Nz;
Mr = 0:m-1;
M = logical( reshape( M, m, 1 ) );
Mr = Mr(M);

% Call to stub
Ir = reshape( I, m, 1 );
Vxr = reshape( Vx, m, 1 );
Vyr = reshape( Vy, m, 1 );
Vzr = reshape( Vz, m, 1 );
Mr = uint64( Mr );
if isa(I,'double')
    Br = nonmaxsup_kernel( Ir, Vxr, Vyr, Vzr, Mr, uint32([Nx;Ny]) );
elseif isa(I,'single')
    Br = nonmaxsup_kernel_single( Ir, Vxr, Vyr, Vzr, Mr, uint32([Nx;Ny]) );
end
clear Vxr;
clear Vyr;
clear Vzr;
B = reshape( Br, Nx, Ny, Nz ); 

end