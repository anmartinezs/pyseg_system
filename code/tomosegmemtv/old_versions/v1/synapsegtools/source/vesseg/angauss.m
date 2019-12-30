%% Anisotropic gaussian filtering
% Author: Antonio Martinez-Sanchez
% I: input tomogram
% s: scale 
% r: aspect ratio
% F: filtered output
function F = angauss( I, s, r )

%% Intialization
[Nx,Ny,Nz] = size( I );
if mod(Nx,2)
    Nx2 = floor( .5*Nx );
    Vnx = -Nx2:Nx2;
else
    Nx2 = .5*Nx;
    Vnx = -Nx2:Nx2-1;
end
if mod(Ny,2)
    Ny2 = floor( .5*Ny );
    Vny = -Ny2:Ny2;
else
    Ny2 = .5*Ny;
    Vny = -Ny2:Ny2-1;
end
if mod(Nz,2)
    Nz2 = floor( .5*Nz );
    Vnz = -Nz2:Nz2;
else
    Nz2 = .5*Nz;
    Vnz = -Nz2:Nz2-1;
end
[X,Y,Z] = meshgrid( Vny, Vnx, Vnz );
A = 1 / ((s*s*sqrt(r*(2*pi)^3)));
a = 1 / (2*s*s);
b = a / r;

%% Kernel
K = A * exp( -a*(X.*X+Y.*Y)-b*Z.*Z ); 

%% Convolution in Fourier domain
F = fftshift( ifftn(fftn(I).*fftn(K)) );

end