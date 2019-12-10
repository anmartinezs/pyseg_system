function S = steer2dtvoting( Is, Io, W0, W2, W4, W6, W8 )
% STEER2DTVOTING  Tensor voting for 2D image by using steerable filters.
%   INPUT:  
%       Is - Input image stickness
%       Io - Input orientation 
%       Wi - Basis filter i in Fourier space
%   OUTPUT:
%       S - Output saliency
%
%   See also: dtvoting, dtvotinge
%   
%   AUTHOR: Antonio Martinez-Sanchez
%   REFERENCES:
%       [1] Martinez-Sanchez A., et al. Robust membrane detection based on tensor voting 
%       for electron tomography. J Struct Biol. 186 (2014) 49-61.

%% Complex features
F0 = fft2( Is );
F2 = fft2( Is .* exp( -1i*2*Io ) );
F2c = fft2( Is .* exp( 1i*2*Io ) );
F4 = fft2( Is .* exp( -1i*4*Io ) );
F6 = fft2( Is .* exp( -1i*6*Io ) );

%% Convolutions
C1 = ifft2( W0.*F2c );
C2 = ifft2( W2.*F0 );
C3 = ifft2( W4.*F2 );
C4 = ifft2( W6.*F4 );
C5 = ifft2( W8.*F6 );

%% Tensor components
U2 = fftshift( C1 + 4*C2 + 6*C3 + 4*C4 + C5 );

%% Tensor descriptors
S = abs( U2 );

end