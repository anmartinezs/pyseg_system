%% Vesicles predetection filter based on template matching for GUI vesseg
% Author: Antonio Martinez-Sanchez
% I: input tomogram
% M: input membrane mask tomogram
% C: input citosol mask tomogram
% rm: minimum radius
% rM: maximum radius
% s: membrane thickness
% wa: angle for missing wedge
% wr: rotational angle for missing wedge
% V: output center vesicles likelyhood (range [255 0])
% R: radius of the vesicles
function [V,R] = vesdettmg( I, M, C, rm, rM, s, wa, wr )

%% Membrane masking (delete predetected membranes)
h = waitbar( 0, 'Vesicles filter...' );
M = growtom( M, 2, 2 );
V = I .* ~(M>0);
waitbar( 1/4, h );

%% Multiescale template matching
[V,R] = ves3dtm( V, [rm,rM], .2, s, wa, wr );
V = V .* ~(M>0) .* C;
waitbar( 1/4, h );

%% Blob detection
V = blobdetec( V, 1, 4 );
V = -V .* (V<0);
waitbar( 1/4, h );

%% Histogram equalization and normalization
V = histeq3dblob( V, C, 255, 1 );
waitbar( 1/4, h );
close( h );

end