% Tomogram (3D) equalization for blob detection
% Author: Antonio Martinez-Sanchez
% T: input tomogram
% M: binary mask for input data
% n: number of bins for the histogram
% mode: 1-> vesicles center enhancement, otherwise-> usual
% E: output histogram (with constrast enhanced) with range [0,n];
function E = histeq3dblob( T, M, n, mode )

% Histogram
M = logical( (T>0).*M );
D = T(M);

% Adjusting parameters
if mode == 1
    [H,B] = hist( log(D), n );
else
    [H,B] = hist( D, n );
end
[~,id] = max( H );
[p,c] = linmap( B(id), B(n), n, 0 );

% Adjusting
E = log(T)*p + c;
E = round( E );
E(E<0) = 0;
E(E>n-1) = n-1;
E = E + 1;

end