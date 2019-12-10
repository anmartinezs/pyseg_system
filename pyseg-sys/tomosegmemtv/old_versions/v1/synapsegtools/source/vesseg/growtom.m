%% Growing a binary tomogram
% I: input data
% t: number of iterations
% c: connectivity (1->6, otherwise->26)
% G: output tomogram
function G = growtom( I, t, c )

%% Initialization
if c == 1
    N(:,:,1) = [0 0 0; 0 1 0; 0 0 0];
    N(:,:,2) = [0 1 0; 1 1 1; 0 1 0];
    N(:,:,3) = [0 0 0; 0 1 0; 0 0 0];
else
    N = ones( 3, 3, 3 );
end

G = I;
for k = 1:t
    G = imdilate( G, N );
end

end