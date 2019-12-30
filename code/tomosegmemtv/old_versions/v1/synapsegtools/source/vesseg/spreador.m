%% Spread orientation
% Author: Antonio Martinez Sanchez
% I: Segmented (binary) input tomogram
% V: input normal vectors (unoriented)
% p: initial point
% n: initial normal vector (should have 1 norm)
% N: output normal vectors
function [N,H] = spreador( I, V, p, n )

%% Initialization
[Nx,Ny,Nz] = size( I );
N = zeros( Nx, Ny, Nz, 3 );
H = zeros( Nx, Ny, Nz );
mn = sqrt( sum(n.*n) ); 
N(p(1),p(2),p(3),:) = n;
H(p(1),p(2),p(3),:) = 1;
w = zeros( 3, 26 );
sm = pi / 4; % Maximum angle between adjacents vectors

%% Flood-fill (explicit) based algorithm

% Add stating point
import java.util.LinkedList
q = LinkedList();
q.add( p );
% Loop
while q.size() > 0
    p = q.getLast();
    q.removeLast();
    x = p(1);
    y = p(2);
    z = p(3);
    nh = reshape( N(x,y,z,:), 3, 1); 
    % Look on the neighbourhood
    w(:,1) = [x-1; y-1; z-1];
    w(:,2) = [x-1; y-1; z];
    w(:,3) = [x-1; y-1; z+1];
    w(:,4) = [x-1; y; z-1];
    w(:,5) = [x-1; y; z];
    w(:,6) = [x-1; y; z+1];
    w(:,7) = [x-1; y+1; z-1];
    w(:,8) = [x-1; y+1; z];
    w(:,9) = [x-1; y+1; z+1];
    w(:,10) = [x; y-1; z-1];
    w(:,11) = [x; y-1; z];
    w(:,12) = [x; y-1; z+1];
    w(:,13) = [x; y; z-1];
    w(:,14) = [x; y; z+1];
    w(:,15) = [x; y+1; z-1];
    w(:,16) = [x; y+1; z];
    w(:,17) = [x; y+1; z+1];
    w(:,18) = [x+1; y-1; z-1];
    w(:,19) = [x+1; y-1; z];
    w(:,20) = [x+1; y-1; z+1];
    w(:,21) = [x+1; y; z-1];
    w(:,22) = [x+1; y; z];
    w(:,23) = [x+1; y; z+1];
    w(:,24) = [x+1; y+1; z-1];
    w(:,25) = [x+1; y+1; z];
    w(:,26) = [x+1; y+1; z+1];
    for k = 1:26
        xh = w(1,k);
        yh = w(2,k);
        zh = w(3,k);
        wh = [xh; yh; zh];
        cond1 = logical( sum( wh<1 ) );
        cond2 = logical( sum( [xh>Nx; yh>Ny; zh>Nz] ) );
        if (~cond1) && (~cond2)
            if (I(xh,yh,zh)==1) && (H(xh,yh,zh)==0)
% if ((xh==229) && (yh==122) && (zh==31)) || ((xh==231) && (yh==117) && (zh==31))
%     jol = 1;
% end
                % Get orientation and add a new point to the queue
                cn = reshape(V(xh,yh,zh,:),3,1);
                md = 1 / (mn * sqrt(sum(cn.*cn)));
                O1 = abs( acos( dot(cn,nh) * md ) );
% Hd(xh,yh,zh) = acos( dot(cn,nh) / md );                
                O2 = abs( acos( dot(-cn,nh) * md ) );
                if O2 < O1
                    if O2 < sm
                        cn = -cn;
                        H(xh,yh,zh) = 1;
                        N(xh,yh,zh,:) = cn;
                        cp = [xh; yh; zh];
                        q.add( cp );
                    end
                else
                    if O1 < sm
                        H(xh,yh,zh) = 1;
                        N(xh,yh,zh,:) = cn;
                        cp = [xh; yh; zh];
                        q.add( cp );
                    end
                end
                
            end
        end
    end
    
end

end