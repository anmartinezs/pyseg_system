%% Get an interplated stack form an oriented membrane (Modified version for caulobacter)
%% It take sample at booth sides and keep the side with greater total density
% Author: Antonio Martinez-Sanchez
% I: original intput tomogram
% M: input binary tormogram with membranes segmented
% Ni: input oriented normals coordinate i
% s: precision step
% t: thickness
% S: output stack [x y z s1 s2 s3 ... s4] where si is the i-sample
function S = membtostack_caulo2( I, M, N1, N2, N3, s, t )

%% Initialization
[Nx,Ny,Nz] = size( I );
off = ceil( t );
K = s * (0:floor(t/s));
lk = length( K );
Kk = 1:lk;
np = sum(M>0);
lk3 = lk+3;
S = zeros( np, lk3 );
Sh1 = zeros( lk );
Sh2 = zeros( lk );
cont = 1;

%% Curvature

%% Tomogram loop
wx = off+1:Nx-off;
wy = off+1:Ny-off;
wz = off+1:Nz-off;
for z = wz
    for y = wy
        for x = wx
           
            if M(x,y,z)
                
                % Get original point and vector
                v = [N1(x,y,z); N2(x,y,z); N3(x,y,z)];                    
                p = [x; y; z];
                
                % Take samples along vector direction
                for k = Kk
                    pv = p + K(k)*v;
                    vst = floor( pv );
                    vs = pv - vst;   
                    W = I(vst(1)-1:vst(1)+1,vst(2)-1:vst(2)+1,vst(3)-1:vst(3)+1);
                    Sh1(k) = trilin3d( W, vs );
                    S(cont,k+3) = trilin3d( W, vs );
                end
                
                % Changing sample direction
                v = -v;
                
                % Take samples along vector direction
                for k = Kk
                    pv = p + K(k)*v;
                    vst = floor( pv );
                    vs = pv - vst;   
                    W = I(vst(1)-1:vst(1)+1,vst(2)-1:vst(2)+1,vst(3)-1:vst(3)+1);
                    Sh2(k) = trilin3d( W, vs );
                end
                
                % Deciding which side is kept
                if sum(Sh1) > sum(Sh2)
                    S(cont,3:lk3) = Sh1;
                else
                    S(cont,3:lk3) = Sh2;
                end
                
                S(cont,1) = x;
                S(cont,2) = y;
                S(cont,3) = z;
                cont = cont + 1;
            end
            
        end
    end
end

end