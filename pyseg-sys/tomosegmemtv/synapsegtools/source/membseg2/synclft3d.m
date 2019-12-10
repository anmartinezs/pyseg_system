function S = synclft3d( T, N, dm, dM )
% SYNCLFT3D  Segment surfaces that have the closest surface along normal (N) direction is in 
%           the range [dm,dM]
%   INPUT:  
%       T - Input binary tomogram
%       N - Normal vector 3D field
%       dm - minimum distance
%       dM - maximum distance
%   OUTPUT:
%       S - Output with the surfaces segmented
%
%   See also: memblabel
%   
%   AUTHOR: Antonio Martinez-Sanchez (an.martinez.s.sw@gmail.com)
%   REFERENCES:
%       [1] Martinez-Sanchez, A., et al. A differential structure approach to membrane segmentation 
%       in electron tomography. J. Struct. Biol. (2011), doi:10.1016/j.jsb.2011.05.010
%       [2] Martinez-Sanchez, A., et al. A ridge-based framework for segmentation of 3D electron 
%       microscopy datasets. J. Struct. Biol. (2012), http://dx.doi.org/10.1016/j.jsb.2012.10.002

%% Initialization
s = .71;
[Nx,Ny,Nz] = size( T );
S = zeros( Nx, Ny, Nz );

h = waitbar( 0, 'Measuring distance between surfaces...' );

%% Tomogram loop
for z = 1:Nz
    for y = 1:Ny
        for x = 1:Nx
           
            if T(x,y,z)
                
                p = [x; y; z];
                v = [N(x,y,z,1); N(x,y,z,2); N(x,y,z,3)];
                lock = 1;
                lock2 = 1;
                n = 0;
                
                % Forward step
                while lock
                    n = n + s;
                    if n <= dM
                        pn = round( p + n*v );
                        if (pn(1)>1) && (pn(1)<Nx) && (pn(2)>1) && (pn(2)<Ny) && (pn(3)>1) && (pn(3)<Nz)
                            W = T(pn(1)-1:pn(1)+1,pn(2)-1:pn(2)+1,pn(3)-1:pn(3)+1);
                            st = sum(sum(W));
                            if st == 0 % Leave home membrane 
                                lock = 0; 
                                while lock2 % Loop for look the neighbour membrane
                                  n = n + s;
                                  if n <= dM
                                      pn = round( p + n*v );
                                      if (pn(1)>1) && (pn(1)<Nx) && (pn(2)>1) && (pn(2)<Ny) && (pn(3)>1) && (pn(3)<Nz)
                                        W = T(pn(1)-1:pn(1)+1,pn(2)-1:pn(2)+1,pn(3)-1:pn(3)+1);
                                        st = sum(sum(W));
                                        if st > 0 
                                            if n >= dm % Reach neighbour membrane
                                                S(x,y,z) = n;
                                                lock2 = 0;
                                            else
                                                lock2 = 0; % The first neighbour is not in the serch region
                                            end
                                        end
                                      else
                                          lock2 = 0; % Reach tomogram boundary
                                      end
                                  else
                                      lock2 = 0; % Reach the end of the search region                                   
                                  end
                                end
                            end
                        else
                            lock = 0; % Reach tomogram boundary
                        end
                    else
                        lock = 0; % Reach the end of the search region
                    end
                end
                
                % Backward step
                lock = 1;
                lock2 = 1;
                n = 0;
                if S(x,y,z)==0
                    while lock
                        n = n + s;
                        if n <= dM
                            pn = round( p - n*v );
                            if (pn(1)>1) && (pn(1)<Nx) && (pn(2)>1) && (pn(2)<Ny) && (pn(3)>1) && (pn(3)<Nz)
                                W = T(pn(1)-1:pn(1)+1,pn(2)-1:pn(2)+1,pn(3)-1:pn(3)+1);
                                st = sum(sum(W));
                                if st == 0 % Leave home membrane 
                                    lock = 0; 
                                    while lock2 % Loop for look the neighbour membrane
                                      n = n + s;
                                      if n <= dM
                                          pn = round( p - n*v );
                                          if (pn(1)>1) && (pn(1)<Nx) && (pn(2)>1) && (pn(2)<Ny) && (pn(3)>1) && (pn(3)<Nz)
                                            W = T(pn(1)-1:pn(1)+1,pn(2)-1:pn(2)+1,pn(3)-1:pn(3)+1);
                                            st = sum(sum(W));
                                            if st > 0 
                                                if n >= dm % Reach neighbour membrane
                                                    S(x,y,z) = n;
                                                    lock2 = 0;
                                                else
                                                    lock2 = 0; % The first neighbour is not in the serch region
                                                end
                                            end
                                          else
                                              lock2 = 0; % Reach tomogram boundary
                                          end
                                      else
                                          lock2 = 0; % Reach the end of the search region                                   
                                      end
                                    end
                                end
                            else
                                lock = 0; % Reach tomogram boundary
                            end
                        else
                            lock = 0; % Reach the end of the search region
                        end
                    end
                end
                
            end
            
        end
    end
    
    waitbar( z/Nz, h );
    
end

S = clrpartf2( S, 26 );
close( h );

end