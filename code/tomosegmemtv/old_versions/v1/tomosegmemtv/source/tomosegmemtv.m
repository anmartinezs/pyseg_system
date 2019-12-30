function [F,Vx,Vy,Vz] = tomosegmemtv( varargin )
% TOMOSEGMEMTV Centreline detection of plane ridges, membranes (Mac OS version 0.2)
%   INPUT:  
%       I - Input tomogram (foreground dark), all of its dimension must be greater than 3 
%       s - Scale factor 
%       t - Membrane thickness factor
%       v - (optional) If equal to 1 verbose mode activated (disabled by default)
%       m - (optional) If equal to 1 ridge detection (default), otherwise edge detection
%       w - (optional) Missing wedge semiangle in Z axis, it tries to prevent missing wedge effects, 
%       if less than 0 is disabled (default)
%       e - (optional) Mode for resolving eigenproblem; 1- Fast (default), 2- Intermediate,
%       otherwise- Accurate
%       d - (optional) Densification scale factor, by default is 10
%       u - (optional) Memory reduction coeficient if it is less than 1 the input data may be
%       divided in blocks.
%   OUTPUT:
%       If the input format for data are "double" the output will have this format, otherwise
%       will have "single" format
%       F - filtered tomogram with the membranes enhanced
%       Vi - Coordinate i of the normal to membrane
%
%   See also: global_analysis, cropt
%   
%   AUTHOR: Antonio Martinez-Sanchez (an.martinez.s.sw@gmail.com)
%   REFERENCES:
%       [1] Martinez-Sanchez A., et al. Robust membrane detection based on tensor voting 
%       for electron tomography. sent to J Struct Biol. (2013)

%% Initialization
mg = 17;
sf = 1.5;

%% Read optional input paramers
numvarargs = length(varargin);
if numvarargs < 3
    error( 'tomosegmemtv.m: not enough input arguments (see help)' );
elseif numvarargs == 3
    I = cell2mat( varargin(1) );
    s = cell2mat( varargin(2) );
    t = cell2mat( varargin(3) );
    v = 0;
    m = 1;
    w = 0;
    e = 1;
    d = s;
    u = 100;
elseif numvarargs == 4
    I = cell2mat( varargin(1) );
    s = cell2mat( varargin(2) );
    t = cell2mat( varargin(3) );
    v = cell2mat( varargin(4) );
    m = 1;
    w = 0;
    e = 1;
    d = s;
    u = 100;
elseif numvarargs == 5
    I = cell2mat( varargin(1) );
    s = cell2mat( varargin(2) );
    t = cell2mat( varargin(3) );
    v = cell2mat( varargin(4) );
    m = cell2mat( varargin(5) );
    w = 0;
    e = 1;
    d = s;
    u = 100;
elseif numvarargs == 6
    I = cell2mat( varargin(1) );
    s = cell2mat( varargin(2) );
    t = cell2mat( varargin(3) );
    v = cell2mat( varargin(4) );
    m = cell2mat( varargin(5) );
    w = cell2mat( varargin(6) );
    e = 1;
    d = s;
    u = 100;
elseif numvarargs == 7
    I = cell2mat( varargin(1) );
    s = cell2mat( varargin(2) );
    t = cell2mat( varargin(3) );
    v = cell2mat( varargin(5) );
    m = cell2mat( varargin(5) );
    w = cell2mat( varargin(6) );
    e = cell2mat( varargin(7) );
    d = s;
    u = 100;
elseif numvarargs == 8
    I = cell2mat( varargin(1) );
    s = cell2mat( varargin(2) );
    t = cell2mat( varargin(3) );
    v = cell2mat( varargin(5) );
    m = cell2mat( varargin(5) );
    w = cell2mat( varargin(6) );
    e = cell2mat( varargin(7) );
    d = cell2mat( varargin(8) );  
    u = 100;
elseif numvarargs == 9
    I = cell2mat( varargin(1) );
    s = cell2mat( varargin(2) );
    t = cell2mat( varargin(3) );
    v = cell2mat( varargin(4) );
    m = cell2mat( varargin(5) );
    w = cell2mat( varargin(6) );
    e = cell2mat( varargin(7) );
    d = cell2mat( varargin(8) );  
    u = cell2mat( varargin(9) );
elseif numvarargs > 9
    error( 'tomosegmemtv.m: requires at most 9 optional inputs (see help)' );
end

if ~isfloat(I)
    I = single( I );
end
if isa(I,'double')
    ds = 8 * numel(I);
else
    ds = 4 * numel(I);
end

%% Divide tomogram in blocks (if necessary) to prevent memory overload
uh = .01 * u;
if (uh<0) || (uh>1)
    fprintf( 1, 'tomosegmemtv.m: warning: Parameter u=%d, it is set to 100.\n', u );
end
[~,ms] = getsysinfo_linux();
sl = ceil( (ds*mg)/double(uh*ms) );
sh = floor( sf*s );
[Nx,Ny,Nz] = size( I );
N = ceil( Nz / sl );

% Ensure that all dimension are even
nxodd = mod(Nx,2);
nyodd = mod(Ny,2);
nzodd = mod(Nz,2);
Nx = Nx - nxodd;
Ny = Ny - nyodd;
Nz = Nz - nzodd;
I = I(1:Nx,1:Ny,1:Nz);

if sl == 1 % Blocks division is not necessary
    
    % Call to membfilter stub
    [Fc,Vxc,Vyc,Vzc] = membflt_kernel( I, s, t, v, m, w, e, d );

else
    
    if isa(I,'double')
        Fc = double( zeros(Nx,Ny,Nz) );
        Vxc = double( zeros(Nx,Ny,Nz) );
        Vyc = double( zeros(Nx,Ny,Nz) );
        Vzc = double( zeros(Nx,Ny,Nz) );
    else
        Fc = single( zeros(Nx,Ny,Nz) );
        Vxc = single( zeros(Nx,Ny,Nz) );
        Vyc = single( zeros(Nx,Ny,Nz) );
        Vzc = single( zeros(Nx,Ny,Nz) );
    end
    lid = 1;
    lidc = 1;
    uid = N + sh;
    uidc = N;
    if uid > Nz
        uid = Nz;
    end
    for k = 1:sl
        if v == 1
            fprintf( 1, 'BLOCK %d/%d:\n', k, sl );
        end
        % Call to membfilter stub 
        [Fh,Vxh,Vyh,Vzh] = membflt_kernel( I(:,:,lid:uid), s, t, v, m, w, e, d );
        offl = lidc - lid + 1;
        offh = uidc - lidc + offl;
        Fc(:,:,lidc:uidc) = Fh(:,:,offl:offh);
        Vxc(:,:,lidc:uidc) = Vxh(:,:,offl:offh);
        Vyc(:,:,lidc:uidc) = Vyh(:,:,offl:offh);
        Vzc(:,:,lidc:uidc) = Vzh(:,:,offl:offh);
        clear Fh;
        clear Vxh;
        clear Vyh;
        clear Vzh;
        lid = uidc - sh;
        lidc = uidc;
        uidc = uidc + N;
        if lid < 1
            lid = 1;
        end
        if uidc > Nz
           uidc = Nz;
           uid = Nz;
        else
           uid = uidc + sh; 
           if uid > Nz
               uid = Nz;
           end
        end
    end
    
end

% Get back to original dimensions
if isa(I,'double')
    F = double( zeros(Nx+nxodd,Ny+nyodd,Nz+nzodd) );
else
    F = single( zeros(Nx+nxodd,Ny+nyodd,Nz+nzodd) );
end
F(1:Nx,1:Ny,1:Nz) = Fc;
clear Fc;
if isa(I,'double')
    Vx = double( zeros(Nx+nxodd,Ny+nyodd,Nz+nzodd) );
else
    Vx = single( zeros(Nx+nxodd,Ny+nyodd,Nz+nzodd) );
end
Vx(1:Nx,1:Ny,1:Nz) = Vxc;
clear Vxc;
if isa(I,'double')
    Vy = double( zeros(Nx+nxodd,Ny+nyodd,Nz+nzodd) );
else
    Vy = single( zeros(Nx+nxodd,Ny+nyodd,Nz+nzodd) );
end
Vy(1:Nx,1:Ny,1:Nz) = Vyc;
clear Vyc;
if isa(I,'double')
    Vz = double( zeros(Nx+nxodd,Ny+nyodd,Nz+nzodd) );
else
    Vz = single( zeros(Nx+nxodd,Ny+nyodd,Nz+nzodd) );
end
Vz(1:Nx,1:Ny,1:Nz) = Vzc;
clear Vzc;
    
end