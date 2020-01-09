%% Script for processing all tomograms in a directory with tomosegmemtv
% Author: Antonio Martinez-Sanchez

%% Clear workspace
close all
clear all

%% Input/Output directories
din = '/path/to/input/tomograms';
dout = '/path/to/output/tomogrmas';

%% Input parameters
s = 15;  % Scale factor 
t = 1; % Membrane thickness factor
v = 1; % Verbose mode activated
m = 1; % Ridge detection (membranes) -default-
w = -1; % Missing wedge semiangle in Z axis 
e = 1; % Mode for resolving eigenproblem (fast) -default-
d = 15; % Densification scale factor, by default is 10 -default-
u = 20; % Maximum physical memory usage

%% Loop on directory files
% Read input directory
D = dir( din );
ld = length( D );
for k = 1:ld
    [~,rname,ext] = fileparts( D(k).name );
    % Only process MRC files
    if strcmp(ext,'.mrc') || strcmp(ext,'.rec') || strcmp(ext,'.em')
        % Load input tomogram
        file = sprintf( '%s/%s', din, D(k).name );
        % T = single( readmrc(file,1024) );
        % Sometimes 'tom_mrcread' fails and I don't know why
        if strcmp(ext,'.em')
            T = tom_emread( file );
        else
            T = tom_mrcread( file );
        end
        T = single( T.Value );
        % clear ans;

        % Processing
        fprintf( 'PROCESSING FILE: %s\n', D(k).name );
        tic; [F,Vx,Vy,Vz] = tomosegmemtv( T, s, t, v, m, w, e, d, u ); toc;

        % Save result
        fout = sprintf( '%s/%s.mrc', dout, rname );
        tom_mrcwrite( T, 'name', fout );
        clear T;
        fout = sprintf( '%s/%s_flt.mrc', dout, rname );
        tom_mrcwrite( F, 'name', fout );
        clear F;
        fout = sprintf( '%s/%s_n1.mrc', dout, rname );
        tom_mrcwrite( Vx, 'name', fout );
        clear Vx;
        fout = sprintf( '%s/%s_n2.mrc', dout, rname );
        tom_mrcwrite( Vy, 'name', fout );
        clear Vy;
        fout = sprintf( '%s/%s_n3.mrc', dout, rname );
        tom_mrcwrite( Vz, 'name', fout );
        clear Vz;
    end
end
