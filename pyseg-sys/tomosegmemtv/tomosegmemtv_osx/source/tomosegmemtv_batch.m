%% Script for processing all tomograms in a directory with membflt
% Author: Antonio Martinez-Sanchez

%% Clear workspace
close all
clear all

%% Input/Output directories
din = '~/Desktop/test/in';
dout = '~/Desktop/test/out_s7_t2';

%% Input parameters
s = 7;  % Scale factor 
t = 2; % Membrane thickness factor
v = 1; % Verbose mode activated
m = 1; % Ridge detection (membranes) -default-
w = -1; % Missing wedge semiangle in Z axis 
e = 1; % Mode for resolving eigenproblem (fast) -default-
d = 3; % Densification scale factor, by default is 10 -default-
u = 60; % Maximum physical memory usage

%% Loop on directory files
% Read input directory
D = dir( din );
ld = length( D );
for k = 1:ld
    [~,rname,ext] = fileparts( D(k).name );
    % Only process MRC files
    if strcmp(ext,'.mrc') || strcmp(ext,'.rec')
        % Load input tomogram
        file = sprintf( '%s/%s', din, D(k).name );
        T = tom_mrcread( file );
        T = single( T.Value );
        clear ans;

        % Processing
        fprintf( 'PROCESSING FILE: %s\n', D(k).name );
        tic; [F,Vx,Vy,Vz] = tomosegmemtv( T, s, t, v, m, w, e, d, u ); toc;
        clear T;
        % Save result
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