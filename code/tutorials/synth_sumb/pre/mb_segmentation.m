%% Script for segmenting (unsigned) with TomoSegMemTV all tomograms with microsomes
% Author: Antonio Martinez-Sanchez

%% Clear workspace
close all
clear all

%% Add TomoSegMemTV path
addpath '../../../../pyseg-sys/tomosegmemtv/tomosegmemtv/source'
addpath '../../../../pyseg-sys/tomosegmemtv/tomosegmemtv/source/mex'

%% Input/Output directories
in_star = '../../../../data/tutorials/synth_sumb/mics/test_1.star';
dout = '../../../../data/tutorials/synth_sumb/segs';
out_star = '../../../../data/tutorials/synth_sumb/segs/test_1_seg.star';

%% Input parameters
s = 15;  % Scale factor 
t = 3; % Membrane thickness factor
v = 1; % Verbose mode activated
m = 1; % Ridge detection (membranes) -default-
w = -1; % Missing wedge semiangle in Z axis 
e = 1; % Mode for resolving eigenproblem (fast) -default-
d = 15; % Densification scale factor, by default is 10 -default-
u = 20; % Maximum physical memory usage
tb = 15; % binarization threshold
tg = 10000; % Threshold for micosome size in pixels
mb_ht = 2; % Membrane half-thickness in pixels
halo_z = 15; % Slices to discard (to set zero) on top and bottom of the tomogram
max_mb_dst = 30; % Maximum distance to membrane for foreground in the output segmentation (0-background, 1-membrane, 2-foreground)

%% Preparing the input STAR file
fstar = fopen(out_star, 'w');
fprintf(fstar, '\ndata_\n\nloop_\n' );
fprintf(fstar, '_rlnMicrographName\n' );
fprintf(fstar, '_rlnImageName\n' );
fprintf(fstar, '_psSegImage\n' );
fprintf(fstar, '_psSegOffX\n');
fprintf(fstar, '_psSegOffY\n');
fprintf(fstar, '_psSegOffZ\n');
fprintf(fstar, '_psSegRot\n');
fprintf(fstar, '_psSegTilt\n');
fprintf(fstar, '_psSegPsi\n');

%% Loop on input STAR file
% Read input directory
Star = tom_starread(in_star);
[ld, ~] = size( Star );
for k = 1:ld
    file = Star(k).rlnImageName;
    mic = Star(k).rlnMicrographName;
    [~,rname,ext] = fileparts( file );
    % Only process MRC files
    if strcmp(ext,'.mrc') || strcmp(ext,'.rec') || strcmp(ext,'.em')
        % Load input tomogram
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
        fprintf( 'PROCESSING FILE: %s\n', file );
        tic; [F,Vx,Vy,Vz] = tomosegmemtv( T, s, t, v, m, w, e, d, u ); toc;
        
        % Post-processing
        fprintf( '-Post-processing.\n');
        G = global_analysis(F, tb, 6, 0);
        [Nx, Ny, Nz] = size(G);
        Mbu = single(zeros(size(G)));
        for z = halo_z:Nz-halo_z
            BW = bwdist(G(:,:,z) >= tg);
            H = zeros([Nx, Ny]);
            H(BW <= (max_mb_dst+mb_ht)) = 2;
            H(BW <= mb_ht) = 1;
            Mbu(:,:,z) = H;
        end
        clear G

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
        fout = sprintf( '%s/%s_mbu.mrc', dout, rname );
        tom_mrcwrite( Mbu, 'name', fout );
        clear Mbu;
        
        % Create entry in the output STAR file
        fprintf(fstar, '%s %s %s %d %d %d %d %d %d\n', mic, file, fout, 0, 0, 0, 0, 0, 0);
    end
end

fclose(fstar);
fprintf('Output Star file in: %s\n', out_star);
fprintf('Succesfully terminated. (%s)\n', datestr(now));
