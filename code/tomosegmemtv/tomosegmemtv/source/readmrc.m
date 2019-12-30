% Read a MRC file (3D image)
% Author: Antonio Martínez Sánchez
% head: head size (1024 or 16)
function [a] = readmrc( fname, head )

[fid,message]=fopen(fname,'r');
if fid == -1
    error('can''t open file');
    a= -1;
    return;
end
nx=fread(fid,1,'long');
ny=fread(fid,1,'long');
nz=fread(fid,1,'long');
type= fread(fid,1,'long');
fprintf(1,'nx= %d ny= %d nz= %d type= %d\n', nx, ny,nz,type);
% Seek to start of data
status=fseek(fid,head,-1);
% Bytes
if type == 0
    a = fread( fid, nx*ny*nz, 'uint8' );
end
% Shorts
if type== 1
    a=fread(fid,nx*ny*nz,'int16');
end
%floats
if type == 2
    a=fread(fid,nx*ny*nz,'float32');
end
fclose( fid);
a= reshape(a, [nx ny nz]);
if nz == 1
    a= reshape(a, [nx ny]);
end