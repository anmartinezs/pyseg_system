function [Data] = tom_mrcread_tmp(varargin)
%TOM_MRCREAD reads MRC format file
%
%   tom_mrcread(varargin)
%
%   i=tom_mrcread
%   i=tom_mrcread('Proj.mrc');
%   i=tom_mrcread('Proj.mrc',Endian);
%
%   Reads a 2D or 3D MRC format file. A MRC format contains a 1024 Bytes
%   header and the raw data. If there is no input then a dialog box appears to
%   select a file. Endian is an option, 'le' for little-endian (PC), 'be' for
%   big-endian (SGI,MAC)
%
%PARAMETERS
%
%  INPUT
%   filelabel           ...
%   threshold           ...
%   label               ...
%   color               ...
%   transformmatrix     ...
%   iconposition        ...
%   host                ...
%  
%  OUTPUT
%   data		...
%
%Structure of MRC-data files:
%MRC Header has a length of 1024 bytes
% SIZE  DATA    NAME    DESCRIPTION
%   4   int     NX      number of Columns    (fastest changing in map)
%   4   int     NY      number of Rows
%   4   int     NZ      number of Sections   (slowest changing in map)
%   4   int     MODE    Types of pixel in image
%                       0 = Image     unsigned bytes
%                       1 = Image     signed short integer (16 bits)
%                       2 = Image     float
%                       3 = Complex   short*2
%                       4 = Complex   float*2     
%	4   int     NXSTART Number of first COLUMN  in map (Default = 0)
%   4   int     NYSTART Number of first ROW     in map      "
%   4   int     NZSTART Number of first SECTION in map      "
%   4   int     MX      Number of intervals along X
%   4   int     MY      Number of intervals along Y
%   4   int     MZ      Number of intervals along Z
%   4   float   Xlen    Cell Dimensions (Angstroms)
%   4   float   Ylen                 "
%   4   float   Zlen                 "
%   4   float   ALPHA   Cell Angles (Degrees)
%   4   float   BETA                 "
%   4   float   GAMMA                "
%   4   int     MAPC    Which axis corresponds to Columns  (1,2,3 for X,Y,Z)
%   4   int     MAPR    Which axis corresponds to Rows     (1,2,3 for X,Y,Z)
%   4   int     MAPS    Which axis corresponds to Sections (1,2,3 for X,Y,Z)
%   4   float   AMIN    Minimum density value
%   4   float   AMAX    Maximum density value
%   4   float   AMEAN   Mean    density value    (Average)
%   2   short   ISPG    Space group number       (0 for images)
%   2   short   NSYMBT  Number of bytes used for storing symmetry operators
%   4   int     NEXT    Number of bytes in extended header
%   2   short   CREATID Creator ID
%   30    -     EXTRA   Not used. All set to zero by default
%   2   short   NINT    Number of integer per section
%   2   short   NREAL   Number of reals per section
%   28    -     EXTRA2  Not used. All set to zero by default
%   2   short   IDTYPE  0=mono, 1=tilt, 2=tilts, 3=lina, 4=lins
%   2   short   LENS    
%   2   short   ND1   
%   2   short   ND2
%   2   short   VD1 
%   2   short   VD2
%   24  float   TILTANGLES
%   4   float   XORIGIN X origin
%   4   float   YORIGIN Y origin
%   4   float   ZORIGIN Z origin
%   4   char    CMAP    Contains "MAP "
%   4   char    STAMP   
%   4   float   RMS 
%   4   int     NLABL   Number of labels being used
%   800 char    10 labels of 80 character
%
%Extended Header (FEI format and IMOD format)
%The extended header contains the information about a maximum of 1024 images. 
%Each section is 128 bytes long. The extended header is thus 1024 * 128 bytes 
%(always the same length, regardless of how many images are present
%   4   float   a_tilt  Alpha tilt (deg)
%   4   float   b_tilt  Beta tilt (deg)
%   4   float   x_stage  Stage x position (Unit=m. But if value>1, unit=mue)
%   4   float   y_stage  Stage y position (Unit=m. But if value>1, unit=mue)
%   4   float   z_stage  Stage z position (Unit=m. But if value>1, unit=mue)
%   4   float   x_shift  Image shift x (Unit=m. But if value>1, unit=mue)
%   4   float   y_shift  Image shift y (Unit=m. But if value>1, unit=mue)
%   4   float   z_shift  Image shift z (Unit=m. But if value>1, unit=mue)
%   4   float   defocus  Defocus Unit=m. But if value>1, unit=mue)
%   4   float   exp_time Exposure time (s)
%   4   float   mean_int Mean value of image
%   4   float   tilt_axis   Tilt axis (deg)
%   4   float   pixel_size  Pixel size of image (m)
%   4   float   magnification   Magnification used
%   4   float   remainder   Not used (filling up to 128 bytes)   
%
%EXAMPLE
%   A fileselect-box appears and the EM-file can be picked
%   i=tom_mrcread
%
%   %read only the header
%    h=tom_mrcread('test.mrc','',1);
%
%REFERENCES
%
%SEE ALSO
%   TOM_EMREAD, TOM_SPIDERREAD, TOM_ISMRCFILE, TOM_MRCWRITE, TOM_MRC2EM
%
%   created by SN 09/25/02
%   last change: 06/19/2013 FF: added MRC.mode==6
%
%   Nickell et al., 'TOM software toolbox: acquisition and analysis for electron tomography',
%   Journal of Structural Biology, 149 (2005), 227-234.
%
%   Copyright (c) 2004-2013
%   TOM toolbox for Electron Tomography
%   Max-Planck-Institute of Biochemistry
%   Dept. Molecular Structural Biology
%   82152 Martinsried, Germany
%   http://www.biochem.mpg.de/tom


%error(nargchk(0,2,nargin))
[comp_typ,maxsize,endian] = computer;
switch endian
    case 'le'
        sysfor='ieee-le';
    case 'L'
        sysfor='ieee-le';
    case 'be'
        sysfor='ieee-be';
    case 'B'
        sysfor='ieee-be';
end

if nargin <1 
    [filename, pathname] = uigetfile({'*.mrc';'*.rec';'*.ali';'*.*'}, 'Pick an MRC-file');
    if isequal(filename,0) | isequal(pathname,0) 
        disp('No data loaded.'); return; 
    end;
    mrc_name=[pathname filename];
end;
if nargin==1
    mrc_name=varargin{1};
end
if (nargin==2)
    mrc_name=varargin{1};
    sf=varargin{2};
    switch sf
        case 'le'
            sysfor='ieee-le';
        case 'be'
            sysfor='ieee-be';
        otherwise
            error(['Bad argument: ' sf]);
    end
end



if nargin<3
   
    headerOnly=0;
end;

if nargin==3
    mrc_name=varargin{1};
    headerOnly=varargin{3}; 
end;

fid = fopen(mrc_name,'r',sysfor);
if fid==-1
   error(['Cannot open: ' mrc_name ' file']); 
end;
MRC.nx = fread(fid,[1],'int');        %integer: 4 bytes
MRC.ny = fread(fid,[1],'int');        %integer: 4 bytes
MRC.nz = fread(fid,[1],'int');        %integer: 4 bytes
MRC.mode = fread(fid,[1],'int');      %integer: 4 bytes
MRC.nxstart= fread(fid,[1],'int');    %integer: 4 bytes
MRC.nystart= fread(fid,[1],'int');    %integer: 4 bytes
MRC.nzstart= fread(fid,[1],'int');    %integer: 4 bytes
MRC.mx= fread(fid,[1],'int');         %integer: 4 bytes
MRC.my= fread(fid,[1],'int');         %integer: 4 bytes
MRC.mz= fread(fid,[1],'int');         %integer: 4 bytes
MRC.xlen= fread(fid,[1],'float');     %float: 4 bytes
MRC.ylen= fread(fid,[1],'float');     %float: 4 bytes
MRC.zlen= fread(fid,[1],'float');     %float: 4 bytes
MRC.alpha= fread(fid,[1],'float');    %float: 4 bytes
MRC.beta= fread(fid,[1],'float');     %float: 4 bytes
MRC.gamma= fread(fid,[1],'float');    %float: 4 bytes
MRC.mapc= fread(fid,[1],'long');       %integer: 4 bytes
MRC.mapr= fread(fid,[1],'long');       %integer: 4 bytes
MRC.maps= fread(fid,[1],'long');       %integer: 4 bytes
MRC.amin= fread(fid,[1],'float');     %float: 4 bytes
MRC.amax= fread(fid,[1],'float');     %float: 4 bytes
MRC.amean= fread(fid,[1],'float');    %float: 4 bytes
MRC.ispg= fread(fid,[1],'short');     %integer: 2 bytes
MRC.nsymbt = fread(fid,[1],'short');  %integer: 2 bytes
MRC.next = fread(fid,[1],'int');      %integer: 4 bytes
MRC.creatid = fread(fid,[1],'short'); %integer: 2 bytes
MRC.unused1 = fread(fid,[30]);        %not used: 30 bytes
MRC.nint = fread(fid,[1],'short');    %integer: 2 bytes
MRC.nreal = fread(fid,[1],'short');   %integer: 2 bytes
MRC.unused2 = fread(fid,[28]);        %not used: 28 bytes
MRC.idtype= fread(fid,[1],'short');   %integer: 2 bytes
MRC.lens=fread(fid,[1],'short');      %integer: 2 bytes
MRC.nd1=fread(fid,[1],'short');       %integer: 2 bytes
MRC.nd2 = fread(fid,[1],'short');     %integer: 2 bytes
MRC.vd1 = fread(fid,[1],'short');     %integer: 2 bytes
MRC.vd2 = fread(fid,[1],'short');     %integer: 2 bytes
for i=1:6                             %24 bytes in total
    MRC.tiltangles(i)=fread(fid,[1],'float');%float: 4 bytes
end
MRC.xorg = fread(fid,[1],'float');    %float: 4 bytes
MRC.yorg = fread(fid,[1],'float');    %float: 4 bytes
MRC.zorg = fread(fid,[1],'float');    %float: 4 bytes
MRC.cmap = fread(fid,[4],'char');     %Character: 4 bytes
MRC.stamp = fread(fid,[4],'char');    %Character: 4 bytes
MRC.rms=fread(fid,[1],'float');       %float: 4 bytes
MRC.nlabl = fread(fid,[1],'int');     %integer: 4 bytes
MRC.labl = fread(fid,[800],'char');   %Character: 800 bytes

if MRC.mode==0
    beval=MRC.nx*MRC.ny*MRC.nz;
    Data_read = zeros(MRC.nx,MRC.ny,MRC.nz,'int8');
elseif MRC.mode==1
    beval=MRC.nx*MRC.ny*MRC.nz*2;
    Data_read = zeros(MRC.nx,MRC.ny,MRC.nz,'int16');
elseif MRC.mode==2
    beval=MRC.nx*MRC.ny*MRC.nz*4;
    Data_read = zeros(MRC.nx,MRC.ny,MRC.nz,'single');
end
Extended.magnification(1)=0;
Extended.exp_time(1)=0;
Extended.pixelsize(1)=0;
Extended.defocus(1)=0;
Extended.a_tilt(1:MRC.nz)=0;
Extended.tiltaxis(1)=0;
if MRC.next~=0%Extended Header
    nbh=MRC.next./128;%128=lengh of FEI extended header
    if nbh==1024%FEI extended Header
        for lauf=1:nbh
            Extended.a_tilt(lauf)= fread(fid,[1],'float');        %float: 4 bytes
            Extended.b_tilt(lauf)= fread(fid,[1],'float');        %float: 4 bytes
            Extended.x_stage(lauf)= fread(fid,[1],'float');       %float: 4 bytes
            Extended.y_stage(lauf)=fread(fid,[1],'float');        %float: 4 bytes
            Extended.z_stage(lauf)=fread(fid,[1],'float');        %float: 4 bytes
            Extended.x_shift(lauf)=fread(fid,[1],'float');        %float: 4 bytes
            Extended.y_shift(lauf)=fread(fid,[1],'float');        %float: 4 bytes
            Extended.defocus(lauf)=fread(fid,[1],'float');        %float: 4 bytes
            Extended.exp_time(lauf)=fread(fid,[1],'float');       %float: 4 bytes
            Extended.mean_int(lauf)=fread(fid,[1],'float');       %float: 4 bytes
            Extended.tiltaxis(lauf)=fread(fid,[1],'float');       %float: 4 bytes
            Extended.pixelsize(lauf)=fread(fid,[1],'float');      %float: 4 bytes
            Extended.magnification(lauf)=fread(fid,[1],'float');  %float: 4 bytes
            fseek(fid,128-52,0);
            %position = ftell(fid)
        end
    else %IMOD extended Header
        fseek(fid,MRC.next,'cof');%go to end end of extended Header
    end
end
%fseek(fid,0,'eof'); %go to the end of file

if (headerOnly==1)
    Data=MRC;
    fclose(fid);
    return;
end;

if (MRC.next==0)
    fseek(fid,1024,'bof');
end;


for i=1:MRC.nz
    if MRC.mode==0
        %fseek(fid,-beval,0); %go to the beginning of the values
        Data_read(:,:,i) = fread(fid,[MRC.nx,MRC.ny],'int8');
    elseif (MRC.mode==1 || MRC.mode==6)
        %fseek(fid,-beval,0); %go to the beginning of the values
        Data_read(:,:,i) = fread(fid,[MRC.nx,MRC.ny],'int16');
    elseif MRC.mode==2
        %fseek(fid,-beval,0); %go to the beginning of the values
        Data_read(:,:,i) = fread(fid,[MRC.nx,MRC.ny],'float');
    else
        error(['Sorry, i cannot read this as an MRC-File !!!']);
        Data_read=[];
    end
end
fclose(fid);
Header=struct(...
    'Voltage',0,...
    'Cs',0,...
    'Aperture',0,...
    'Magnification',Extended.magnification(1),...
    'Postmagnification',0,...
    'Exposuretime',Extended.exp_time(1),...
    'Objectpixelsize',Extended.pixelsize(1).*1e9,...
    'Microscope',0,...
    'Pixelsize',0,...
    'CCDArea',0,...
    'Defocus',Extended.defocus(1),...
    'Astigmatism',0,...
    'AstigmatismAngle',0,...
    'FocusIncrement',0,...
    'CountsPerElectron',0,...
    'Intensity',0,...
    'EnergySlitwidth',0,...
    'EnergyOffset',0,... 
    'Tiltangle',Extended.a_tilt(1:MRC.nz),...
    'Tiltaxis',Extended.tiltaxis(1),...
    'Username',num2str(zeros(20,1)),...
    'Date',num2str(zeros(8)),...
    'Size',[MRC.nx,MRC.ny,MRC.nz],...
    'Comment',num2str(zeros(80,1)),...
    'Parameter',num2str(zeros(40,1)),...
    'Fillup',num2str(zeros(256,1)),...
    'Filename',mrc_name,...
    'Marker_X',0,...
    'Marker_Y',0,...
    'MRC',MRC);

Data=struct('Value',Data_read,'Header',Header);

clear Data_read;
