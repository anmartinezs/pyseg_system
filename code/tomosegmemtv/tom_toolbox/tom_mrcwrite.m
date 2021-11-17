function tom_mrcwrite(varargin)
%TOM_MRCWRITE writes data to MRC format file.
%
%   tom_mrcwrite(Data)
%
%   tom_mrcwrite(Data)
%   tom_mrcwrite(Data,'name',property1,'style',property2)
%
%   Writes 'Data' in a MRC format under the name property1. If 'Data' is not a
%   structure, a default header is created 3 different style can be used:
%   'classic' for a standart MRC file, 'fei' for a MRC file compatible FEI or
%   'imod' for a MRC file compatible IMOD
%
%PARAMETERS
%
%  INPUT
%   Data                Structure of Image Data
%   'name'              'PATHNAME and FILENAME' of the output file. if this option is not
%                       used, a dialog box appears to define the path and name of the
%                       output file
%   'style'             {'classic'} | 'fei' | 'imod'
%                       Specify the compatibily of the MRC file. 'classic' save as a
%                       standart MRC file, 'fei' as a FEI style and 'imod' as a IMOD style.
% 
%STRUCTURE OF MRC-DATA FILES:
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
%   4   float   x_stage  Stage x position (Unit=m. But if value>1, unit=�m)
%   4   float   y_stage  Stage y position (Unit=m. But if value>1, unit=�m)
%   4   float   z_stage  Stage z position (Unit=m. But if value>1, unit=�m)
%   4   float   x_shift  Image shift x (Unit=m. But if value>1, unit=�m)
%   4   float   y_shift  Image shift y (Unit=m. But if value>1, unit=�m)
%   4   float   z_shift  Image shift z (Unit=m. But if value>1, unit=�m)
%   4   float   defocus  Defocus Unit=m. But if value>1, unit=�m)
%   4   float   exp_time Exposuse time (s)
%   4   float   mean_int Mean value of image
%   4   float   tilt_axis   Tilt axis (deg)
%   4   float   pixel_size  Pixel size of image (m)
%   4   float   magnification   Magnification used
%   4   float   remainder   Not used (filling up to 128 bytes)   
%
%
%EXAMPLE
%   A fileselect-box appears and the data can be saved with the selected 
%   filename. The style is by default 'classic'.
%   load clown;
%   tom_mrcwrite(X);
%
%   Save the data X in 'test.mrc' with style FEI. 
%   load clown;
%   tom_mrcwrite(X,'name','test.mrc','style','fei');        
%
%REFERENCES
%
%SEE ALSO
%   TOM_MRCREAD, TOM_EMWRITE, TOM_MRCSTACK2EMSERIES, TOM_READMRCHEADER
%
%   created by SN 09/25/02
%   updated by WDN 06/22/05
%
%   Nickell et al., 'TOM software toolbox: acquisition and analysis for electron tomography',
%   Journal of Structural Biology, 149 (2005), 227-234.
%
%   Copyright (c) 2004-2007
%   TOM toolbox for Electron Tomography
%   Max-Planck-Institute of Biochemistry
%   Dept. Molecular Structural Biology
%   82152 Martinsried, Germany
%   http://www.biochem.mpg.de/tom
%

msgInvalidPair = 'Bad value for argument: ''%s''';

options = struct('name', '',...
                 'style', 'classic',...
                 'format','');
[comp_typ,maxsize,endian] = computer;
switch endian
    case 'L'
        options.format='ieee-le';
    case 'B'
        options.format='ieee-be';
end

if nargin<1 %no data
    error(['Data not specified (e.g. tom_mrcwrite(out)']);
elseif nargin==1
    if isstruct(varargin{1})
        Data=varargin{1};
    else
        Data.Value=varargin{1};
    end
    [filename, pathname] = uiputfile({'*.mrc';'*.*'}, 'Save as MRC-file');
    if isequal(filename,0) | isequal(pathname,0) 
        disp('Data not saved.'); return; 
    end
    options.name=[pathname filename];
    if isempty(findstr('.mrc',options.name))
        options.name=[options.name '.mrc'];
    end 
elseif nargin>=1
    if isstruct(varargin{1})
        Data=varargin{1};
    else
        Data.Value=varargin{1};
    end
    paramlist = varargin;
    optionsnames = lower(fieldnames(options));
    for i=2:2:length(paramlist)
        pname = paramlist{i};
        pvalue = paramlist{i+1};
        ind = strmatch(lower(pname),optionsnames);
        if isempty(ind)
            error(['Invalid parameter: ''' pname '''.']);
        elseif length(ind) > 1
            error(['Ambiguous parameter: ''' pname '''.']);
        end
        switch(optionsnames{ind})
            case 'name'
                if ischar(pvalue)
                    options.name = pvalue;
                else
                    error(sprintf(msgInvalidPair,pname));
                end
            case 'style'
                if ischar(pvalue)
                    options.style = pvalue;
                else
                    error(sprintf(msgInvalidPair,pname));
                end
            case 'format'
                if ischar(pvalue)
                    options.format = pvalue;
                else
                    error(sprintf(msgInvalidPair,pname));
                end                
        end
    end
    if isempty(options.name)
        [filename, pathname] = uiputfile({'*.mrc';'*.*'}, 'Save as MRC-file');
        if isequal(filename,0) | isequal(pathname,0)
            disp('Data not saved.'); return;
        end
        options.name=[pathname filename];
        if isempty(findstr('.mrc',options.name))
            options.name=[options.name '.mrc'];
        end
    end
end

MRC=struct('nx',0,'ny',0,'nz',0,'mode',0,...
    'nxstart',0,'nystart',0,'nzstart',0,...
    'mx',0,'my',0,'mz',0,... 
    'xlen',0,'ylen',0,'zlen',0,...
    'alpha',0,'beta',0,'gamma',0,...
    'mapc',1,'mapr',2,'maps',3,...
    'amin',0,'amax',0,'amean',0,...
    'ispg',0,'nsymbt',0,'next',0,...
    'creatid',0,'nint',0,'nreal',0,...
    'idtype',0,'lens',0,'nd1',0,'nd2',0,'vd1',0,'vd2',0,...
    'tiltangles',[0 0 0 0 0 0],...
    'xorg',0,'yorg',0,'zorg',0,...
    'cmap','MAP ','stamp',[0 0 0 0],...
    'rms',0,'nlabl',0,'labl','0');

switch options.style
    case 'classic'
        [MRC,Data]=mrc_classic(MRC, Data);
        fid=fopen(options.name,'w',options.format);
        if fid==-1
            error(['Cannot open: ' options.name ' file']);
        end;
        wclassic(MRC, Data, fid);
        wdata(MRC, Data, fid);
        fclose(fid);
    case 'imod'
        %MRC=mrc_imod(MRC, Data)
        %fid=fopen(options.name,'w',options.format)
        %if fid==-1
        %    error(['Cannot open: ' options.name ' file']);
        %end;
        %writefile(MRC, fid);
    case 'fei'
        [MRC,Data]=mrc_classic(MRC, Data);
        MRC.next=1024*128; %size of ext. header of FEI
        fid=fopen(options.name,'w',options.format);
        if fid==-1
            error(['Cannot open: ' options.name ' file']);
        end;
        wclassic(MRC, Data, fid);
        wextheader(MRC, Data, fid)
        wdata(MRC, Data, fid);
        fclose(fid);
end   


clear MRC;
clear Data;

% --------------------------------------------------------------------
% --------------------------------------------------------------------
% ----------- Other function used by tom_mrcwrite --------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------

% ----------- Function MRC_CLASSIC ------------
% use to create a classical style MRC structure
% ---------------------------------------------
function [MRC, Data]=mrc_classic(MRC,Data)
MRC.nx=size(Data.Value,1);
MRC.ny=size(Data.Value,2);
if size(Data.Value)>2
    MRC.nz=size(Data.Value,3);
else
    MRC.nz=0;
end
if isa(Data.Value,'double') | isa(Data.Value,'single')
    MRC.mode=2;
end;
if isa(Data.Value,'int16')
    MRC.mode=1;
end;
if isa(Data.Value,'int8')
    MRC.mode=0;
end;
MRC.amin=min(min(min(Data.Value)));
MRC.amax=max(max(max(Data.Value)));
MRC.amean=mean(mean(mean(Data.Value)));
if isfield(Data, 'Header')%case of an existing header
    if size(Data.Header.Comment,1)<800
        fillup=char(zeros(800-size(Data.Header.Comment,1),1));
        Data.Header.Comment=[Data.Header.Comment' fillup']';
        MRC.labl=Data.Header.Comment;
    end
    if size(Data.Header.Parameter,1)<52
        fillup=zeros(52-size(Data.Header.Parameter,1),1);
        Data.Header.Parameter=[Data.Header.Parameter' fillup']';
    end;
    Data.Header.Parameter(14)=Data.Header.Tiltangle;
else%case of no header
    Data.Header.Comment=char(zeros(800,1));
    MRC.labl=Data.Header.Comment;
    Data.Header.Parameter=zeros(52,1);
    if ndims(Data.Value)==1
        Data.Header.Size=[size(Data.Value,1)];
    elseif ndims(Data)==2
        Data.Header.Size=[size(Data.Value,1) size(Data.Value,2)];
    elseif ndims(Data)==3
        Data.Header.Size=[size(Data.Value,1) size(Data.Value,2) size(Data.Value,3)];
    end;
    Data.Header.Tiltangle=0;
    Data.Header.Tiltaxis=0;
    Data.Header.Defocus=0;
    Data.Header.Exposuretime=0;
    Data.Header.Tiltaxis=0;
    Data.Header.Pixelsize=0;
    Data.Header.Magnification=0;
end
MRC.mx=MRC.nx;
MRC.my=MRC.ny;
MRC.mz=MRC.nz;
MRC.xlen=MRC.nx;
MRC.ylen=MRC.ny;
MRC.zlen=MRC.nz;

% ----------- Function WCLASSIC ---------------
% use to write the MRC Header
% ---------------------------------------------
function wclassic(MRC, Data, fid) 
%%%%%%%% Write the header: 1024 bytes %%%%%%%% 
fwrite(fid,MRC.nx,'int');           %integer: 4 bytes
fwrite(fid,MRC.ny,'int');           %integer: 4 bytes
fwrite(fid,MRC.nz,'int');           %integer: 4 bytes
fwrite(fid,MRC.mode,'int');         %integer: 4 bytes
fwrite(fid,MRC.nxstart,'int');      %integer: 4 bytes
fwrite(fid,MRC.nystart,'int');      %integer: 4 bytes
fwrite(fid,MRC.nzstart,'int');      %integer: 4 bytes
fwrite(fid,MRC.mx,'int');           %integer: 4 bytes
fwrite(fid,MRC.my,'int');           %integer: 4 bytes
fwrite(fid,MRC.mz,'int');           %integer: 4 bytes
fwrite(fid,MRC.xlen,'float');       %float: 4 bytes
fwrite(fid,MRC.ylen,'float');       %float: 4 bytes
fwrite(fid,MRC.zlen,'float');       %float: 4 bytes
fwrite(fid,MRC.alpha,'float');      %float: 4 bytes
fwrite(fid,MRC.beta,'float');       %float: 4 bytes
fwrite(fid,MRC.gamma,'float');      %float: 4 bytes
fwrite(fid,MRC.mapc,'int');         %integer: 4 bytes
fwrite(fid,MRC.mapr,'int');         %integer: 4 bytes
fwrite(fid,MRC.maps,'int');         %integer: 4 bytes
fwrite(fid,MRC.amin,'float');       %float: 4 bytes
fwrite(fid,MRC.amax,'float');       %float: 4 bytes
fwrite(fid,MRC.amean,'float');      %float: 4 bytes
fwrite(fid,MRC.ispg,'short');       %integer: 2 bytes
fwrite(fid,MRC.nsymbt,'short');     %integer: 2 bytes
fwrite(fid,MRC.next,'int');         %integer: 4 bytes
fwrite(fid,MRC.creatid,'short');    %integer: 2 bytes
for i=1:30 
    fwrite(fid,[1],'char');         %not used: 30 bytes
end
fwrite(fid,MRC.nint,'short');       %integer: 2 bytes
fwrite(fid,MRC.nreal,'short');      %integer: 2 bytes
for i=1:28
    fwrite(fid,[1],'char');         %not used: 28 bytes
end
fwrite(fid,MRC.idtype,'short');     %integer: 2 bytes
fwrite(fid,MRC.lens,'short');       %integer: 2 bytes
fwrite(fid,MRC.nd1,'short');        %integer: 2 bytes
fwrite(fid,MRC.nd2,'short');        %integer: 2 bytes
fwrite(fid,MRC.vd1,'short');        %integer: 2 bytes
fwrite(fid,MRC.vd2,'short');        %integer: 2 bytes
fwrite(fid,MRC.tiltangles,'float'); %float: 6*4 bytes=24 bytes
fwrite(fid,MRC.xorg,'float');       %float: 4 bytes
fwrite(fid,MRC.yorg,'float');       %float: 4 bytes
fwrite(fid,MRC.zorg,'float');       %float: 4 bytes
fwrite(fid,MRC.cmap,'char');        %Character: 4 bytes
fwrite(fid,MRC.stamp,'char');       %Character: 4 bytes
fwrite(fid,MRC.nlabl,'int');        %integer: 4 bytes
fwrite(fid,MRC.rms,'float');        %float: 4 bytes
fwrite(fid,MRC.labl,'char');        %Character: 800 bytes
%position = ftell(fid)

% ----------- Function WEXTHEADER -------------
% use to write the extended Header
% ---------------------------------------------
function wextheader(MRC, Data, fid)
%%%%%%%% Write extended header: 1024*(52+76) bytes %%%%%%%% 
for lauf=1:1024
    fwrite(fid,Data.Header.Tiltangle,'float');  %float: 4 bytes
    fwrite(fid,Data.Header.Tiltaxis,'float');   %float: 4 bytes
    fwrite(fid,0,'float');                      %float: 4 bytes
    fwrite(fid,0,'float');                      %float: 4 bytes
    fwrite(fid,0,'float');                      %float: 4 bytes
    fwrite(fid,0,'float');                      %float: 4 bytes
    fwrite(fid,0,'float');                      %float: 4 bytes
    fwrite(fid,Data.Header.Defocus,'float');    %float: 4 bytes
    fwrite(fid,Data.Header.Exposuretime,'float');%float: 4 bytes
    fwrite(fid,0,'float');                      %float: 4 bytes
    fwrite(fid,Data.Header.Tiltaxis,'float');   %float: 4 bytes
    fwrite(fid,Data.Header.Pixelsize,'float');  %float: 4 bytes
    fwrite(fid,Data.Header.Magnification,'float');%float: 4 bytes
    %                                            total: 52 bytes                                         
    for c1=1:19 
        fwrite(fid,[1],'float');%19*4=76 bytes
    end
    %lenght of extended header: 52+76=128 bytes
end
%position = ftell(fid)
%total lenght of extended header: 1024*128= 131072 bytes

% ----------- Function WDATA ------------------
% use to write the DATA after the header
% ---------------------------------------------
function wdata(MRC, Data, fid)
%%%%%%%% Write values %%%%%%%% 
for lauf=1:(MRC.nz)
	Data_write=Data.Value(1:(MRC.nx),1:(MRC.ny),lauf);
	if MRC.mode==0
		fwrite(fid,Data_write,'int8');
	elseif MRC.mode==1
		fwrite(fid,Data_write,'int16');
	elseif MRC.mode==2
		fwrite(fid,Data_write,'float');
	else
		disp('Sorry, i cannot write this as an EM-File !!!');
	end;
end;
clear Data_write;

