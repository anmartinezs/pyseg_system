function [star_st,Header]=tom_starread(filename,outputFlavour,NumOfLines2chk)
% tom_starread read star files
%  
%     star_st=tom_starread(filename,outputFlavour)
%  
%  PARAMETERS
%  
%    INPUT
%     filename         filename of the star file
%     outputFlavour    ('struct') or 'matrix' 
%     NumOfLines2chk   (10) lines to check 4 consitent data-type  
%
%    OUTPUT
%     star_st           matlab struct 
%                          
%  
%  EXAMPLE
%      star_st=tom_starread('in.star');
%
%  REFERENCES
%  
%  NOTE:
%  
%
%  SEE ALSO
%      tom_xmippdocread,tom_spiderread
%  
%     created by FB 04/12/13
%  
%     Nickell et al., 'TOM software toolbox: acquisition and analysis for electron tomography',
%     Journal of Structural Biology, 149 (2005), 227-234.
%  
%     Copyright (c) 2004-2007
%     TOM toolbox for Electron Tomography
%     Max-Planck-Institute of Biochemistry
%     Dept. Molecular Structural Biology
%     82152 Martinsried, Germany
%     http://www.biochem.mpg.de/tom
% 

if (nargin < 2)
   outputFlavour='struct'; 
end;

if (nargin < 3)
   NumOfLines2chk=1; 
end;


fid=fopen(filename);
if (fid < 0)
    error(['Cannot read: ' filename]);
end;

Header=parse_header(fid);
[framatString,isStrVect]=genFormatString(fid,NumOfLines2chk,Header.NumOfColumns,Header.NumOfTotalLines);

dataRaw=readData(fid,framatString);
star_st=transformDataStruct(dataRaw,isStrVect,Header,outputFlavour);


function star_st=transformDataStruct(dataRaw,isStrVect,Header,outputFlavour)

if (strcmp(outputFlavour,'struct'))
    star_st=Raw2Struct(dataRaw,isStrVect,Header.fieldNamesNoCommet);
    star_st(1).Header=Header;
end;
if (strcmp(outputFlavour,'matrix'))
    star_st=Raw2Matrix(dataRaw);
end;


function matrix=Raw2Matrix(dataRaw)

for i=1:size(dataRaw,2)
    tmp=dataRaw{i}(:);
    if (iscell(tmp))
        matrix(:,i)=dataRaw{i}(:);
    else
        matrix(:,i)=num2cell(dataRaw{i});
    end;
end;


function st=Raw2Struct(dataRaw,isStrVect,fieldnames)

structCommndString='st=struct(';
for i=1:length(fieldnames)
    if (isStrVect(i))
        structCommndString=[structCommndString '''' fieldnames{i} '''' ',dataRaw{' num2str(i) '},' ];
    else
        structCommndString=[structCommndString '''' fieldnames{i} '''' ', num2cell(dataRaw{' num2str(i) '}),' ]; 
    end;
end;
structCommndString=structCommndString(1:end-1);
structCommndString=[structCommndString ');'];
eval(structCommndString);

function data=readData(fid,framatString)

data = textscan(fid,framatString);



function [formatStr,isStrVect]=genFormatString(fid,NumOfLines2chk,NumOfColumns,NumOfHeaderLines)

formatStr='';
for i=1:NumOfColumns
    formatStr=[formatStr '%s '];
end;
tmp = textscan(fid,formatStr,NumOfLines2chk);

formatStr='';
for colNr=1:NumOfColumns
   col=tmp{colNr};
   for ii=1:length(col)
         isString(ii)=isnan(str2double(col{ii}));
   end;
   if (std(isString)~=0)
       error('Colomn data type is inconsistend!!');
   end;
   if (round(mean(isString))==1)
       formatStr=[formatStr ' %s']; 
       isStrVect(colNr)=1;
   else
       formatStr=[formatStr ' %f']; 
       isStrVect(colNr)=0;
   end;
end;

frewind(fid);
for i=1:NumOfHeaderLines
    fgetl(fid);
end;


function Header=parse_header(fid)

numOfTotalLines=0;
Header.isLoop=0;
zz=1;
while 1
    tmpLine=fgetl(fid);
    numOfTotalLines=numOfTotalLines+1; 
    
    if (isempty(tmpLine)  ||  strcmp(tmpLine(1),'#' ) )
        continue;
    end;
    
    if ( (tmpLine(1)=='_') || strcmp(deblank(tmpLine),'loop_') || isempty(findstr(tmpLine,'data_'))==0 )
        
        if (isempty(findstr(tmpLine,'data_'))==0)
            Header.title=tmpLine;
        end;
        
        if (strcmp(deblank(tmpLine),'loop_'))
             Header.isLoop=1; 
        end;
        
        if (tmpLine(1)=='_')
            Header.fieldNames{zz}=tmpLine;
            [tok rest]=strtok(tmpLine(2:end),' ');
            Header.fieldNamesNoCommet{zz}=tok;
            Header.fieldNamesCommet{zz}=strrep(rest,' ','');
            zz=zz+1;
        end;
    else
        break;
    end;
end;
Header.NumOfColumns=zz-1;
Header.NumOfTotalLines=numOfTotalLines-1;

frewind(fid);
for i=1:Header.NumOfTotalLines
    fgetl(fid);
end;
