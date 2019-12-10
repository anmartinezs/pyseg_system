%% GUI for supervising vesicles segmentation
% Author: Antonio Martinez-Sanchez

function varargout = vesseg(varargin)
% VESSEG MATLAB code for vesseg.fig
%      VESSEG, by itself, creates a new VESSEG or raises the existing
%      singleton*.
%
%      H = VESSEG returns the handle to a new VESSEG or the handle to
%      the existing singleton*.
%
%      VESSEG('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in VESSEG.M with the given input arguments.
%
%      VESSEG('Property','Value',...) creates a new VESSEG or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before vesseg_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to vesseg_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help vesseg

% Last Modified by GUIDE v2.5 31-Jan-2013 11:27:59

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @vesseg_OpeningFcn, ...
                   'gui_OutputFcn',  @vesseg_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before vesseg is made visible.
function vesseg_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to vesseg (see VARARGIN)

% Choose default command line output for vesseg
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes vesseg wait for user response (see UIRESUME)
% uiwait(handles.figure1);

% Declare global variables
global I;
global L;
global C;
global F;
global R;
global S;
global V;
% global M;
global Irgb;
global cur_state;
global pa;
global pc;
global la;
global idir;
global vesq;
global vesr;
global curs;
global ul;
global ur;

% Initialization
cs = 0;
cur_state = 0;
pc = zeros( 3, 1 );
curs = datacursormode( hObject );

% Load input tomograms
[fname,idir] = uigetfile( {'*.mrc';'*.rec';'*.*'}, 'Stem name of input data' );    
if fname == 0
    close( hObject );
    return
end
file = sprintf( '%s/%s', idir, fname );
T = tom_mrcread( file );
I = T.Value;
% I = readmrc( file, 1024 );
[~,fstem,fext] = fileparts( fname );
file = sprintf( '%s/%s_lbl%s', idir, fstem, fext );
T = tom_mrcread( file );
L = single( T.Value );
% L = readmrc( file, 1024 );
file = sprintf( '%s/%s_clft%s', idir, fstem, fext );
T = tom_mrcread( file );
C = T.Value;
% C = readmrc( file, 1024 );
[Nx,Ny,Nz] = size( I );
V = zeros( Nx, Ny, Nz, 3 );
file = sprintf( '%s/%s_n1%s', idir, fstem, fext );
T = tom_mrcread( file );
V(:,:,:,1) = double( T.Value );
% V(:,:,:,1) = readmrc( file, 1024 );
file = sprintf( '%s/%s_n2%s', idir, fstem, fext );
T = tom_mrcread( file );
V(:,:,:,2) = double( T.Value );
% V(:,:,:,2) = readmrc( file, 1024 );
file = sprintf( '%s/%s_n3%s', idir, fstem, fext );
T = tom_mrcread( file );
V(:,:,:,3) = double( T.Value );
% V(:,:,:,3) = readmrc( file, 1024 );
Irgb = zeros( Nx, Ny, 3 );

% Create slice slide range
slider_step(1) = 1/Nz;
slider_step(2) = 5/Nz;
set(handles.sldr_zslc,'sliderstep', slider_step, 'max', Nz, 'min', 1 );
set( handles.sldr_zslc, 'Value', round(Nz/2) );

% Show input image
S = zeros( Nx, Ny, Nz );
Id = L > 0;
S(Id) = 2;
Id = C > 0;
S(Id) = 1;
update_disp( handles );


% --- Outputs from this function are returned to the command line.
function varargout = vesseg_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on slider movement.
function sldr_zslc_Callback(hObject, eventdata, handles)
% hObject    handle to sldr_zslc (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

update_disp( handles );


% imshow( I(:,:,round(slider_value)), [] );

% --- Executes during object creation, after setting all properties.
function sldr_zslc_CreateFcn(hObject, eventdata, handles)
% hObject    handle to sldr_zslc (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on button press in pbtn_getazp.
function pbtn_getazp_Callback(hObject, eventdata, handles)
% hObject    handle to pbtn_getazp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Diplay a cursor for catching a point from AZ

global cur_state;
global curs;
global la;
global pa;
global C;
global S;

if cur_state == 1
    % Check if the point is a membrane
    st = getCursorInfo( curs );
    x = st.Position(2);
    y = st.Position(1);
    z = round( get(handles.sldr_zslc,'Value') );
    la = C(x,y,z);
    if la>0
        pa = [x; y; z];
        Id = C == la;
        S = zeros( size(S) );
        S(Id) = 1;
        datacursormode off;
        cur_state = 0;
        set( hObject, 'String', 'AZ Cursor' );
    else
        warndlg('An AZ point must be part of a cleft membrane' );
    end    
elseif cur_state == 0
    datacursormode on;
    cur_state = 1;
    set( hObject, 'String', 'Capture' );
end


% --- Executes on button press in pbtn_getpsnp.
function pbtn_getpsnp_Callback(hObject, eventdata, handles)
% hObject    handle to pbtn_getpsnp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Diplay a cursor for catching a point from Presynaptic Citoplasm

global cur_state;
global curs;
global pc;
global L;

if cur_state == 2
    % Check if the point is a membrane
    st = getCursorInfo( curs );
    x = st.Position(2);
    y = st.Position(1);
    z = round( get(handles.sldr_zslc,'Value') );
    % Check if the point is not a membrane
    if L(x,y,z) == 0
        pc = [x; y; z];
        datacursormode off;
        cur_state = 0;
        set( hObject, 'String', 'PSN Cursor' );
    else
        warndlg('An PSN point must not be part of a membrane' );
    end      
elseif cur_state == 0
    datacursormode on;
    cur_state = 2;
    set( hObject, 'String', 'Capture' );
end

% --- Executes on button press in pbtn_upc.
function pbtn_upc_Callback(hObject, eventdata, handles)
% hObject    handle to pbtn_upc (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Fill up presinaptic citoplasmic region
global L;
global C;
global V;
global S;
global la;
global pc;
global pa;

if (la>0) && (sum(pc)>0)
    H = (C==la) .* L;
    lm = mode( H(H>0) );
    clear H;
    Idh = L == lm;
    Id = logical( Idh .* (S~=1) );
    S(Id) = 2;
    clear Id;
    crop_syn_memb();
    Idh = logical( (S==2) + (S==1) );
    h = helpdlg('Getting membrane orientation, it can take some time...','vesseg Info');
    v = pc - pa;
    v = v / sqrt(sum(v.*v));
    N = spreador( Idh, V, pa, v );
    clear Idh;
    close( h );
    N = filling( N );
    Id = N>0;
    S(Id) = 3;
    update_disp( handles );
else
    warndlg( 'No AZ and PSN points selected' );
end

% --- Executes on button press in pbtn_azdist.
function pbtn_azdist_Callback(hObject, eventdata, handles)
% hObject    handle to pbtn_azdist (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get PSN region attached to AZ
global S;

Id1 = (S==1);
s1 = length( S(Id1) );
if s1 > 0
    Id2 = (S==1);
    s1 = length( S(Id2) );
    if s1 > 0
        B = bwdist( Id2 );
        th = str2double( get(handles.edt_az,'String') );
        Sh = (S==3) + (S==4);
        Id1 = logical( Sh .* (B<=th) );
        S(Id1) = 4;
        Id1 = logical( Sh .* (~Id1) );
        S(Id1) = 3;
        update_disp( handles );
    else
        warndlg( 'No AZ segmented' );
    end
else
    warndlg( 'No membrane segmented' );
end


function edt_az_Callback(hObject, eventdata, handles)
% hObject    handle to edt_az (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edt_az as text
%        str2double(get(hObject,'String')) returns contents of edt_az as a double


% --- Executes during object creation, after setting all properties.
function edt_az_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edt_az (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pbtn_savl.
function pbtn_savl_Callback(hObject, eventdata, handles)
% hObject    handle to pbtn_savl (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Save segmentation result
global S;

%% TODO: save according to the format used by Vladan
tom_mrcwrite( S );

% --- Executes on button press in pbtn_upsens.
function pbtn_upsens_Callback(hObject, eventdata, handles)
% hObject    handle to pbtn_upsens (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

import java.util.LinkedList
global F;
global R;
global vesq;
global ul;
global ur;

% Update current sesitivity
cs = str2double( get(handles.ed_sens,'String') );
if (cs>=1) && (cs<=255)

    % Get crop versions of the tomogram
    Fh = F(ul(2):ur(2),ul(1):ur(1),ul(3):ur(3));
    Rh = R(ul(2):ur(2),ul(1):ur(1),ul(3):ur(3));

    % Get vesicles from cropped filtered tomogram
    T = logical( Fh < cs );
    St = regionprops( T, 'Centroid' );
    vesq = LinkedList();
    if ~isempty(St)
        C = round( cat(1, St.Centroid) );
        lc = length(C(:,1));
        Se = zeros( lc, 1 );
        Re = zeros( lc, 1 );
        for k = 1:lc
            Se(k) = Fh(C(k,2),C(k,1),C(k,3));
            Re(k) = Rh(C(k,2),C(k,1),C(k,3));
        end

        % Sort by sensitivity ascend
        [Se,Id] = sort( Se, 1, 'ascend' );
        C(:,1) = C(Id,1);
        C(:,2) = C(Id,2);
        C(:,3) = C(Id,3);
        Re = Re(Id);

        % Add to vesicle queue
        for k = 1:lc
            vesq.add( [C(k,2)+ul(2)-1; C(k,1)+ul(1)-1; C(k,3)+ul(3); Re(k); Se(k); 1] );
        end

        % Repaint segmentation
        set( handles.rbtn_flt, 'Value', 1 );
        uipan_view_SelectionChangeFcn(handles.rbtn_flt, eventdata, handles);
    end
else
    helpdlg('Sensitivity must be in range [1,255].','vesseg Info');
end

% --- Executes on button press in pbtn_flt.
function pbtn_flt_Callback(hObject, eventdata, handles)
% hObject    handle to pbtn_flt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

import java.util.LinkedList
global I;
global S;
global F;
global R;
% global M;
% global V;
global vesq;
global ul;
global ur;
global cs;

% Check if we are reading for applying the filter
Id = S >= 4;
Id = logical( Id>0 );
s1 = sum(sum(sum( Id )));
if s1 > 0
    % Update GUI
    vesq = LinkedList();
    cs = 0;
    S(Id) = 4;
    update_disp( handles );
    [Nx,Ny,Nz] = size( S );
    rm = str2double( get( handles.ed_mr, 'String' ) );
    rM = str2double( get( handles.ed_Mr, 'String' ) );
    th = str2double( get( handles.ed_thick, 'String' ) );
    wa = str2double( get( handles.ed_mw_ang, 'String' ) );
    wr = str2double( get( handles.ed_mw_angrot, 'String' ) );
    % sen = sen(1);
    cth = ceil( th );
    % Crop tomogram to AZ region in order to increase speed
    St = regionprops( Id, 'Area', 'BoundingBox' );
    if ~isempty(St)
        A = round( cat(1,St.Area) );
        B = round( cat(1,St.BoundingBox) );
        [~,id] = sort( A, 1, 'descend' );
        uli = floor( [B(id(1),1); B(id(1),2); B(id(1),3)] );
        width = ceil( [B(id(1),4); B(id(1),5); B(id(1),6)] );
        ul = uli - rM - cth;
        if ul(1) < 1
            ul(1) = 1;
        end
        if ul(2) < 1
            ul(2) = 1;
        end
        if ul(3) < 1
            ul(3) = 1;
        end
        ur = uli + width + rM + cth;
        if ur(1) > Ny
            ur(1) = Ny;
        end
        if ur(2) > Nx
            ur(2) = Nx;
        end
        if ur(3) > Nz
            ur(3) = Nz;
        end
        Ic = I .* Id;
        Ic = Ic(ul(2):ur(2),ul(1):ur(1),ul(3):ur(3));
        Sc = S(ul(2):ur(2),ul(1):ur(1),ul(3):ur(3));
        Id = Id(ul(2):ur(2),ul(1):ur(1),ul(3):ur(3));        
        % Vesicles filter
        [p,c] = linmap( min(Ic(Id)), max(Ic(Id)), 0, 1 );
        Ic(Id) = Ic(Id)*p + c;
        [Fh,Rh] = vesdettmg( Ic, (Sc>0).*(Sc<3), Sc==4, rm, rM, th, wa, wr );
        F = zeros( Nx, Ny, Nz );
        R = zeros( Nx, Ny, Nz );
        F(ul(2):ur(2),ul(1):ur(1),ul(3):ur(3)) = Fh;
        R(ul(2):ur(2),ul(1):ur(1),ul(3):ur(3)) = Rh;
    else
        warndlg( 'No Presynaptic region segmented' );
    end
else
    warndlg( 'No AZ region segmented' );
end

function ed_mr_Callback(hObject, eventdata, handles)
% hObject    handle to ed_mr (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ed_mr as text
%        str2double(get(hObject,'String')) returns contents of ed_mr as a double


% --- Executes during object creation, after setting all properties.
function ed_mr_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ed_mr (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function ed_Mr_Callback(hObject, eventdata, handles)
% hObject    handle to ed_Mr (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ed_Mr as text
%        str2double(get(hObject,'String')) returns contents of ed_Mr as a double


% --- Executes during object creation, after setting all properties.
function ed_Mr_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ed_Mr (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function ed_thick_Callback(hObject, eventdata, handles)
% hObject    handle to ed_thick (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ed_thick as text
%        str2double(get(hObject,'String')) returns contents of ed_thick as a double


% --- Executes during object creation, after setting all properties.
function ed_thick_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ed_thick (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pbtn_savf.
function pbtn_savf_Callback(hObject, eventdata, handles)
% hObject    handle to pbtn_savf (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global idir;
global F;
global R;

% Save filtered result
if numel(size(F)) > 0
    fspec{1} = '*.mrc';
    fspec{2} = '*.rec';
    [FileName,PathName] = uiputfile( fspec, 'Save Filtered Result (Coef. and Radius)', idir );
    [~,name,ext] = fileparts( FileName ); 
    file = sprintf( '%s%s_ves%s', PathName, name, ext );
    tom_mrcwrite( F, 'name', file );
    file = sprintf( '%s%s_rad%s', PathName, name, ext );
    tom_mrcwrite( R, 'name', file );
else
    warndlg( 'No filtered result to save.' );
end

% Update image on screen
function update_disp( handles )

global I;
global S;
global Irgb;

% Get slice
slider_value = round( get( handles.sldr_zslc, 'Value' ) );

% Crete a RBG gray background image
Ih = I(:,:,slider_value);
Sh = S(:,:,slider_value);
[p c] = linmap( min(min(min(I))), max(max(max(I))), 0, 255 );
Ih = Ih*p + c;
Ih1 = Ih;
Ih2 = Ih;
Ih3 = Ih;

% Paint objects
% AZ
Id = Sh == 1;
if sum(sum(Id)) > 0
    Ih1(Id) = 255;
    Ih2(Id) = 0;
    Ih3(Id) = 0;
end
% Ext. Memb
Id = Sh == 2;
if sum(sum(Id)) > 0
    Ih1(Id) = 0;
    Ih2(Id) = 255;
    Ih3(Id) = 0;
end
% Cito.
Id = Sh == 3;
if sum(sum(Id)) > 0
    hv = Ih(Id);
    lh = length( hv );
    clear H;
    H(:,1) = .6667 * ones( lh, 1 ); % Red
    H(:,2) = .5 * ones( lh, 1 );
    H(:,3) = hv;
    H = hsv2rgb( H );
    Ih1(Id) = H(:,1);
    Ih2(Id) = H(:,2);
    Ih3(Id) = H(:,3);
end
% Dist.
Id = Sh == 4;
if sum(sum(Id)) > 0
    hv = Ih(Id);
    lh = length( hv );
    clear H;
    H(:,1) = .5 * ones( lh, 1 ); % Cyan
    H(:,2) = .5 * ones( lh, 1 );
    H(:,3) = hv;
    H = hsv2rgb( H );
    Ih1(Id) = H(:,1);
    Ih2(Id) = H(:,2);
    Ih3(Id) = H(:,3);
end
% Vesicles
Id = Sh > 4;
if sum(sum(Id)) > 0
    hv = Ih(Id);
    lh = length( hv );
    clear H;
    H(:,1) = .1667 * ones( lh, 1 ); % Yellow
    H(:,2) = .5 * ones( lh, 1 );
    H(:,3) = hv;
    H = hsv2rgb( H );
    Ih1(Id) = H(:,1);
    Ih2(Id) = H(:,2);
    Ih3(Id) = H(:,3);
end

% Print on screen
Irgb(:,:,1) = Ih1;
Irgb(:,:,2) = Ih2;
Irgb(:,:,3) = Ih3;
axes( handles.disp );
imshow( Irgb/255, [] );

% mode: if == 1 -> filtered vesicles, otherwise -> refined vesicles
function paint_vesicles( vq )

global S;
import java.util.LinkedList

% Erase previous vesicles
[Nx,Ny,Nz] = size( S );
Id = S >= 4;
S(Id) = 4;

% Loop for vesicles
if ~isempty( vq )
    lv = vq.size();
    h = waitbar( 0, 'Repainting vesicles...' );
    mx = 1 / lv;
    for k = 0:lv-1
        ves = vq.get( k );
        if ves(6) > 0
            r = ceil( ves(4) );
            x = round( ves(1) );
            y = round( ves(2) );
            z = round( ves(3) );
            % Get ROI
            if (x-r) > 0
                rxl = -r;
                wxl = x - r;
            else
                rxl = -x+1;
                wxl = 1;
            end
            if (x+r) < Nx
                rxh = r;
                wxh = x + r;
            else
                rxh = Nx - x;
                wxh = Nx;
            end
            if (y-r) > 0
                ryl = -r;
                wyl = y - r;
            else
                ryl = -y+1;
                wyl = 1;
            end
            if (y+r) < Ny
                ryh = r;
                wyh = y + r;
            else
                ryh = Ny - y;
                wyh = Ny;
            end
            if (z-r) > 0
                rzl = -r;
                wzl = z - r;
            else
                rzl = -z+1;
                wzl = 1;
            end
            if (z+r) < Nz
                rzh = r;
                wzh = z + r;
            else
                rzh = Nz - z;
                wzh = Nz;
            end

            % Build model
            [X Y Z] = meshgrid( ryl:ryh, rxl:rxh, rzl:rzh );
            R = sqrt( X.*X+Y.*Y+Z.*Z );
            Id = R<=r;
            R(Id) = (k+5);
            Id = ~Id;
            Sh = S(wxl:wxh,wyl:wyh,wzl:wzh);
            R(Id) = Sh(Id);
            % Paint vesicle
            S(wxl:wxh,wyl:wyh,wzl:wzh) = R;
        end
        waitbar( k*mx, h );
    end
    close( h );
end

function fill_vesicles( or )

import java.util.LinkedList
global S;
% global M;
% global V;
global R;
global vesq;
global vesr;

% Erase previous painted vesicles
% [Nx,Ny,Nz] = size( S );
Id = S >= 4;
S(Id) = 4;

% Loop for vesicles
lv = vesq.size();
h = waitbar( 0, 'Refining vesicles...' );
mx = 1 / lv;
% r = ceil( w/2 );
vesr = LinkedList(); 
for k = 0:lv-1
    ves = vesq.get( k );
    x = ves(1);
    y = ves(2);
    z = ves(3);
%     % Get ROI
%     if (x-r) > 0
%         wxl = x - r;
%         xc = r + 1;
%     else
%         wxl = 1;
%         xc = x;
%     end
%     if (x+r) < Nx
%         wxh = x + r;
%     else
%         wxh = Nx;
%     end
%     if (y-r) > 0
%         wyl = y - r;
%         yc = r + 1;
%     else
%         wyl = 1;
%         yc = y;
%     end
%     if (y+r) < Ny
%         wyh = y + r;
%     else
%         wyh = Ny;
%     end
%     if (z-r) > 0
%         zc = r + 1;
%         wzl = z - r;
%     else
%         wzl = 1;
%         zc = z;
%     end
%     if (z+r) < Nz
%         wzh = z + r;
%     else
%         wzh = Nz;
%     end
    
    % Get cropped original data
%     Mc = M(wxl:wxh,wyl:wyh,wzl:wzh);
%     Vc = V(wxl:wxh,wyl:wyh,wzl:wzh,:);
%     Sc = S(wxl:wxh,wyl:wyh,wzl:wzh,:);
%     Rc = R(wxl:wxh,wyl:wyh,wzl:wzh,:);
%     Mc = Mc .* Sc;    
    
%     % Fill inside vesicle
%     if sn > 0
%         [co ro] = fillinves2( Mc.*((Sc==4)+(Sc==3)), Vc, [xc; yc; zc], sn, rg, or );
%         if ro > 0
%             % vesr.add( [wxl+co(1); wyl+co(2); wzl+co(3); ro; ves(5); 0] ); 
%             if x < 1
%                 x = 1;
%             elseif x > Nx
%                 x = Nx;
%             end
%             if y < 1
%                 y = 1;
%             elseif y > Ny
%                 y = Ny;
%             end
%             if z < 1
%                 z = 1;
%             elseif z > Nz
%                 z = Nz;
%             end
%             vesr.add( [wxl+co(1); wyl+co(2); wzl+co(3); R(x,y,z)+or; ves(5); 0] ); 
%         end
%     else
        vesr.add( [x; y; z; R(x,y,z)+or; ves(5); 0] );
%     end
    
    waitbar( k*mx, h );
end
close( h );

% Global rules filtering
glrulesves( vesr );


% --- Executes on button press in pbtn_fill.
function pbtn_fill_Callback(hObject, eventdata, handles)
% hObject    handle to pbtn_fill (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% s = ceil( str2double(get(handles.ed_thick,'String')) );
% Mr = str2double(get(handles.ed_Mr,'String'));
% mr = str2double(get(handles.ed_mr,'String'));
% or = str2double(get(handles.ed_off,'String'));
% r = ceil( Mr );
% fill_vesicles( 2*(s+r), [mr Mr], sn, or );
or = str2double(get(handles.ed_off,'String'));
fill_vesicles( or );
set( handles.rbtn_ref, 'Value', 1 );
uipan_view_SelectionChangeFcn(handles.rbtn_ref, eventdata, handles);



function ed_mw_ang_Callback(hObject, eventdata, handles)
% hObject    handle to ed_mw_ang (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ed_mw_ang as text
%        str2double(get(hObject,'String')) returns contents of ed_mw_ang as a double


% --- Executes during object creation, after setting all properties.
function ed_mw_ang_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ed_mw_ang (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function ed_sph_Callback(hObject, eventdata, handles)
% hObject    handle to ed_sph (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ed_sph as text
%        str2double(get(hObject,'String')) returns contents of ed_sph as a double


% --- Executes during object creation, after setting all properties.
function ed_sph_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ed_sph (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function ed_off_Callback(hObject, eventdata, handles)
% hObject    handle to ed_off (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ed_off as text
%        str2double(get(hObject,'String')) returns contents of ed_off as a double


% --- Executes during object creation, after setting all properties.
function ed_off_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ed_off (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes when selected object is changed in uipan_view.
function uipan_view_SelectionChangeFcn(hObject, eventdata, handles)
% hObject    handle to the selected object in uipan_view 
% eventdata  structure with the following fields (see UIBUTTONGROUP)
%	EventName: string 'SelectionChanged' (read only)
%	OldValue: handle of the previously selected object or empty if none was selected
%	NewValue: handle of the currently selected object
% handles    structure with handles and user data (see GUIDATA)

global vesq;
global vesr;

if hObject == handles.rbtn_flt
    paint_vesicles( vesq );
else
    paint_vesicles( vesr );
end

update_disp( handles );


% Global rules for vesicles filtering
function glrulesves( vq )
    
import java.util.LinkedList
global S;

h = waitbar( 0, 'Applying global rules...' );

% Distance transform for membrane
B = bwdist( S<3 );
[Nx Ny Nz] = size( B );

% Loop for distance between vesicles
lq = vq.size();
Id = ones( lq, 1 );
mx = 1 / lq;
for k = 0:lq-1
        
    waitbar( (k+1)*mx, h );

    % Current vesicle
    ves = vq.get( k );
    x = round( ves(1) );
    if x < 1 
        x = 1;
    elseif x > Nx
        x = Nx;
    end
    y = round( ves(2) );
    if y < 1 
        y = 1;
    elseif y > Ny
        y = Ny;
    end
    z = round( ves(3) );
    if z < 1 
        z = 1;
    elseif z > Nz
        z = Nz;
    end
    r = ves(4);

    % 1st rule -> distance to membrane
    if (r>0) && (B(x,y,z)>r)    

        % 2nd rule -> distance between vesicles
        for kk = 0:lq-1;
            if k ~= kk
                vesn = vq.get( kk );
                d = sqrt( (x-vesn(1))^2 + (y-vesn(2))^2 + (z-vesn(3))^2 );
                if d < (r+vesn(4))
                    if vesn(5) >= ves(5)
                        Id(kk+1) = 0;
                        Id(k+1) = 1;
                    else
                        Id(k+1) = 0;
                        Id(kk+1) = 1;
                    end
                end
            end
        end
    else
        Id(k+1) = 0;
    end
end

% Mark included vesicles
for k = 0:lq-1;
    if Id(k+1)
        ves = vq.get( k );
        ves(6) = 1;
        vq.set( k, ves );
    end
end

close( h );



function ed_mw_angrot_Callback(hObject, eventdata, handles)
% hObject    handle to ed_mw_angrot (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ed_mw_angrot as text
%        str2double(get(hObject,'String')) returns contents of ed_mw_angrot as a double


% --- Executes during object creation, after setting all properties.
function ed_mw_angrot_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ed_mw_angrot (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function ed_sens_Callback(hObject, eventdata, handles)
% hObject    handle to ed_sens (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ed_sens as text
%        str2double(get(hObject,'String')) returns contents of ed_sens as a double


% --- Executes during object creation, after setting all properties.
function ed_sens_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ed_sens (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% Crop presynaptic membrane to AZ in Z axis
function crop_syn_memb()

global S;

St = regionprops( S==1, 'Area', 'BoundingBox' );
if ~isempty(St)
    A = round( cat(1,St.Area) );
    B = round( cat(1,St.BoundingBox) );
    [~,Id] = sort( A, 1, 'descend' );
    Zl = B(Id(1),3);
    Zu = Zl + B(Id(1),6);
    H = ones( size(S) );
    H(:,:,Zl:Zu) = 0;
    Id = logical( (S==2) .* H );
    S(Id) = 0;
end
