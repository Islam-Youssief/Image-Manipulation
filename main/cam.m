function varargout = cam(varargin)
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @cam_OpeningFcn, ...
                   'gui_OutputFcn',  @cam_OutputFcn, ...
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


function cam_OpeningFcn(hObject, eventdata, handles, varargin)

handles.output = hObject;
guidata(hObject, handles);

function varargout = cam_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;


function pushbutton1_Callback(hObject, eventdata, handles)
vid = videoinput('winvideo', 1);
set(vid, 'ReturnedColorSpace', 'RGB');
img = getsnapshot(vid);
imshow(img)
imwrite(img,'E:\\myfirstimage.jpg'); 


function pushbutton2_Callback(hObject, eventdata, handles)
close(gcbf)


function pushbutton3_Callback(hObject, eventdata, handles)
%READ AN 2D IMAGE
A=imread('E:\\myfirstimage.jpg');
b=im2double(A);
[m,n]=size(A);

%PAD THE MATRIX WITH ZEROS ON ALL SIDES
modifyA=zeros(size(A)+2);
B=zeros(size(A));

%COPY THE ORIGINAL IMAGE MATRIX TO THE PADDED MATRIX
        for x=1:size(A,1)
            for y=1:size(A,2)
                modifyA(x+1,y+1)=A(x,y);
            end
        end
      %LET THE WINDOW BE AN ARRAY
      %STORE THE 3-by-3 NEIGHBOUR VALUES IN THE ARRAY
      %SORT AND FIND THE MIDDLE ELEMENT
       
for i= 1:size(modifyA,1)-2
    for j=1:size(modifyA,2)-2
        window=zeros(9,1);
        inc=1;
        for x=1:3
            for y=1:3
                window(inc)=modifyA(i+x-1,j+y-1);
                inc=inc+1;
            end
        end
       
        med=sort(window);
        %PLACE THE MEDIAN ELEMENT IN THE OUTPUT MATRIX
        B(i,j)=med(5);
       
    end
end
%CONVERT THE OUTPUT MATRIX TO 0-255 RANGE IMAGE TYPE
B=uint8(B);
imshow(B);



function pushbutton6_Callback(hObject, eventdata, handles)
rgb_image=imread('E:\\myfirstimage.jpg');
gray_image=0.2989 * rgb_image(:,:,1) + 0.5870 * rgb_image(:,:,2) + 0.1140 * rgb_image(:,:,3); 
imshow(gray_image)

% Sobel mask
function pushbutton7_Callback(hObject, eventdata, handles)
A=imread('E:\\myfirstimage.jpg');
B=rgb2gray(A);
C=double(B);
for i=1:size(C,1)-2
    for j=1:size(C,2)-2
        %Sobel mask for x-direction:
        Gx=((2*C(i+2,j+1)+C(i+2,j)+C(i+2,j+2))-(2*C(i,j+1)+C(i,j)+C(i,j+2)));
        %Sobel mask for y-direction:
        Gy=((2*C(i+1,j+2)+C(i,j+2)+C(i+2,j+2))-(2*C(i+1,j)+C(i,j)+C(i+2,j)));
       
        %The gradient of the image
        %B(i,j)=abs(Gx)+abs(Gy);
        B(i,j)=sqrt(Gx.^2+Gy.^2);
       
    end
end
imshow(B);

function pushbutton8_Callback(hObject, eventdata, handles)
i=imread('E:\\myfirstimage.jpg');
[r,c]=size(i);
val=0:255;
scale=0:255;
for x=1:r        % on rows
  for y=1:c      % on cols
     val(i(x,y)+1)=val(i(x,y)+1)+1;
   end
end
stem(scale,val);

function pushbutton9_Callback(hObject, eventdata, handles)
old_freqs=zeros(256,1);
new_freqs=zeros(256,1);
a=80; b=256;        %small range
c=0; d=256;          %big range

%initializing the given histogram table 
for i=1:256
    old_freqs(i)=i;
end
%initializing the new histogram table with new frequencies
for j=1:256 
    if(old_freqs(j)<a)
        new_freqs(j)=c;
    end
    if(old_freqs(j)>=a||old_freqs(j)<=b)
        new_freqs(j)=((d-c)\(b-a))*(old_freqs(j)-a)+c;
    end
    if(old_freqs(j)>b)
        new_freqs(j)=d;
    end
end
stem(new_freqs);


function pushbutton10_Callback(hObject, eventdata, handles)
o = imread ('E:\\myfirstimage.jpg');
w = size (o, 2);
 samplesHalf = floor(w / 2);
 samplesQuarter = floor(w / 4);
 samplesEighth = floor(w / 8);
 ci2 = [];
 ci4 = [];
 ci8 = [];

for k=1:3% all color layers: RGB
    for i=1:size(o, 1)% all rows
 
rowDCT = dct(double(o(i,:,k)));
 
ci2(i,:,k) = idct(rowDCT(1:samplesHalf), w);
 
ci4(i,:,k) = idct(rowDCT(1:samplesQuarter), w);
 
ci8(i,:,k) = idct(rowDCT(1:samplesEighth), w);
 
     end
 end
h = size(o, 1);
samplesHalf = floor(h / 2);
 samplesQuarter = floor(h / 4);
 samplesEighth = floor(h / 8);
 ci2f = [];
 ci4f = [];
 ci8f = [];
for k=1:3% all color layers: RGB
 
    for i=1:size(o, 2)% all columns
    
   columnDCT2=dct(double(ci2(:,i,k)));
 
   columnDCT4=dct(double(ci4(:,i,k)));
 
   columnDCT8=dct(double(ci8(:,i,k)));
 
ci2f(:,i,k) = idct(columnDCT2(1:samplesHalf), h);
 
ci4f(:,i,k) = idct(columnDCT4(1:samplesQuarter), h);
 
ci8f(:,i,k) = idct(columnDCT8(1:samplesEighth), h);
 
      end
 
end


function pushbutton18_Callback(hObject, eventdata, handles)


function pushbutton19_Callback(hObject, eventdata, handles)


function pushbutton20_Callback(hObject, eventdata, handles)

function pushbutton21_Callback(hObject, eventdata, handles)


function pushbutton22_Callback(hObject, eventdata, handles)


function pushbutton23_Callback(hObject, eventdata, handles)


function pushbutton24_Callback(hObject, eventdata, handles)
o = imread ('E:\\myfirstimage.jpg');
w = size (o, 2);
 
samplesHalf = floor(w / 2);
 
samplesQuarter = floor(w / 4);
 
samplesEighth = floor(w / 8);
 
ci2 = [];
 
ci4 = [];
 
ci8 = [];

for k=1:3% all color layers: RGB
    for i=1:size(o, 1)% all rows
 
rowDCT = dct(double(o(i,:,k)));
 
ci2(i,:,k) = idct(rowDCT(1:samplesHalf), w);
 
ci4(i,:,k) = idct(rowDCT(1:samplesQuarter), w);
 
ci8(i,:,k) = idct(rowDCT(1:samplesEighth), w);
 
     end
 
end
figure
subplot(2,2,1), image(uint8(o)), title('Original Image');
subplot(2,2,2), image(uint8(ci2)), title('Compression Factor 2*2');
subplot(2,2,3), image(uint8(ci4)), title('Compression Factor 4*4');
subplot(2,2,4), image(uint8(ci8)), title('Compression Factor 8*8');


function pushbutton25_Callback(hObject, eventdata, handles)


function pushbutton26_Callback(hObject, eventdata, handles)


function pushbutton27_Callback(hObject, eventdata, handles)


function pushbutton28_Callback(hObject, eventdata, handles)


function pushbutton29_Callback(hObject, eventdata, handles)


function pushbutton30_Callback(hObject, eventdata, handles)

function pushbutton31_Callback(hObject, eventdata, handles)






function pushbutton16_Callback(hObject, eventdata, handles)

MyImage=imread('E:\\myfirstimage.jpg');
% get the # of pixels in the image
% size(MyImage,1)-----> # of rows 
% size(MyImage,2)-----> # of columns

doubleMyImagea=double(MyImage);
 

numofpixels=size(MyImage,1)*size(MyImage,2);
UnsignedInteger8 =uint8(zeros(size(MyImage,1),size(MyImage,2)));
frequancy=zeros(256,1);  % this means 256 row ,1 column 
probablityOfPixels=zeros(256,1);
for i=1:size(MyImage,1) % pass through the rows 

    for j=1:size(MyImage,2) % pass through the 1 

        % get the pixels value by value 
        value=MyImage(i,j);
        frequancy(value+1)=frequancy(value+1)+1;
        probablityOfPixels(value+1)=frequancy(value+1)/numofpixels;
    end
    
end

probc=zeros(256,1); 
sum=0;
flag=255;
cum=zeros(256,1);
output=zeros(256,1);
for i=1:size(probablityOfPixels) %size(X) returns the sizes of each dimension of array X 

   sum=sum+frequancy(i);

   cum(i)=sum;  %As a value for example 36 then get the next value 

   probc(i)=cum(i)/numofpixels;  % As a probability as for example 0.36  Not 36 

   output(i)=round(  probc(i)  * flag  ); % this is the value we need   0.036 is close to  3 

end

% rows and clumns 
% get the new value
for i=1:size(MyImage,1)
    for j=1:size(MyImage,2)

      UnsignedInteger8(i,j)=output(MyImage(i,j)+1);

    end
end

doubleUnsigned=double(UnsignedInteger8);
hist(doubleUnsigned) ;
% subplot(3,2,4),surf(UnsignedInteger8) ;
% subplot(3,2,6),imshow(UnsignedInteger8);


% --- Executes on button press in pushbutton32.
function pushbutton32_Callback(hObject, eventdata, handles)
imshow('E:\\myfirstimage.jpg');


% --- Executes on button press in pushbutton44.
function pushbutton44_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton44 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton45.
function pushbutton45_Callback(hObject, eventdata, handles)


I=imread('E:\\myfirstimage.jpg');
I=rgb2gray(I);
d0=30;
d1=120;
n=4;
% Butterworth Bandpass Filter
% This simple  function was written for my Digital Image Processing course
% at Eastern Mediterranean University taught by
% Assoc. Prof. Dr. Hasan Demirel
% for the 2010-2011 Spring Semester
% for the complete report:
% http://www.scribd.com/doc/51981950/HW4-Frequency-Domain-Bandpass-Filtering
% 
% Written By:
% Leonardo O. Iheme (leonardo.iheme@cc.emu.edu.tr)
% 23rd of March 2011
% 
%   I = The input grey scale image
%   d0 = Lower cut off frequency
%   d1 = Higher cut off frequency
%   n = order of the filter
% 
% The function makes use of the simple principle that a bandpass filter
% can be obtained by multiplying a lowpass filter with a highpass filter
% where the lowpass filter has a higher cut off frquency than the high pass filter.
% 
% Usage BUTTERWORTHBPF(I,DO,D1,N)
% Example

f = double(I);
[nx ny] = size(f);
f = uint8(f);
fftI = fft2(f,2*nx-1,2*ny-1);
fftI = fftshift(fftI);

% 
% function fftshow(fftI,'log')
% Usage: FFTSHOW(F,TYPE)
%
% Displays the fft matrix F using imshow, where TYPE must be one of
% 'abs' or 'log'. If TYPE='abs', then then abs(f) is displayed; if
% TYPE='log' then log(1+abs(f)) is displayed. If TYPE is omitted, then
% 'log' is chosen as a default.
%
% Example:
% c=imread('cameraman.tif');
% cf=fftshift(fft2(c));
% fftshow(cf,'abs')
%



title('Image in Fourier Domain')
% Initialize filter.
filter1 = ones(2*nx-1,2*ny-1);
filter2 = ones(2*nx-1,2*ny-1);
filter3 = ones(2*nx-1,2*ny-1);

for i = 1:2*nx-1
    for j =1:2*ny-1
        dist = ((i-(nx+1))^2 + (j-(ny+1))^2)^.5;
        % Create Butterworth filter.
        filter1(i,j)= 1/(1 + (dist/d1)^(2*n));
        filter2(i,j) = 1/(1 + (dist/d0)^(2*n));
        filter3(i,j)= 1.0 - filter2(i,j);
        filter3(i,j) = filter1(i,j).*filter3(i,j);
    end
end
% Update image with passed frequencies.
filtered_image = fftI + filter3.*fftI;



if nargin<2,'log'=='log';
end
if ('log'=='log')
fl = log(1+abs(filter3));
fm = max(fl(:));
% imshow(im2uint8(fl/fm))
elseif ('log'=='abs')
fa=abs(filter3);
fm=max(fa(:));
% imshow(fa/fm)
else
error('TYPE must be abs or log.');
end

filtered_image = ifftshift(filtered_image);
filtered_image = ifft2(filtered_image,2*nx-1,2*ny-1);
filtered_image = real(filtered_image(1:nx,1:ny));
filtered_image = uint8(filtered_image);
imshow(filtered_image,[])


% hObject    handle to pushbutton45 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)








% --- Executes on button press in pushbutton71.


function H = lpfilter(type, M, N, D0, n)

[U, V] = dftuv(M, N);

D = sqrt(U.^2 + V.^2);

switch type
case 'ideal'
   H = double(D <=D0);
case 'btw'
   if nargin == 4
      n = 1;
   end
   H = 1./(1 + (D./D0).^(2*n));
case 'gaussian'
   H = exp(-(D.^2)./(2*(D0^2)));
otherwise
   error('Unknown filter type.')
end


function PQ = paddedsize(AB, CD, PARAM)
if nargin == 1
   PQ = 2*AB;
elseif nargin == 2 & ~ischar(CD)
   PQ = AB + CD - 1;
   PQ = 2 * ceil(PQ / 2);
elseif nargin == 2
   m = max(AB); % Maximum dimension.
   P = 2^nextpow2(2*m);
   PQ = [P, P];
elseif nargin == 3
   m = max([AB CD]); %Maximum dimension.
   P = 2^nextpow2(2*m);
   PQ = [P, P];
else
   error('Wrong number of inputs.')
end

function H = notch(type, M, N, D0, x, y, n)
if nargin == 6
   n = 1; % Default value of n.
end
Hlp = lpfilter(type, M, N, D0, n);
H = 1 - Hlp;
H = circshift(H, [y-1 x-1]);

function [U, V] = dftuv(M, N)
% Set up range of variables.
u = 0:(M-1);
v = 0:(N-1);
% Compute the indices for use in meshgrid
idx = find(u > M/2);
u(idx) = u(idx) - M;
idy = find(v > N/2);
v(idy) = v(idy) - N;
% Compute the meshgrid arrays
[V, U] = meshgrid(v, u);
% --- Executes on button press in pushbutton46.
function pushbutton46_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton46 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% hObject    handle to pushbutton71 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
footBalls=imread('E:\myfirstimage.jpg');
footBall=rgb2gray(footBalls);
%Determine good padding for Fourier transform
PQ = paddedsize(size(footBall));
%Create Notch filters corresponding to extra peaks in the Fourier transform
H1 = notch('btw', PQ(1), PQ(2), 10, 50, 100);
H2 = notch('btw', PQ(1), PQ(2), 10, 1, 400);
H3 = notch('btw', PQ(1), PQ(2), 10, 620, 100);
H4 = notch('btw', PQ(1), PQ(2), 10, 22, 414);
H5 = notch('btw', PQ(1), PQ(2), 10, 592, 414);
H6 = notch('btw', PQ(1), PQ(2), 10, 1, 114);
% Calculate the discrete Fourier transform of the image
F=fft2(double(footBall),PQ(1),PQ(2));
% Apply the notch filters to the Fourier spectrum of the image
FS_football = F.*H1.*H2.*H3.*H4.*H5.*H6;
% convert the result to the spacial domain.
F_football=real(ifft2(FS_football)); 
% Crop the image to undo padding
F_football=F_football(1:size(footBall,1), 1:size(footBall,2));
%Display the blurred image
imshow(F_football,[])
% Display the Fourier Spectrum 
% Move the origin of the transform to the center of the frequency rectangle.
Fc=fftshift(F); 
Fcf=fftshift(FS_football);
% use abs to compute the magnitude and use log to brighten display
S1=log(1+abs(Fc)); 
S2=log(1+abs(Fcf));
% figure, imshow(S1,[])
% figure, imshow(S2,[])


% --- Executes on button press in pushbutton47.



function PQ = paddedsize(AB, CD, PARAM)
%PADDEDSIZE Computes padded sizes useful for FFT-based filtering.
%   PQ = PADDEDSIZE(AB), where AB is a two-element size vector,
%   computes the two-element size vector PQ = 2*AB.
%
%   PQ = PADDEDSIZE(AB, 'PWR2') computes the vector PQ such that
%   PQ(1) = PQ(2) = 2^nextpow2(2*m), where m is MAX(AB).
%
%   PQ = PADDEDSIZE(AB, CD), where AB and CD are two-element size
%   vectors, computes the two-element size vector PQ.  The elements
%   of PQ are the smallest even integers greater than or equal to 
%   AB + CD -1.
%   
%   PQ = PADDEDSIZE(AB, CD, 'PWR2') computes the vector PQ such that
%   PQ(1) = PQ(2) = 2^nextpow2(2*m), where m is MAX([AB CD]).

if nargin == 1
   PQ = 2*AB;
elseif nargin == 2 & ~ischar(CD)
   PQ = AB + CD - 1;
   PQ = 2 * ceil(PQ / 2);
elseif nargin == 2
   m = max(AB); % Maximum dimension.

   % Find power-of-2 at least twice m.
   P = 2^nextpow2(2*m);
   PQ = [P, P];
elseif nargin == 3
   m = max([AB CD]); %Maximum dimension.
   P = 2^nextpow2(2*m);
   PQ = [P, P];
else
   error('Wrong number of inputs.')
end


function H = lpfilter(type, M, N, D0, n)
%LPFILTER Computes frequency domain lowpass filters
%   H = LPFILTER(TYPE, M, N, D0, n) creates the transfer function of
%   a lowpass filter, H, of the specified TYPE and size (M-by-N).  To
%   view the filter as an image or mesh plot, it should be centered
%   using H = fftshift(H).
%
%   Valid values for TYPE, D0, and n are:
%
%   'ideal'    Ideal lowpass filter with cutoff frequency D0.  n need
%              not be supplied.  D0 must be positive
%
%   'btw'      Butterworth lowpass filter of order n, and cutoff D0.
%              The default value for n is 1.0.  D0 must be positive.
%
%   'gaussian' Gaussian lowpass filter with cutoff (standard deviation)
%              D0.  n need not be supplied.  D0 must be positive.

% Use function dftuv to set up the meshgrid arrays needed for 
% computing the required distances.
[U, V] = dftuv(M, N);

% Compute the distances D(U, V).
D = sqrt(U.^2 + V.^2);

% Begin fiter computations.
switch type
case 'ideal'
   H = double(D <=D0);
case 'btw'
   if nargin == 4
      n = 1;
   end
   H = 1./(1 + (D./D0).^(2*n));
case 'gaussian'
   H = exp(-(D.^2)./(2*(D0^2)));
otherwise
   error('Unknown filter type.')
end


function H = hpfilter(type, M, N, D0, n)
%HPFILTER Computes frequency domain highpass filters
%   H = HPFILTER(TYPE, M, N, D0, n) creates the transfer function of
%   a highpass filter, H, of the specified TYPE and size (M-by-N).
%   Valid values for TYPE, D0, and n are:
%
%   'ideal'     Ideal highpass filter with cutoff frequency D0.  n
%               need not be supplied.  D0 must be positive
%
%   'btw'       Butterworth highpass filter of order n, and cutoff D0.
%               The default value for n is 1.0.  D0 must be positive.
%
%   'gaussian'  Gaussian highpass filter with cutoff (standard deviation)
%               D0.  n need not be supplied.  D0 must be positive.
% 

% The transfer function Hhp of a highpass filter is 1 - Hlp,
% where Hlp is the transfer function of the corresponding lowpass
% filter.  Thus, we can use function lpfilter to generate highpass
% filters.

if nargin == 4
   n = 1; % Default value of n.
end

% Generate highpass filter.
Hlp = lpfilter(type, M, N, D0, n);
H = 1 - Hlp;

function [U, V] = dftuv(M, N)
%DFTUV Computes meshgrid frequency matrices.
%   [U, V] = DFTUV(M, N) computes meshgrid frequency matrices U and
%   V. U and V are useful for computing frequency-domain filter 
%   functions that can be used with DFTFILT.  U and V are both M-by-N.

% Set up range of variables.
u = 0:(M-1);
v = 0:(N-1);

% Compute the indices for use in meshgrid
idx = find(u > M/2);
u(idx) = u(idx) - M;
idy = find(v > N/2);
v(idy) = v(idy) - N;

% Compute the meshgrid arrays
[V, U] = meshgrid(v, u);

function pushbutton47_Callback(hObject, eventdata, handles)

% hObject    handle to pushbutton72 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
footBall=imread('E:\myfirstimage.jpg');
%Convert to grayscale
footBall=rgb2gray(footBall); 
% imshow(footBall)
%Determine good padding for Fourier transform
PQ = paddedsize(size(footBall));
%Create a Gaussian Highpass filter 5% the width of the Fourier transform
D0 = 0.05*PQ(1);
H = hpfilter('gaussian', PQ(1), PQ(2), D0);
% Calculate the discrete Fourier transform of the image
F=fft2(double(footBall),size(H,1),size(H,2));
% Apply the highpass filter to the Fourier spectrum of the image
HPFS_football = H.*F;
% convert the result to the spacial domain.
HPF_football=real(ifft2(HPFS_football)); 
% Crop the image to undo padding
HPF_football=HPF_football(1:size(footBall,1), 1:size(footBall,2));
%Display the "Sharpened" image
imshow(HPF_football, [])
% Display the Fourier Spectrum
% Move the origin of the transform to the center of the frequency 

Fc=fftshift(F);
Fcf=fftshift(HPFS_football);
% use abs to compute the magnitude and use
S1=log(1+abs(Fc)); 
S2=log(1+abs(Fcf));
% figure, imshow(S1,[])
% figure, imshow(S2,[])








% --- Executes on button press in pushbutton48.
function pushbutton48_Callback(hObject, eventdata, handles)

% Demo macro to filter an image in the Foureir domain.
  % Erase all existing variables. Or clearvars if you want.
format long g;
format compact;
fontSize = 20;
% Read in demo image.
folder = 'E:\';
% folder = fullfile(matlabroot, '\toolbox\images\imdemos');
baseFileName = 'myfirstimage.jpg';
% Get the full filename, with path prepended.
fullFileName = fullfile(folder, baseFileName);
% Check if file exists.
if ~exist(fullFileName, 'file')
	% File doesn't exist -- didn't find it there.  Check the search path for it.
	fullFileName = baseFileName; % No path this time.
	if ~exist(fullFileName, 'file')
		% Still didn't find it.  Alert user.
		errorMessage = sprintf('Error: %s does not exist in the search path folders.', fullFileName);
		uiwait(warndlg(errorMessage));
		return;
	end
end
grayImage = imread(fullFileName);
[rows columns numberOfColorBands] = size(grayImage);
if numberOfColorBands > 1
	grayImage = rgb2gray(grayImage);
end

 grayImage = double(grayImage);
 frequencyImage = fftshift(fft2(grayImage));
 amplitudeImage = log(abs(frequencyImage));
 minValue = min(min(amplitudeImage))
 maxValue = max(max(amplitudeImage))

[midpointX, midpointY] = find(amplitudeImage == maxValue)
filterWindowHalfWidth = 1;
for row = midpointY-filterWindowHalfWidth:midpointY+filterWindowHalfWidth
	for column = midpointX-filterWindowHalfWidth:midpointX+filterWindowHalfWidth
		frequencyImage(row-4, column+7) = 0;
		frequencyImage(row+4, column-7) = 0;
	end
end
amplitudeImage2 = log(abs(frequencyImage));
minValue = min(min(amplitudeImage2));
maxValue = max(max(amplitudeImage2));
% subplot(2,2, 1);
% imshow(amplitudeImage2, [minValue maxValue]);
% title('Two dots zeroed out', 'FontSize', fontSize);
% zoom(10)
filteredImage = ifft2(fftshift(frequencyImage));
amplitudeImage3 = abs(filteredImage);
minValue = min(min(amplitudeImage3));
maxValue = max(max(amplitudeImage3));
% subplot(2,2, 2);
imshow(amplitudeImage3, [minValue maxValue]);
clc
% zoom(10)

% hObject    handle to pushbutton73 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton74.

function [U, V] = dftuv(M, N)
%DFTUV Computes meshgrid frequency matrices.
%   [U, V] = DFTUV(M, N) computes meshgrid frequency matrices U and
%   V. U and V are useful for computing frequency-domain filter 
%   functions that can be used with DFTFILT.  U and V are both M-by-N.

% Set up range of variables.
u = 0:(M-1);
v = 0:(N-1);

% Compute the indices for use in meshgrid
idx = find(u > M/2);
u(idx) = u(idx) - M;
idy = find(v > N/2);
v(idy) = v(idy) - N;

% Compute the meshgrid arrays
[V, U] = meshgrid(v, u);

function H = lpfilter(type, M, N, D0, n)
%LPFILTER Computes frequency domain lowpass filters
%   H = LPFILTER(TYPE, M, N, D0, n) creates the transfer function of
%   a lowpass filter, H, of the specified TYPE and size (M-by-N).  To
%   view the filter as an image or mesh plot, it should be centered
%   using H = fftshift(H).
%
%   Valid values for TYPE, D0, and n are:
%
%   'ideal'    Ideal lowpass filter with cutoff frequency D0.  n need
%              not be supplied.  D0 must be positive
%
%   'btw'      Butterworth lowpass filter of order n, and cutoff D0.
%              The default value for n is 1.0.  D0 must be positive.
%
%   'gaussian' Gaussian lowpass filter with cutoff (standard deviation)
%              D0.  n need not be supplied.  D0 must be positive.

% Use function dftuv to set up the meshgrid arrays needed for 
% computing the required distances.
[U, V] = dftuv(M, N);

% Compute the distances D(U, V).
D = sqrt(U.^2 + V.^2);

% Begin fiter computations.
switch type
case 'ideal'
   H = double(D <=D0);
case 'btw'
   if nargin == 4
      n = 1;
   end
   H = 1./(1 + (D./D0).^(2*n));
case 'gaussian'
   H = exp(-(D.^2)./(2*(D0^2)));
otherwise
   error('Unknown filter type.')
end

function PQ = paddedsize(AB, CD, PARAM)
%PADDEDSIZE Computes padded sizes useful for FFT-based filtering.
%   PQ = PADDEDSIZE(AB), where AB is a two-element size vector,
%   computes the two-element size vector PQ = 2*AB.
%
%   PQ = PADDEDSIZE(AB, 'PWR2') computes the vector PQ such that
%   PQ(1) = PQ(2) = 2^nextpow2(2*m), where m is MAX(AB).
%
%   PQ = PADDEDSIZE(AB, CD), where AB and CD are two-element size
%   vectors, computes the two-element size vector PQ.  The elements
%   of PQ are the smallest even integers greater than or equal to 
%   AB + CD -1.
%   
%   PQ = PADDEDSIZE(AB, CD, 'PWR2') computes the vector PQ such that
%   PQ(1) = PQ(2) = 2^nextpow2(2*m), where m is MAX([AB CD]).

if nargin == 1
   PQ = 2*AB;
elseif nargin == 2 & ~ischar(CD)
   PQ = AB + CD - 1;
   PQ = 2 * ceil(PQ / 2);
elseif nargin == 2
   m = max(AB); % Maximum dimension.

   % Find power-of-2 at least twice m.
   P = 2^nextpow2(2*m);
   PQ = [P, P];
elseif nargin == 3
   m = max([AB CD]); %Maximum dimension.
   P = 2^nextpow2(2*m);
   PQ = [P, P];
else
   error('Wrong number of inputs.')
end

% --- Executes on button press in pushbutton49.
function pushbutton49_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton49 (see GCBO)
footBall=imread('E:\\myfirstimage.jpg');
%Convert to grayscale
footBall=rgb2gray(footBall); 
%imshow(footBall)

%Determine good padding for Fourier transform
PQ = paddedsize(size(footBall));

%Create a Gaussian Lowpass filter 5% the width of the Fourier transform
D0 = 0.05*PQ(1);
H = lpfilter('gaussian', PQ(1), PQ(2), D0);

% Calculate the discrete Fourier transform of the image
F=fft2(double(footBall),size(H,1),size(H,2));

% Apply the highpass filter to the Fourier spectrum of the image
LPFS_football = H.*F;

% convert the result to the spacial domain.
LPF_football=real(ifft2(LPFS_football)); 

% Crop the image to undo padding
LPF_football=LPF_football(1:size(footBall,1), 1:size(footBall,2));

%Display the blurred image
imshow(LPF_football, [])

% Display the Fourier Spectrum 
% Move the origin of the transform to the center of the frequency rectangle.
Fc=fftshift(F);
Fcf=fftshift(LPFS_football);
% use abs to compute the magnitude and use log to brighten display
S1=log(1+abs(Fc)); 
S2=log(1+abs(Fcf));
%figure, imshow(S1,[])
%figure, imshow(S2,[])

% --- Executes on button press in pushbutton50.
function pushbutton50_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton50 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton51.
function pushbutton51_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton51 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton52.
function pushbutton52_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton52 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton53.
function pushbutton53_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton53 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton54.
function pushbutton54_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton54 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton55.
function pushbutton55_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton55 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton56.
function pushbutton56_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton56 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton57.
function pushbutton57_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton57 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton58.
function pushbutton58_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton58 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton59.
function pushbutton59_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton59 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton60.
function pushbutton60_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton60 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton61.
function pushbutton61_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton61 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton62.
function pushbutton62_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton62 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton70.

function pushbutton70_Callback(hObject, eventdata, handles)

I=imread('E:\\myfirstimage.jpg');
I=rgb2gray(I);
d0=30;
d1=120;
n=4;
% Butterworth Bandpass Filter
% This simple  function was written for my Digital Image Processing course
% at Eastern Mediterranean University taught by
% Assoc. Prof. Dr. Hasan Demirel
% for the 2010-2011 Spring Semester
% for the complete report:
% http://www.scribd.com/doc/51981950/HW4-Frequency-Domain-Bandpass-Filtering
% 
% Written By:
% Leonardo O. Iheme (leonardo.iheme@cc.emu.edu.tr)
% 23rd of March 2011
% 
%   I = The input grey scale image
%   d0 = Lower cut off frequency
%   d1 = Higher cut off frequency
%   n = order of the filter
% 
% The function makes use of the simple principle that a bandpass filter
% can be obtained by multiplying a lowpass filter with a highpass filter
% where the lowpass filter has a higher cut off frquency than the high pass filter.
% 
% Usage BUTTERWORTHBPF(I,DO,D1,N)
% Example

f = double(I);
[nx ny] = size(f);
f = uint8(f);
fftI = fft2(f,2*nx-1,2*ny-1);
fftI = fftshift(fftI);

% 
% function fftshow(fftI,'log')
% Usage: FFTSHOW(F,TYPE)
%
% Displays the fft matrix F using imshow, where TYPE must be one of
% 'abs' or 'log'. If TYPE='abs', then then abs(f) is displayed; if
% TYPE='log' then log(1+abs(f)) is displayed. If TYPE is omitted, then
% 'log' is chosen as a default.
%
% Example:
% c=imread('cameraman.tif');
% cf=fftshift(fft2(c));
% fftshow(cf,'abs')
%
               

if nargin<2,'log'=='log';
end
if ('log'=='log')
fl = log(1+abs(fftI));
fm = max(fl(:));
imshow(im2uint8(fl/fm))
elseif ('log'=='abs')
fa=abs(fftI);
fm=max(fa(:));
imshow(fa/fm)
else
error('TYPE must be abs or log.');
end;



% Initialize filter.
filter1 = ones(2*nx-1,2*ny-1);
filter2 = ones(2*nx-1,2*ny-1);
filter3 = ones(2*nx-1,2*ny-1);

for i = 1:2*nx-1
    for j =1:2*ny-1
        dist = ((i-(nx+1))^2 + (j-(ny+1))^2)^.5;
        % Create Butterworth filter.
        filter1(i,j)= 1/(1 + (dist/d1)^(2*n));
        filter2(i,j) = 1/(1 + (dist/d0)^(2*n));
        filter3(i,j)= 1.0 - filter2(i,j);
        filter3(i,j) = filter1(i,j).*filter3(i,j);
    end
end
% Update image with passed frequencies.
filtered_image = fftI + filter3.*fftI;



if nargin<2,'log'=='log';
end
if ('log'=='log')
fl = log(1+abs(filter3));
fm = max(fl(:));
% figure,imshow(im2uint8(fl/fm))
elseif ('log'=='abs')
fa=abs(filter3);
fm=max(fa(:));
% figure,imshow(fa/fm)
else
error('TYPE must be abs or log.');
end


filtered_image = ifftshift(filtered_image);
filtered_image = ifft2(filtered_image,2*nx-1,2*ny-1);
filtered_image = real(filtered_image(1:nx,1:ny));
filtered_image = uint8(filtered_image);
% end


% --- Executes on button press in pushbutton71.


function H = lpfilter(type, M, N, D0, n)

[U, V] = dftuv(M, N);

D = sqrt(U.^2 + V.^2);

switch type
case 'ideal'
   H = double(D <=D0);
case 'btw'
   if nargin == 4
      n = 1;
   end
   H = 1./(1 + (D./D0).^(2*n));
case 'gaussian'
   H = exp(-(D.^2)./(2*(D0^2)));
otherwise
   error('Unknown filter type.')
end


function PQ = paddedsize(AB, CD, PARAM)
if nargin == 1
   PQ = 2*AB;
elseif nargin == 2 & ~ischar(CD)
   PQ = AB + CD - 1;
   PQ = 2 * ceil(PQ / 2);
elseif nargin == 2
   m = max(AB); % Maximum dimension.
   P = 2^nextpow2(2*m);
   PQ = [P, P];
elseif nargin == 3
   m = max([AB CD]); %Maximum dimension.
   P = 2^nextpow2(2*m);
   PQ = [P, P];
else
   error('Wrong number of inputs.')
end

function H = notch(type, M, N, D0, x, y, n)
if nargin == 6
   n = 1; % Default value of n.
end
Hlp = lpfilter(type, M, N, D0, n);
H = 1 - Hlp;
H = circshift(H, [y-1 x-1]);

function [U, V] = dftuv(M, N)
% Set up range of variables.
u = 0:(M-1);
v = 0:(N-1);
% Compute the indices for use in meshgrid
idx = find(u > M/2);
u(idx) = u(idx) - M;
idy = find(v > N/2);
v(idy) = v(idy) - N;
% Compute the meshgrid arrays
[V, U] = meshgrid(v, u);

function pushbutton71_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton71 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

footBalls=imread('E:\myfirstimage.jpg');
footBall=rgb2gray(footBalls);
%Determine good padding for Fourier transform
PQ = paddedsize(size(footBall));
%Create Notch filters corresponding to extra peaks in the Fourier transform
H1 = notch('btw', PQ(1), PQ(2), 10, 50, 100);
H2 = notch('btw', PQ(1), PQ(2), 10, 1, 400);
H3 = notch('btw', PQ(1), PQ(2), 10, 620, 100);
H4 = notch('btw', PQ(1), PQ(2), 10, 22, 414);
H5 = notch('btw', PQ(1), PQ(2), 10, 592, 414);
H6 = notch('btw', PQ(1), PQ(2), 10, 1, 114);
% Calculate the discrete Fourier transform of the image
F=fft2(double(footBall),PQ(1),PQ(2));
% Apply the notch filters to the Fourier spectrum of the image
FS_football = F.*H1.*H2.*H3.*H4.*H5.*H6;
% convert the result to the spacial domain.
F_football=real(ifft2(FS_football)); 
% Crop the image to undo padding
F_football=F_football(1:size(footBall,1), 1:size(footBall,2));
%Display the blurred image
% figure, imshow(F_football,[])
% Display the Fourier Spectrum 
% Move the origin of the transform to the center of the frequency rectangle.
Fc=fftshift(F); 
Fcf=fftshift(FS_football);
% use abs to compute the magnitude and use log to brighten display
S1=log(1+abs(Fc)); 
S2=log(1+abs(Fcf));
imshow(S1,[])
figure, imshow(S2,[])
% --- Executes on button press in pushbutton72.



function PQ = paddedsize(AB, CD, PARAM)
%PADDEDSIZE Computes padded sizes useful for FFT-based filtering.
%   PQ = PADDEDSIZE(AB), where AB is a two-element size vector,
%   computes the two-element size vector PQ = 2*AB.
%
%   PQ = PADDEDSIZE(AB, 'PWR2') computes the vector PQ such that
%   PQ(1) = PQ(2) = 2^nextpow2(2*m), where m is MAX(AB).
%
%   PQ = PADDEDSIZE(AB, CD), where AB and CD are two-element size
%   vectors, computes the two-element size vector PQ.  The elements
%   of PQ are the smallest even integers greater than or equal to 
%   AB + CD -1.
%   
%   PQ = PADDEDSIZE(AB, CD, 'PWR2') computes the vector PQ such that
%   PQ(1) = PQ(2) = 2^nextpow2(2*m), where m is MAX([AB CD]).

if nargin == 1
   PQ = 2*AB;
elseif nargin == 2 & ~ischar(CD)
   PQ = AB + CD - 1;
   PQ = 2 * ceil(PQ / 2);
elseif nargin == 2
   m = max(AB); % Maximum dimension.

   % Find power-of-2 at least twice m.
   P = 2^nextpow2(2*m);
   PQ = [P, P];
elseif nargin == 3
   m = max([AB CD]); %Maximum dimension.
   P = 2^nextpow2(2*m);
   PQ = [P, P];
else
   error('Wrong number of inputs.')
end


function H = lpfilter(type, M, N, D0, n)
%LPFILTER Computes frequency domain lowpass filters
%   H = LPFILTER(TYPE, M, N, D0, n) creates the transfer function of
%   a lowpass filter, H, of the specified TYPE and size (M-by-N).  To
%   view the filter as an image or mesh plot, it should be centered
%   using H = fftshift(H).
%
%   Valid values for TYPE, D0, and n are:
%
%   'ideal'    Ideal lowpass filter with cutoff frequency D0.  n need
%              not be supplied.  D0 must be positive
%
%   'btw'      Butterworth lowpass filter of order n, and cutoff D0.
%              The default value for n is 1.0.  D0 must be positive.
%
%   'gaussian' Gaussian lowpass filter with cutoff (standard deviation)
%              D0.  n need not be supplied.  D0 must be positive.

% Use function dftuv to set up the meshgrid arrays needed for 
% computing the required distances.
[U, V] = dftuv(M, N);

% Compute the distances D(U, V).
D = sqrt(U.^2 + V.^2);

% Begin fiter computations.
switch type
case 'ideal'
   H = double(D <=D0);
case 'btw'
   if nargin == 4
      n = 1;
   end
   H = 1./(1 + (D./D0).^(2*n));
case 'gaussian'
   H = exp(-(D.^2)./(2*(D0^2)));
otherwise
   error('Unknown filter type.')
end


function H = hpfilter(type, M, N, D0, n)
%HPFILTER Computes frequency domain highpass filters
%   H = HPFILTER(TYPE, M, N, D0, n) creates the transfer function of
%   a highpass filter, H, of the specified TYPE and size (M-by-N).
%   Valid values for TYPE, D0, and n are:
%
%   'ideal'     Ideal highpass filter with cutoff frequency D0.  n
%               need not be supplied.  D0 must be positive
%
%   'btw'       Butterworth highpass filter of order n, and cutoff D0.
%               The default value for n is 1.0.  D0 must be positive.
%
%   'gaussian'  Gaussian highpass filter with cutoff (standard deviation)
%               D0.  n need not be supplied.  D0 must be positive.
% 

% The transfer function Hhp of a highpass filter is 1 - Hlp,
% where Hlp is the transfer function of the corresponding lowpass
% filter.  Thus, we can use function lpfilter to generate highpass
% filters.

if nargin == 4
   n = 1; % Default value of n.
end

% Generate highpass filter.
Hlp = lpfilter(type, M, N, D0, n);
H = 1 - Hlp;

function [U, V] = dftuv(M, N)
%DFTUV Computes meshgrid frequency matrices.
%   [U, V] = DFTUV(M, N) computes meshgrid frequency matrices U and
%   V. U and V are useful for computing frequency-domain filter 
%   functions that can be used with DFTFILT.  U and V are both M-by-N.

% Set up range of variables.
u = 0:(M-1);
v = 0:(N-1);

% Compute the indices for use in meshgrid
idx = find(u > M/2);
u(idx) = u(idx) - M;
idy = find(v > N/2);
v(idy) = v(idy) - N;

% Compute the meshgrid arrays
[V, U] = meshgrid(v, u);
function pushbutton72_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton72 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
footBall=imread('E:\myfirstimage.jpg');
%Convert to grayscale
footBall=rgb2gray(footBall); 
% imshow(footBall)
%Determine good padding for Fourier transform
PQ = paddedsize(size(footBall));
%Create a Gaussian Highpass filter 5% the width of the Fourier transform
D0 = 0.05*PQ(1);
H = hpfilter('gaussian', PQ(1), PQ(2), D0);
% Calculate the discrete Fourier transform of the image
F=fft2(double(footBall),size(H,1),size(H,2));
% Apply the highpass filter to the Fourier spectrum of the image
HPFS_football = H.*F;
% convert the result to the spacial domain.
HPF_football=real(ifft2(HPFS_football)); 
% Crop the image to undo padding
HPF_football=HPF_football(1:size(footBall,1), 1:size(footBall,2));
%Display the "Sharpened" image
% figure, imshow(HPF_football, [])
% Display the Fourier Spectrum
% Move the origin of the transform to the center of the frequency 

Fc=fftshift(F);
Fcf=fftshift(HPFS_football);
% use abs to compute the magnitude and use
S1=log(1+abs(Fc)); 
S2=log(1+abs(Fcf));
% figure, imshow(S1,[])
imshow(S2,[])















function pushbutton73_Callback(hObject, eventdata, handles)
format long g;
format compact;
fontSize = 20;
% Read in demo image.
folder = 'E:\';
% folder = fullfile(matlabroot, '\toolbox\images\imdemos');
baseFileName = 'myfirstimage.jpg';
% Get the full filename, with path prepended.
fullFileName = fullfile(folder, baseFileName);
% Check if file exists.
if ~exist(fullFileName, 'file')
	% File doesn't exist -- didn't find it there.  Check the search path for it.
	fullFileName = baseFileName; % No path this time.
	if ~exist(fullFileName, 'file')
		% Still didn't find it.  Alert user.
		errorMessage = sprintf('Error: %s does not exist in the search path folders.', fullFileName);
		uiwait(warndlg(errorMessage));
		return;
	end
end
grayImage = imread(fullFileName);
[rows columns numberOfColorBands] = size(grayImage);
if numberOfColorBands > 1
	grayImage = rgb2gray(grayImage);
end
% subplot(2,2, 1);
% imshow(grayImage, [0 255]);
% set(gcf, 'units','normalized','outerposition',[0 0 1 1]); % Maximize figure.
grayImage = double(grayImage);
frequencyImage = fftshift(fft2(grayImage));
%  subplot(2,2, 2);
amplitudeImage = log(abs(frequencyImage));
minValue = min(min(amplitudeImage));
maxValue = max(max(amplitudeImage));
imshow(amplitudeImage, []);

% zoom(10)

% hObject    handle to pushbutton73 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton74.

function [U, V] = dftuv(M, N)
%DFTUV Computes meshgrid frequency matrices.
%   [U, V] = DFTUV(M, N) computes meshgrid frequency matrices U and
%   V. U and V are useful for computing frequency-domain filter 
%   functions that can be used with DFTFILT.  U and V are both M-by-N.

% Set up range of variables.
u = 0:(M-1);
v = 0:(N-1);

% Compute the indices for use in meshgrid
idx = find(u > M/2);
u(idx) = u(idx) - M;
idy = find(v > N/2);
v(idy) = v(idy) - N;

% Compute the meshgrid arrays
[V, U] = meshgrid(v, u);

function H = lpfilter(type, M, N, D0, n)
%LPFILTER Computes frequency domain lowpass filters
%   H = LPFILTER(TYPE, M, N, D0, n) creates the transfer function of
%   a lowpass filter, H, of the specified TYPE and size (M-by-N).  To
%   view the filter as an image or mesh plot, it should be centered
%   using H = fftshift(H).
%
%   Valid values for TYPE, D0, and n are:
%
%   'ideal'    Ideal lowpass filter with cutoff frequency D0.  n need
%              not be supplied.  D0 must be positive
%
%   'btw'      Butterworth lowpass filter of order n, and cutoff D0.
%              The default value for n is 1.0.  D0 must be positive.
%
%   'gaussian' Gaussian lowpass filter with cutoff (standard deviation)
%              D0.  n need not be supplied.  D0 must be positive.

% Use function dftuv to set up the meshgrid arrays needed for 
% computing the required distances.
[U, V] = dftuv(M, N);

% Compute the distances D(U, V).
D = sqrt(U.^2 + V.^2);

% Begin fiter computations.
switch type
case 'ideal'
   H = double(D <=D0);
case 'btw'
   if nargin == 4
      n = 1;
   end
   H = 1./(1 + (D./D0).^(2*n));
case 'gaussian'
   H = exp(-(D.^2)./(2*(D0^2)));
otherwise
   error('Unknown filter type.')
end

function PQ = paddedsize(AB, CD, PARAM)
%PADDEDSIZE Computes padded sizes useful for FFT-based filtering.
%   PQ = PADDEDSIZE(AB), where AB is a two-element size vector,
%   computes the two-element size vector PQ = 2*AB.
%
%   PQ = PADDEDSIZE(AB, 'PWR2') computes the vector PQ such that
%   PQ(1) = PQ(2) = 2^nextpow2(2*m), where m is MAX(AB).
%
%   PQ = PADDEDSIZE(AB, CD), where AB and CD are two-element size
%   vectors, computes the two-element size vector PQ.  The elements
%   of PQ are the smallest even integers greater than or equal to 
%   AB + CD -1.
%   
%   PQ = PADDEDSIZE(AB, CD, 'PWR2') computes the vector PQ such that
%   PQ(1) = PQ(2) = 2^nextpow2(2*m), where m is MAX([AB CD]).

if nargin == 1
   PQ = 2*AB;
elseif nargin == 2 & ~ischar(CD)
   PQ = AB + CD - 1;
   PQ = 2 * ceil(PQ / 2);
elseif nargin == 2
   m = max(AB); % Maximum dimension.

   % Find power-of-2 at least twice m.
   P = 2^nextpow2(2*m);
   PQ = [P, P];
elseif nargin == 3
   m = max([AB CD]); %Maximum dimension.
   P = 2^nextpow2(2*m);
   PQ = [P, P];
else
   error('Wrong number of inputs.')
end


function pushbutton74_Callback(hObject, eventdata, handles)
footBall=imread('E:\\myfirstimage.jpg');
%Convert to grayscale
footBall=rgb2gray(footBall); 
%imshow(footBall)

%Determine good padding for Fourier transform
PQ = paddedsize(size(footBall));

%Create a Gaussian Lowpass filter 5% the width of the Fourier transform
D0 = 0.05*PQ(1);
H = lpfilter('gaussian', PQ(1), PQ(2), D0);

% Calculate the discrete Fourier transform of the image
F=fft2(double(footBall),size(H,1),size(H,2));

% Apply the highpass filter to the Fourier spectrum of the image
LPFS_football = H.*F;

% convert the result to the spacial domain.
LPF_football=real(ifft2(LPFS_football)); 

% Crop the image to undo padding
LPF_football=LPF_football(1:size(footBall,1), 1:size(footBall,2));

%Display the blurred image
%figure, imshow(LPF_football, [])

% Display the Fourier Spectrum 
% Move the origin of the transform to the center of the frequency rectangle.
Fc=fftshift(F);
Fcf=fftshift(LPFS_football);
% use abs to compute the magnitude and use log to brighten display
S1=log(1+abs(Fc)); 
S2=log(1+abs(Fcf));
imshow(S1,[])
%figure, imshow(S2,[])


% --- Executes on button press in pushbutton75.
function pushbutton75_Callback(hObject, eventdata, handles)%READ AN 2D IMAGE
A=imread('E:\\myfirstimage.jpg');
b=im2double(A);
[m,n]=size(A);
b=imnoise(b,'salt & pepper',0.02);
imshow(b);
%PAD THE MATRIX WITH ZEROS ON ALL SIDES
modifyA=zeros(size(A)+2);
B=zeros(size(A));

%COPY THE ORIGINAL IMAGE MATRIX TO THE PADDED MATRIX
        for x=1:size(A,1)
            for y=1:size(A,2)
                modifyA(x+1,y+1)=A(x,y);
            end
        end
      %LET THE WINDOW BE AN ARRAY
      %STORE THE 3-by-3 NEIGHBOUR VALUES IN THE ARRAY
      %SORT AND FIND THE MIDDLE ELEMENT
       
for i= 1:size(modifyA,1)-2
    for j=1:size(modifyA,2)-2
        window=zeros(9,1);
        inc=1;
        for x=1:3
            for y=1:3
                window(inc)=modifyA(i+x-1,j+y-1);
                inc=inc+1;
            end
        end
       
        med=sort(window);
        %PLACE THE MEDIAN ELEMENT IN THE OUTPUT MATRIX
        B(i,j)=med(5);
       
    end
end
%CONVERT THE OUTPUT MATRIX TO 0-255 RANGE IMAGE TYPE
B=uint8(B);
