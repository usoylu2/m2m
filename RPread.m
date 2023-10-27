%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% loads the ultrasound RF data saved from the Sonix software
%%
%% Inputs:  
%%     filename - The path of the data to open
%%
%% Return:
%%     Im -         The image data returned into a 3D array (h, w, numframes)
%%     header -     The file header information   
%%
%% Corina Leung, corina.leung@ultrasonix.com
%% Modified by Ali Baghani, Nov 2012 to support recent changes to the user
%% interface ali.baghani@ultrasonix.com
%% Ultrasonix Medical Corporation Nov 2007
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Im,header] = RPread(varargin)

% to determine whether a later version of Sonix is being used
if nargin > 1
    filename = varargin{1};    
    version  = varargin{2};         % Valid versions are: 5 for Sonix versions prior to 6.0
                                    %                     6 for newer Sonix versions 6.0.2, 6.0.3, 6.0.4, 6.0.6, etc 
else
    filename = varargin{1};
    version  = '6.0.3';
end

fid= fopen(filename, 'r');
fileExt = filename(end-3:end);

if( fid == -1)
    error('Cannot open file');
end

% read the header info
hinfo = fread(fid, 19, 'int32');

% load the header information into a structure and save under a separate file
header = struct('filetype', 0, 'nframes', 0, 'w', 0, 'h', 0, 'ss', 0, 'ul', [0,0], 'ur', [0,0], 'br', [0,0], 'bl', [0,0], 'probe',0, 'txf', 0, 'sf', 0, 'dr', 0, 'ld', 0, 'extra', 0);
header.filetype = hinfo(1);
header.nframes = hinfo(2);
header.w = hinfo(3);
header.h = hinfo(4);
header.ss = hinfo(5);
header.ul = [hinfo(6), hinfo(7)];
header.ur = [hinfo(8), hinfo(9)];
header.br = [hinfo(10), hinfo(11)];
header.bl = [hinfo(12), hinfo(13)];
header.probe = hinfo(14);
header.txf = hinfo(15);
header.sf = hinfo(16);
header.dr = hinfo(17);
header.ld = hinfo(18);
header.extra = hinfo(19);

% --------------  memory initialization for speeding up -------------------
switch header.filetype    
    case {2, 4, 8, 16, 64, 256, 1024, 4096, 8192, 16384, 32768}
        Im = zeros(header.h, header.w, header.nframes);
    case {32}
        Im = zeros(header.h, header.nframes);
    case {128}
        Im = zeros(header.h, 1, header.nframes);
    case {512}
        Im = zeros(header.h, header.w * header.extra, header.nframes);
    case {2048}
        Im = zeros(header.h, header.w * 2, header.nframes);
    case {524288}
        Im = zeros(header.w, header.nframes);
    otherwise
        Im = [];
end

h = waitbar(0, 'Generando las lineas RF');
% load the data and save into individual .mat files
for frame_count = 1:header.nframes
 
    if(header.filetype == 2) %.bpr
        if (regexp(version, '5.*') == 1) tag = fread(fid,1,'int32'); end % Each frame has 4 byte header for frame number in older versions
        [v,count] = fread(fid,header.w*header.h,'uchar=>uchar'); 
        Im(:,:,frame_count) = reshape(v,header.h,header.w);
   
    elseif(header.filetype == 4) %postscan B .b8
         if (regexp(version, '5.*') == 1) tag = fread(fid,1,'int32'); end
         [v,count] = fread(fid,header.w*header.h,'uint8'); 
         temp = int16(reshape(v,header.w,header.h));
         Im(:,:,frame_count) = imrotate(temp, -90); 
    
    elseif(header.filetype == 8) %postscan B .b32
         if (regexp(version, '5.*') == 1) tag = fread(fid,1,'int32'); end
         [v,count] = fread(fid,header.w*header.h,'uint32'); 
         temp = reshape(v,header.w,header.h);
         Im(:,:,frame_count) = imrotate(temp, -90); 
   
    elseif(header.filetype == 16) %rf
        if (regexp(version, '5.*') == 1) tag = fread(fid,1,'int32'); end
        [v,count] = fread(fid,header.w*header.h,'int16'); 
        Im(:,:,frame_count) = int16(reshape(v,header.h,header.w));
        
    elseif(header.filetype == 32) %.mpr
        if (regexp(version, '5.*') == 1) tag = fread(fid,1,'int32'); end
        [v,count] = fread(fid,header.h,'int16');
        Im(:,frame_count) = v;
    
    elseif(header.filetype == 64) %.m
        [v,count] = fread(fid,'uint8');
        temp = reshape(v,header.w,header.h);  
        Im(:,:,frame_count) = imrotate(temp,-90);
        
    elseif(header.filetype == 128) %.drf
        if (regexp(version, '5.*') == 1) tag = fread(fid,1,'int32'); end
        [v,count] = fread(fid,header.h,'int16'); 
        Im(:,:,frame_count) = int16(reshape(v,header.h,1));
        
    elseif(header.filetype == 512) %crf
        if (regexp(version, '5.*') == 1) tag = fread(fid,1,'int32'); end
        [v,count] = fread(fid,header.extra*header.w*header.h,'int16'); 
        Im(:,:,frame_count) = reshape(v,header.h,header.w*header.extra);
        % to obtain data per packet size use:
        % Im(:,:,:,frame_count) = reshape(v,header.h,header.w,header.extra);
   
    elseif(header.filetype == 256) %.pw
        [v,count] = fread(fid,'uint8');
        temp = reshape(v,header.w,header.h);  
        Im(:,:,frame_count) = imrotate(temp,-90);
        
%     elseif(header.filetype == 1024) %.col        %The old file format for SONIX version 3.0X
%          [v,count] = fread(fid,header.w*header.h,'int'); 
%          temp = reshape(v,header.w,header.h);
%          temp2 = imrotate(temp, -90); 
%          Im(:,:,frame_count) = mirror(temp2,header.w);
%     
%     elseif((header.filetype == 2048) & (fileExt == '.sig')) %color .sig  %The old file format for SONIX version 3.0X
%         %Each frame has 4 byte header for frame number
%         tag = fread(fid,1,'int32'); 
%         [v,count] = fread(fid,header.w*header.h,'uchar=>uchar'); 
%         temp = reshape(v,header.w,header.h);   
%         temp2 = imrotate(temp, -90);
%         Im(:,:,frame_count) = mirror(temp2,header.w);
        
     elseif(header.filetype == 1024) %.col
         [v,count] = fread(fid,header.w*header.h,'int'); 
         temp = reshape(v,header.w,header.h);
         temp2 = imrotate(temp, -90); 
         Im(:,:,frame_count) = mirror(temp2,header.w);
        
    elseif((header.filetype == 2048)) %color .cvv (the new format as of SONIX version 3.1X)
        % velocity data
        [v,count] = fread(fid,header.w*header.h,'uint8'); 
        temp = reshape(v,header.w,header.h); 
        temp2 = imrotate(temp, -90);
        tempIm1 = mirror(temp2,header.w);
        
        % sigma
        [v,count] =fread(fid, header.w*header.h,'uint8');
        temp = reshape(v,header.w, header.h);
        temp2 = imrotate(temp, -90);
        tempIm2 = mirror(temp2,header.w);
        
        Im(:,:,frame_count) = [tempIm1 tempIm2];
    
    elseif(header.filetype == 4096) %color vel
        if (regexp(version, '5.*') == 1) tag = fread(fid,1,'int32'); end
        [v,count] = fread(fid,header.w*header.h,'uchar=>uchar'); 
        temp = reshape(v,header.w,header.h); 
        temp2 = imrotate(temp, -90);
        Im(:,:,frame_count) = mirror(temp2,header.w);
        
    elseif(header.filetype == 8192) %.el
        [v,count] = fread(fid,header.w*header.h,'int32'); 
        temp = reshape(v,header.w,header.h);
        temp2 = imrotate(temp, -90);
        Im(:,:,frame_count) = mirror(temp2,header.w);
    
    elseif(header.filetype == 16384) %.elo
        [v,count] = fread(fid,header.w*header.h,'uchar=>uchar'); 
        temp = int16(reshape(v,header.w,header.h));
        temp2 = imrotate(temp, -90); 
        Im(:,:,frame_count) = mirror(temp2,header.w);
  
    elseif(header.filetype == 32768) %.epr
        [v,count] = fread(fid,header.w*header.h,'uchar=>uchar'); 
        Im(:,:,frame_count) = int16(reshape(v,header.h,header.w));
       
    elseif(header.filetype == 65536) %.ecg
        [v,count] = fread(fid,header.w*header.h,'uchar=>uchar'); 
        Im = v;
        
    elseif or(header.filetype == 131072, header.filetype == 262144) %.gps
        gps_posx(:, frame_count) =  fread(fid, 1, 'double');   %8 bytes for double
        gps_posy(:, frame_count) = fread(fid, 1, 'double');
        gps_posz(:, frame_count) = fread(fid, 1, 'double');
        if strcmp(version, '6.0.3')
            gps_a(:, frame_count) =  fread(fid, 1, 'double');  
            gps_e(:, frame_count) = fread(fid, 1, 'double');
            gps_r(:, frame_count) = fread(fid, 1, 'double');
        end         
        gps_s(:,:, frame_count) = fread(fid, 9, 'double');    %position matrix
        if strcmp(version, '6.0.3')
            gps_q(:, frame_count) =  fread(fid, 4, 'double');   
        end
        gps_time(:, frame_count) = fread(fid, 1, 'double');
        gps_quality(:, frame_count) = fread(fid, 1, 'ushort'); % 2 bytes for unsigned short
        Zeros(:, frame_count) =  fread(fid, 3, 'uint16');     % 6 bytes of padded zeroes
    
    elseif (header.filetype == 524288) % time stamps
        LineStartInClkCycles = fread(fid, header.w, 'uint32');
        Im(:, frame_count) = LineStartInClkCycles;
    else
        disp('Data not supported');
    end
    
    if (ishandle(h))
        waitbar(frame_count/header.nframes, h);
    else 
        h = waitbar(frame_count/header.nframes, 'Generando las senales RF.');
    end   
end

% Scaling factor (Optional)
Im = Im*1.5;

if or(header.filetype == 131072, header.filetype == 262144) %.gps
    Im.gps_posx    = gps_posx;
    Im.gps_posy    = gps_posy;
    Im.gps_posz    = gps_posz;
    Im.gps_s       = gps_s;
    Im.gps_time    = gps_time;
    Im.gps_quality = gps_quality;
    Im.gps_Zeros   = Zeros;
    if strcmp(version, '6.0.3')
     Im.gps_a    = gps_a;
     Im.gps_e    = gps_e;
     Im.gps_r    = gps_r;
     Im.gps_q    = gps_q;
    end
end

pause(0.1);
if (ishandle(h))
    delete(h);
end
fclose(fid);

% % for RF data, plot both the RF center line and image 
% if(header.filetype == 16 || header.filetype == 512)
%     RPviewrf(Im, header, 24);
% 
% elseif(header.filetype == 128)
%     PDRFLine = Im(:,:,1);
%     figure, plot(PDRFLine, 'b');
%     title('PW RF');
%     
% elseif(header.filetype == 256)
%     figure, imagesc(Im(:,:,1));
%     colormap(gray);
%     title('PW spectrum');
% end
