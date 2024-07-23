function eq = afc_calsettings(BP_calibrate,FS)
% 4 May 2010, Regis
% Compute the coefficients of diff FIR Filters for each speaker of the dome,
% depending on the BP_calibrate matrix.
% Write those coeffs in a binary file (FIRcoefs.f32)
% designed for the HrtfFir component of RPvdsEx

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% variables %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def.samplerate = round(FS);

def.calTableEqualize = 'fir';			% equalize based on calTable: '' = not, 'fir'

% parameters for runtime window FIR filter design from calTable
def.calFilterDesignLow	= 190;			% should be >= def.samplerate/def.calFilterDesignFirCoef
def.calFilterDesignUp = 10000;
def.calFilterDesignFirCoef = 512;		% 64, 128, 256, 512

def.calTable = round(BP_calibrate.filters(:,1) + BP_calibrate.filters(:,2) / 2);
def.calTable = [def.calTable BP_calibrate.inv]; % dB inv values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% design fir filter

eq = [];


fftpts = def.samplerate;
binfactor = fftpts / def.samplerate;

xi = [0:def.samplerate/2+1]'/binfactor;


x = def.calTable(:,1);
y = def.calTable(:,2:end);

for idx = 1:size(y,2) 
	
	yi = interp1(x,y(:,idx),xi,'cubic'); % interpolation

	% new table with boundaries	
	% find NaN range
	nanRange = 1- isnan(yi); % returns one where we have values
	nan_low = min(find(nanRange > 0));
	nan_upp = max(find(nanRange > 0));
	
	yi(1:nan_low) = yi(nan_low);
	yi(nan_upp:end) = yi(nan_upp);
	
	xi_low = max(find(xi < def.calFilterDesignLow));
	xi_upp = min(find(xi > def.calFilterDesignUp));
	
	yi(1:xi_low) = yi(xi_low+1);
	yi(xi_upp:end) = yi(xi_upp-1);
	
	yi = 10.^(yi/20);
	
	invyi = zeros(fftpts,1);
	invyi(1:length(yi)) = yi;
	
    invyi(fftpts/2+2:end) = flipud(invyi(2:fftpts/2));
	%%% make minimum phase (comment out if not minimum phase)
    invyi = invyi.*exp(-i*imag(hilbert(log(invyi))));
	invyi(fftpts/2+2:end) = 0;
    %%%
    
	invyi(1) = 0;	%no dc
	fir = fftshift(2*real(ifft(invyi))); 
	
	%%%% window cut
	% asymmetric window for minimum phase
	winds = hannfl(def.calFilterDesignFirCoef,def.calFilterDesignFirCoef/16,def.calFilterDesignFirCoef/4);
	fir2 = fir(fftpts/2+1-def.calFilterDesignFirCoef/16:fftpts/2+def.calFilterDesignFirCoef-def.calFilterDesignFirCoef/16);
	fir2 = fir2.*winds;
	%otherwise
	%winds = hannfl(def.calFilterDesignFirCoef,def.calFilterDesignFirCoef/4,def.calFilterDesignFirCoef/4);
	%fir2 = fir(fftpts/2+1-def.calFilterDesignFirCoef/2:fftpts/2+def.calFilterDesignFirCoef/2);
	%fir2 = fir2.*winds;
	
	eq = [eq fir2];

end

eq = eq';

% % test filters
% bandpass = fir1(512,2000/(FS/2),'low');
% eq(15,:) = bandpass(1,1:512);

ntaps = def.calFilterDesignFirCoef;
nfilters = size(BP_calibrate.inv,2);
% creating binary file to be used with the Hrtf filter in RPvdsEx
% header
header = zeros(12,1);
header(1) = nfilters + 1; %Number of filter positions in RAM buffer. (number of Azimuths * Number of elevations)+1. 
header(2) = (ntaps * 2) + 2; % Number of taps (coefficients) including Interaural delay (ITD) delay (x2) per filter. e.g. 31 tap filter =31 x 2 + 2 (delay values)=64
header(3) = ntaps + 1; % Number of taps including the delay. e.g. 31 tap filter= 31 taps + delay value
header(4) = 1; % Minimum Azimuth value in degrees (e.g. -165)
header(5) = nfilters; % Maximum Azimuth value in degrees (e.g. 180)
header(6) = 1; % Inverse of the Position separation of Az in degrees, defined as 1.0/(AZ separation) e.g. 15 degrees between channel would =0.066666.
header(7) = nfilters;% Number of Az positions at each elevation.
header(8) = 0; % Minimum Elevation value in degrees.
header(9) = 90; % Maximum Elevation value in degrees (Must include a value for 90).
header(10) = 1/90; % Inverse of the Position separation of Elevation in degrees, defined as 1.0/(EL separation). e.g. 30 degrees between EL would be 1/30=.0333.
header(11) = 2; % Number of elevation positions for each Azimuth+1. The additional value is for the filter at 90 degrees. In cases where there will be no filter at 90 degrees elevation it is still necessary to include a dummy filter.
header(12) = 1000000 / FS; % Filter sampling period in microseconds. Calculated as the inverse of the sampling rate * 1,000,000.


f32file = 'C:/Users/Dome/DomeToolbox/textfiles/FIRcoefs.f32';

r = '';
while ~(strcmp(r,'y') || strcmp(r,'n'))
    r = input(sprintf('Save %s ? (y/n)\n',f32file),'s');
end
if strcmp(r,'y')
    fid = fopen(f32file,'w'); % open binary file

    % writing the header (see RPvdsEx help)
    fwrite(fid,header(1:3),'int32');
    fwrite(fid,header(4:6),'single');
    fwrite(fid,header(7),'int32');
    fwrite(fid,header(8:10),'single');
    fwrite(fid,header(11),'int32');
    fwrite(fid,header(12),'single');

    % writing the data

    % bandpass = fir1(256,[5000/(48828/2) 5500/(48828/2)]);
    % bandpass = bandpass(1:256);

    for j=1:513 % dummy set for El 90°
        fwrite(fid,0,'single'); % writing zeros in the left group
        fwrite(fid,0,'single'); % writing zeros in the right group
    end
    speaker=80; %%%% WARNING : Az will be shifted by 1 in RPvds (filter #80 will be applied with an Az of 79)
    while speaker>0 % FIR data
        fwrite(fid,eq(speaker,:),'single'); % writing left group
        fwrite(fid,0,'single'); % group delay
        for j=1:513
            fwrite(fid,0,'single'); % writing zeros in the right group
        end
        speaker=speaker-1;
    end

    fclose(fid); % close binary file
end
