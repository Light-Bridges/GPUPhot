from unittest import TestCase

import astropy.io.fits
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

from gpuphot.astrometry.utils import (
    get_target_ephemeris,
    px_to_wcs,
    wcs_to_px,
    get_target_ra_dec,
    get_target_pix,
    get_ccw,
    get_if_header_already_post_processed,
    cat_input_from_header,
)


class TestAstrometryUtils(TestCase):
    # TTT2_QHY411-2_2023-02-19-20-56-11-676285_1993VB.fits file
    _header2_str = """
    SIMPLE  =                    T / conforms to FITS standard                      
BITPIX  =                  -32 / array data type                                
NAXIS   =                    2 / number of array dimensions                     
NAXIS1  =                14200                                                  
NAXIS2  =                10650                                                  
COMMENT ***************************                                             
COMMENT          TELESCOPE                                                      
COMMENT ***************************                                             
TELESCOP= 'TTT2    '           / Telescope name                                 
SITELAT =          28.29871203 / Telescope latitude, in degrees                 
SITELONG=         -16.50948217 / Telescope longitude, in degrees                
SITEALT =              2359.11 / Telescope altitude, in meters                  
DIAMETER=                800.0 / Telescope diameter, in mm                      
FOCAL   =                 6.85 / Telescope focal ratio                          
FOCALEN =               5480.0 / Telescope focal length, in mm                  
TRACK   =                    1 / Sidereal tracking, 0=no, 1=yes                 
COMMENT ***************************                                             
COMMENT          INSTRUMENT                                                     
COMMENT ***************************                                             
SHMODE  = 'Rolling shutter'    / Electronic shutter mode                        
DRMODE  = 'HDR     '           / Dynamic range mode                             
COMODE  = 'Mono 16 '           / Color mode                                     
RDMODE  = 'Full Frame Read Mode #4' / Readout mode                              
RDNOISE =                  2.8 / Readout noise, in e-                           
INTEMP  =                -15.0 / Temperature of the camera                      
SATLEVEL=                65536 / Saturation level, in ADU                       
GAIN    =                1.024 / Gain, in e/cnt(ADU)                            
DC      =                0.007 / Dark current                                   
SCALEORI=                0.142 / [arcsec/px] Nominal plate scale                
PARITY  = 'pos     '           / Image parity                                   
XBINNING=                    1 / Binning in x axis                              
YBINNING=                    1 / Binning in y axis                              
OVERSCSX=                   51 / Overscan region origin in X axis               
OVERSCNX=                14253 / Overscan region lines in X axis                
OVERSCSY=                10659 / Overscan region origin in Y axis               
OVERSCNY=                   88 / Overscan region lines in Y axis                
EFFECTSX=                   51 / Effective region origin in X axis              
EFFECTNX=                14253 / Effective region lines in X axis               
EFFECTSY=                    0 / Effective region origin in Y axis              
EFFECTNY=                10650 / Effective region lines in Y axis               
STATSX  =                 6640 / Stats region origin in X axis                  
STATNX  =                 1024 / Stats region lines in X axis                   
STATSY  =                 4862 / Stats region origin in Y axis                  
STATNY  =                 1024 / Stats region lines in Y axis                   
INSTRUME= 'QHY411-TTT2'        / Instrument                                     
INMODEL = 'QHY411MERIS'        / Instrument model                               
INSERIAL= '2566a14f9ee62fe1a'  / Instrument serial number                       
INSDK   = '22-7-25.16'         / Instrument SDK version                         
OFFSET  =                 10.0 / Pedestal level, in ADU                         
PXSIZE  =                 3.76 / Pixel size, in micron                          
ORISIZEX=                14304 / Number of pixels in axis X                     
ORISIZEY=                10748 / Number of pixels in axis Y                     
ALISIZEX=                14304 / Image xsize for aligning, centered             
ALISIZEY=                10748 / Image ysize for aligning, centered             
FINSIZEX=                14304 / Image xsize for final saved image, centered    
FINSIZEY=                10748 / Image ysize for final saved image, centered    
COMMENT ***************************                                             
COMMENT       EXPOSITION DATA                                                   
COMMENT ***************************                                             
CAMERA  = 'QHY411-2'           / Camera tag name                                
INFIRMW = '2017_5_6'           / Instrument firmware version                    
DATE-OBS= '2023-02-19T20:56:11.676285' / Date of capture observation            
GPSDAT  = '2023-02-19T20:56:11.676285' / Date of capture observation from the GP
GPSLAT  =            28.298815 / Latitude measured by the GPS                   
GPSLON  =   -16.50944833333334 / Longitude measured by the GPS                  
SEQNUM  =                   41 / Number of the frame in the sequence            
JD-OBS  =    2459995.372357364 / Julian date                                    
MJD-OBS =    59994.87235736441 / Modified julian date                           
PCDATE  = '2023-02-19T20:56:11.683098' / Date of capture observation from the PC
EXPTIME =             5.326963 / Exposition time                                
GAMODE  =                  0.0 / Gain mode                                      
WINDOWSX=                    0 / Window region origin in X axis                 
WINDOWNX=                14304 / Window region lines in X axis                  
WINDOWSY=                    0 / Window region origin in Y axis                 
WINDOWNY=                10748 / Window region lines in Y axis                  
OBJECT  = '1993VB  '           / Object                                         
FILTER  = 'Lum     '           / Filter used                                    
POINTRA =    4.423111699254016 / Right ascension (J2000) sent to telescope      
POINTDEC=    26.56427818078933 / Declination (J2000) sent to telescope          
UT1     = '2023-02-19T20:56:11.676285' / Datetime of sub-frame 1                
GPSDAT1 = '2023-02-19T20:56:11.676285' / GPS Datetime of sub-frame 1            
PCDAT1  = '2023-02-19T20:56:11.683098' / PC Datetime of sub-frame 1             
EXPT1   =             5.326963 / Exposure time of sub-frame 1                   
TEMP1   =                -15.0 / Camera temperature (C) of sub-frame 1          
SEQNUM1 =                   41 / Frame number of sub-frame 1                    
FOCUS   =                13898 / Focuser position in micrometers                
UTOBS   = '2023-02-19T20:56:11.676285' / Mean datetime of observation           
INTEGT  =                  5.3 / Sum of all exposition times                    
TOTIMA  =                    1 / Number of captures for this image              
OBLINEID=               111545 / Observation Line ID from the database          
COMMENT ***************************                                             
COMMENT         WEATHER DATA                                                    
COMMENT ***************************                                             
HUMIDITY=                55.44 / Relative humidity (%)                          
COMMENT ***************************                                             
COMMENT         PRE-REDUCTION                                                   
COMMENT ***************************                                             
BIASCORR= 'bias    '           / Bias correction method                         
BIASMEAN=              171.718 / MasterBias mean, in ADU                        
BIASSTD =                 3.32 / MasterBias standard deviation, in ADU          
BIASOVER=              171.734 / Median level in MasterBias overscan, in ADU    
BIASN   =                   21 / Number of stacked frames in MasterBias         
BIASTEMP=                -12.3 / Instrument temperature of MasterBias, in Celsiu
BIASDATE= '2023-02-19T18:36:44.834639' / Date of capture of MasterBias          
FLATMEAN=            14175.844 / Mean of the MasterFlat frame, in ADU           
FLATSTD =              129.843 / Standard deviation in flat stacking, in ADU    
FLATN   =                   11 / Number of staked frames in MasterFlat          
FLATTEMP=                -13.0 / Instrument temperature of MasterFlat, in Celsiu
FLATDATE= '2023-02-19T07:18:19.075587' / Date of capture of MasterFlat          
DATE    = '2023-04-15T01:38:21' / Date of file creation                         
COMMENT ***************************                                             
COMMENT        ASTROMETRY                                                       
COMMENT ***************************                                             
RA      =    66.42826273907058 / [deg] Central right ascension                  
DEC     =    26.62161140115776 / [deg] Central declination                      
RAHMS   = '04:25:42.783057'    / [hh:mm:ss] Central Right ascension             
DECDMS  = '26:37:17.801044'    / [dd:mm:ss] Central declination                 
AZ      =           269.626985 / [deg] Azimut                                   
ALT     =    71.93717599999999 / [deg] Altitude                                 
ZD      =            18.062824 / [deg] Zenital distance, in degrees             
AIRMASS =             1.051838 / Airmass                                        
LONGAL  =           171.230494 / [deg] Galactic longitude                       
LATGAL  =           -15.555384 / [deg] Galactic latitude                        
LONECL  =            68.973102 / [deg] Baricentric mean ecliptic longitude      
LATECL  =             4.886683 / [deg] Baricentric mean ecliptic latitude       
WCSAXES =                    2 / Number of coordinate axes                      
EQUINOX =               2000.0 / Equatorial coordinates definition (yr)         
LONPOLE =                180.0 / Native longitude of celestial pole (deg)       
LATPOLE =                  0.0 / Native latitude of celestial pole (deg)        
CRVAL1  =     66.5731118308627 / RA of reference point                          
CRVAL2  =    26.70513733216426 / DEC of reference point                         
CRPIX1  =    9764.406291715912 / X reference pixel                              
CRPIX2  =    8199.798698904298 / Y reference pixel                              
CUNIT1  = 'deg     '           / X pixel scale units                            
CUNIT2  = 'deg     '           / Y pixel scale units                            
CD1_1   = 6.84340927896256E-06 / Transformation matrix                          
CD1_2   = 3.87017232734834E-05 / Transformation matrix                          
CD2_1   = 3.86905151533621E-05 / Transformation matrix                          
CD2_2   = -6.8299073795454E-06 / Transformation matrix                          
CTYPE1  = 'RA---TAN-SIP'       / TAN (gnomonic) projection + SIP distortions    
CTYPE2  = 'DEC--TAN-SIP'       / TAN (gnomonic) projection + SIP distortions    
A_ORDER =                    3 / Polynomial order, axis 1                       
A_0_0   =                  0.0 / Polynomial coefficient, axis 1                 
A_0_1   =                  0.0 / Polynomial coefficient, axis 1                 
A_0_2   = 4.73780990402977E-08 / Polynomial coefficient, axis 1                 
A_0_3   = 4.22005169691596E-12 / Polynomial coefficient, axis 1                 
A_1_0   =                  0.0 / Polynomial coefficient, axis 1                 
A_1_1   = -5.7176900597633E-08 / Polynomial coefficient, axis 1                 
A_1_2   = -3.5659289348658E-12 / Polynomial coefficient, axis 1                 
A_2_0   = -1.3129239874033E-07 / Polynomial coefficient, axis 1                 
A_2_1   = -1.0242455126531E-11 / Polynomial coefficient, axis 1                 
A_3_0   = -1.7383258240407E-11 / Polynomial coefficient, axis 1                 
B_ORDER =                    3 / Polynomial order, axis 2                       
B_0_0   =                  0.0 / Polynomial coefficient, axis 2                 
B_0_1   =                  0.0 / Polynomial coefficient, axis 2                 
B_0_2   = 6.69007879531292E-09 / Polynomial coefficient, axis 2                 
B_0_3   = -2.5737331530670E-14 / Polynomial coefficient, axis 2                 
B_1_0   =                  0.0 / Polynomial coefficient, axis 2                 
B_1_1   = -4.0365293412060E-08 / Polynomial coefficient, axis 2                 
B_1_2   = -9.3644790143299E-12 / Polynomial coefficient, axis 2                 
B_2_0   = -7.0322927217340E-08 / Polynomial coefficient, axis 2                 
B_2_1   = -3.3331436414930E-12 / Polynomial coefficient, axis 2                 
B_3_0   = -1.0658621983011E-11 / Polynomial coefficient, axis 2                 
AP_ORDER=                    3 / Inv polynomial order, axis 1                   
AP_0_0  = -0.00161476092667527 / Inv polynomial coefficient, axis 1             
AP_0_1  = -1.7220915038522E-07 / Inv polynomial coefficient, axis 1             
AP_0_2  = -4.7363792125673E-08 / Inv polynomial coefficient, axis 1             
AP_0_3  = -4.2195087651648E-12 / Inv polynomial coefficient, axis 1             
AP_1_0  = 9.56583422983116E-08 / Inv polynomial coefficient, axis 1             
AP_1_1  = 5.74198698131822E-08 / Inv polynomial coefficient, axis 1             
AP_1_2  = 3.58426306939799E-12 / Inv polynomial coefficient, axis 1             
AP_2_0  = 1.31683014580356E-07 / Inv polynomial coefficient, axis 1             
AP_2_1  = 1.02894027357783E-11 / Inv polynomial coefficient, axis 1             
AP_3_0  = 1.74308119033378E-11 / Inv polynomial coefficient, axis 1             
BP_ORDER=                    3 / Inv polynomial order, axis 2                   
BP_0_0  = -0.00128863202567437 / Inv polynomial coefficient, axis 2             
BP_0_1  = -1.1728483577720E-07 / Inv polynomial coefficient, axis 2             
BP_0_2  = -6.6351435582233E-09 / Inv polynomial coefficient, axis 2             
BP_0_3  = 3.21998549459442E-14 / Inv polynomial coefficient, axis 2             
BP_1_0  = 4.04272314790572E-08 / Inv polynomial coefficient, axis 2             
BP_1_1  = 4.05799078044546E-08 / Inv polynomial coefficient, axis 2             
BP_1_2  = 9.38479325507474E-12 / Inv polynomial coefficient, axis 2             
BP_2_0  = 7.05901127290546E-08 / Inv polynomial coefficient, axis 2             
BP_2_1  = 3.37127115036146E-12 / Inv polynomial coefficient, axis 2             
BP_3_0  = 1.06902353633817E-11 / Inv polynomial coefficient, axis 2             
RA      =    66.42826273907058                                                  
DEC     =    26.62161140115776                                                  
COMMENT ***************************                                             
COMMENT        PHOTOMETRY                                                       
COMMENT ***************************                                             
FOVX    =   0.4200833333333333 / Image horizontal axis Fiel of View(deg)        
FOVY    =   0.5601111111111111 / Image vertical axis Fiel of View(deg)          
ZP      =    23.15573159902409 / Zero point                                     
EZP     =  0.06360210866361185 / Zero point's error                             
FWHM    =    5.001328468322754 / Full width                                     
EFWHM   =     0.56470787525177 / Full width's error                             
M_LIM   =      18.720908109705 / Limit magnitude (3 sigmas)                     
M_SKY   =     24.4681875057086 / Sky's magnitud                                 
SKY     =    3.135298728942871 / Sky flux                                       
ESKY    =    4.061130523681641 / Sky's flux error                               
SCALE   =                0.142 / Image scale in arcsec                          
CCW     =    100.0193684428295 / Field rotation                                 
AP_SNR  =                   49 / Opening radius maximising snr                  
AP_FLUX =                   49 / Opening radius maximising flux                 
END
"""
    _header2 = astropy.io.fits.Header.fromstring(_header2_str, sep="\n")
    _header1_str = """
    SIMPLE  =                    T / conforms to FITS standard                      
BITPIX  =                  -32 / array data type                                
NAXIS   =                    2 / number of array dimensions                     
NAXIS1  =                 2048                                                  
NAXIS2  =                 2048                                                  
COMMENT ***************************                                             
COMMENT          TELESCOPE                                                      
COMMENT ***************************                                             
TELESCOP= 'TTT1    '           / Telescope name                                 
SITELAT =          28.29871121 / Telescope latitude, in degrees                 
SITELONG=         -16.50956893 / Telescope longitude, in degrees                
SITEALT =              2359.12 / Telescope altitude, in meters                  
DIAMETER=                800.0 / Telescope diameter, in mm                      
FOCAL   =                 6.85 / Telescope focal ratio                          
FOCALEN =               5480.0 / Telescope focal length, in mm                  
TRACK   =                    1 / Sidereal tracking, 0=no, 1=yes                 
COMMENT ***************************                                             
COMMENT          INSTRUMENT                                                     
COMMENT ***************************                                             
SHMODE  = 'Mechanical shutter' / Electronic shutter mode                        
DRMODE  =  / Dynamic range mode                                                 
COMODE  =  / Color mode                                                         
RDMODE  = 'Multi track'        / Readout mode                                   
RDNOISE =                  7.0 / Readout noise, in e-                           
INTEMP  =              -55.883 / Temperature of the camera                      
SATLEVEL=                65536 / Saturation level, in ADU                       
GAIN    =                  1.0 / Gain, in e/cnt(ADU)                            
DC      =                 0.02 / Dark current                                   
SCALEORI=                0.508 / [arcsec/px] Nominal plate scale                
PARITY  = 'pos     '           / Image parity                                   
XBINNING=                    1 / Binning in x axis                              
YBINNING=                    1 / Binning in y axis                              
EFFECTSX=                    0 / Effective region origin in X axis              
EFFECTNX=                 2048 / Effective region lines in X axis               
EFFECTSY=                    0 / Effective region origin in Y axis              
EFFECTNY=                 2048 / Effective region lines in Y axis               
STATSX  =                    0 / Stats region origin in X axis                  
STATNX  =                 2048 / Stats region lines in X axis                   
STATSY  =                    0 / Stats region origin in Y axis                  
STATNY  =                 2048 / Stats region lines in Y axis                   
INSTRUME= 'iKon936-TTT1'       / Instrument                                     
INMODEL = 'iKon936 '           / Instrument model                               
INSERIAL=                27305 / Instrument serial number                       
INSDK   = '2.104   '           / Instrument SDK version                         
OFFSET  =  / Pedestal level, in ADU                                             
PXSIZE  =                 13.5 / Pixel size, in micron                          
ORISIZEX=                 2048 / Number of pixels in axis X                     
ORISIZEY=                 2048 / Number of pixels in axis Y                     
ALISIZEX=                 2048 / Image xsize for aligning, centered             
ALISIZEY=                 2048 / Image ysize for aligning, centered             
FINSIZEX=                 2048 / Image xsize for final saved image, centered    
FINSIZEY=                 2048 / Image ysize for final saved image, centered    
COMMENT ***************************                                             
COMMENT       EXPOSITION DATA                                                   
COMMENT ***************************                                             
CAMERA  = 'iKon936-1'          / Camera tag name                                
INFIRMW = '20.12   '           / Instrument firmware version                    
HSSPEED =                  1.0 / Horizontal shift speed in MHz (readout rate)   
VSSPEED =            38.549999 / Vertical shift speed in microseconds per pixel 
PREAMPGA=                  4.0 / Pre Amp gain factor                            
DATE-OBS= '2023-02-08T20:52:40.538105' / Date of capture observation            
JD-OBS  =    2459984.369913635 / Julian date                                    
MJD-OBS =    59983.86991363548 / Modified julian date                           
PCDATE  = '2023-02-08T20:52:40.538105' / Date of capture observation from the PC
COOLERST= 'DRV_TEMP_NOT_REACHED' / Status of the cooler                         
EXPTIME =               13.736 / Exposition time                                
WINDOWSX=                    0 / Window region origin in X axis                 
WINDOWNX=                 2048 / Window region lines in X axis                  
WINDOWSY=                    0 / Window region origin in Y axis                 
WINDOWNY=                 2048 / Window region lines in Y axis                  
OBJECT  = 'Kellyoconnor'       / Object                                         
FILTER  = 'SDSSg   '           / Filter used                                    
POINTRA =    3.652383384165774 / Right ascension (J2000) sent to telescope      
POINTDEC=    11.85168222874598 / Declination (J2000) sent to telescope          
UT1     = '2023-02-08T20:52:40.538105' / Datetime of sub-frame 1                
PCDAT1  = '2023-02-08T20:52:40.538105' / PC Datetime of sub-frame 1             
EXPT1   =               13.736 / Exposure time of sub-frame 1                   
TEMP1   =              -55.883 / Camera temperature (C) of sub-frame 1          
FOCUS   =                12190 / Focuser position in micrometers                
UTOBS   = '2023-02-08T20:52:40.538105' / Mean datetime of observation           
INTEGT  =                 13.7 / Sum of all exposition times                    
TOTIMA  =                    1 / Number of captures for this image              
OBLINEID=                81156 / Observation Line ID from the database          
COMMENT ***************************                                             
COMMENT         WEATHER DATA                                                    
COMMENT ***************************                                             
HUMIDITY=                 4.15 / Relative humidity (%)                          
MIRRHUM =               15.601 / Telescope mirror humidity, in Celsius          
PRESSURE=                766.0 / Local pressure, in mbar                        
AMBTEMP =                 3.41 / Ambient temperature, in Celsius                
MIRRTEMP=                 0.96 / Telescope mirror temperature, in Celsius       
CLOUD   =                  0.0 / Cloud presence, 0 = Clear, 1 = Cloudy          
ILLUMINA=                  1.6 / Solar illuminance, in lux                      
WINDDIR =              216.415 / Wind direction (in deg)                        
WINDVEL =                2.447 / Wind speed, in m/s                             
DUSTPLA =                0.003 / Dust, in Polystyrol-Latex-Aerosols (PLA)/m3    
DUSTPM1 =                 0.04 / Particle matter <1 micron, in 1e-6g/m3         
DUSTPM10=                 0.04 / Particle matter <10 microns, in 1e-6g/m3       
DUSTPM25=                 0.04 / Particle matter <2.5 microns, in 1e-6g/m3      
PWV     =                 1.51 / Precipitable water vapour, in mm               
TESSMAG =                20.84 / TESS-W cenital sky brightness, in mag/arcsec2  
SKYIRTEM=               -49.75 / IR cenital sky temperature, in Celsius         
COMMENT ***************************                                             
COMMENT         PRE-REDUCTION                                                   
COMMENT ***************************                                             
BIASCORR= 'bias+dark'          / Bias correction method                         
BIASMEAN=              268.871 / MasterBias mean, in ADU                        
BIASSTD =                6.412 / MasterBias standard deviation, in ADU          
BIASN   =                   21 / Number of stacked frames in MasterBias         
BIASTEMP=              -53.287 / Instrument temperature of MasterBias, in Celsiu
BIASDATE= '2023-03-01T07:39:43.292806' / Date of capture of MasterBias          
DARKMEAN=                0.662 / Mean of the MasterDark frame, in ADU/s         
DARKSTD =                0.287 / MasterDark standard deviation, in ADU/sec      
DARKN   =                   21 / Number of staked frames in MasterDark          
DARKTEMP=              -53.287 / Instrument temperature of MasterDark, in Celsiu
DARKDATE= '2023-03-01T07:44:13.077313' / Date of capture of MasterDark          
FLATMEAN=            15940.665 / Mean of the MasterFlat frame, in ADU           
FLATSTD =              552.083 / Standard deviation in flat stacking, in ADU    
FLATN   =                   11 / Number of staked frames in MasterFlat          
FLATTEMP=              -56.532 / Instrument temperature of MasterFlat, in Celsiu
FLATDATE= '2023-02-02T19:13:25.192610' / Date of capture of MasterFlat          
DATE    = '2023-04-15T03:26:34' / Date of file creation                         
COMMENT ***************************                                             
COMMENT        ASTROMETRY                                                       
COMMENT ***************************                                             
RA      =    54.78885815432049 / [deg] Central right ascension                  
DEC     =    11.85325565554055 / [deg] Central declination                      
RAHMS   = '03:39:9.325957'     / [hh:mm:ss] Central Right ascension             
DECDMS  = '11:51:11.720360'    / [dd:mm:ss] Central declination                 
AZ      =           233.224488 / [deg] Azimut                                   
ALT     =            64.983079 / [deg] Altitude                                 
ZD      =            25.016921 / [deg] Zenital distance, in degrees             
AIRMASS =              1.10353 / Airmass                                        
LONGAL  =           174.721503 / [deg] Galactic longitude                       
LATGAL  =           -33.665493 / [deg] Galactic latitude                        
LONECL  =            55.312732 / [deg] Baricentric mean ecliptic longitude      
LATECL  =            -7.447103 / [deg] Baricentric mean ecliptic latitude       
WCSAXES =                    2 / Number of coordinate axes                      
EQUINOX =               2000.0 / Equatorial coordinates definition (yr)         
LONPOLE =                180.0 / Native longitude of celestial pole (deg)       
LATPOLE =                  0.0 / Native latitude of celestial pole (deg)        
CRVAL1  =    54.89839141198505 / RA of reference point                          
CRVAL2  =    11.82245370857722 / DEC of reference point                         
CRPIX1  =    1777.618690422958 / X reference pixel                              
CRPIX2  =    781.3124627786053 / Y reference pixel                              
CUNIT1  = 'deg     '           / X pixel scale units                            
CUNIT2  = 'deg     '           / Y pixel scale units                            
CD1_1   = 0.000116964650936684 / Transformation matrix                          
CD1_2   = -7.8500358575673E-05 / Transformation matrix                          
CD2_1   = -7.8551079513038E-05 / Transformation matrix                          
CD2_2   = -0.00011691809547351 / Transformation matrix                          
CTYPE1  = 'RA---TAN-SIP'       / TAN (gnomonic) projection + SIP distortions    
CTYPE2  = 'DEC--TAN-SIP'       / TAN (gnomonic) projection + SIP distortions    
A_ORDER =                    3 / Polynomial order, axis 1                       
A_0_0   =                  0.0 / Polynomial coefficient, axis 1                 
A_0_1   =                  0.0 / Polynomial coefficient, axis 1                 
A_0_2   = -5.3744942932387E-07 / Polynomial coefficient, axis 1                 
A_0_3   = -2.5802215056889E-10 / Polynomial coefficient, axis 1                 
A_1_0   =                  0.0 / Polynomial coefficient, axis 1                 
A_1_1   = 1.73787726649304E-06 / Polynomial coefficient, axis 1                 
A_1_2   = -3.4733304094105E-10 / Polynomial coefficient, axis 1                 
A_2_0   = 6.86166720388377E-07 / Polynomial coefficient, axis 1                 
A_2_1   = 1.13276746176446E-09 / Polynomial coefficient, axis 1                 
A_3_0   = 1.31927020229763E-10 / Polynomial coefficient, axis 1                 
B_ORDER =                    3 / Polynomial order, axis 2                       
B_0_0   =                  0.0 / Polynomial coefficient, axis 2                 
B_0_1   =                  0.0 / Polynomial coefficient, axis 2                 
B_0_2   = -5.0382145553174E-08 / Polynomial coefficient, axis 2                 
B_0_3   = -2.0787529074745E-10 / Polynomial coefficient, axis 2                 
B_1_0   =                  0.0 / Polynomial coefficient, axis 2                 
B_1_1   = 6.35335248126149E-08 / Polynomial coefficient, axis 2                 
B_1_2   = -1.3019742707861E-10 / Polynomial coefficient, axis 2                 
B_2_0   = -2.7512502647287E-07 / Polynomial coefficient, axis 2                 
B_2_1   = -3.5845416487618E-11 / Polynomial coefficient, axis 2                 
B_3_0   = -1.5459837760668E-10 / Polynomial coefficient, axis 2                 
AP_ORDER=                    3 / Inv polynomial order, axis 1                   
AP_0_0  = 1.83767365063758E-05 / Inv polynomial coefficient, axis 1             
AP_0_1  = -7.7012534406790E-07 / Inv polynomial coefficient, axis 1             
AP_0_2  = 5.37179517105436E-07 / Inv polynomial coefficient, axis 1             
AP_0_3  = 2.58688170888574E-10 / Inv polynomial coefficient, axis 1             
AP_1_0  = 1.04309490716990E-06 / Inv polynomial coefficient, axis 1             
AP_1_1  = -1.7430364824405E-06 / Inv polynomial coefficient, axis 1             
AP_1_2  = 3.46894225396888E-10 / Inv polynomial coefficient, axis 1             
AP_2_0  = -6.8416717034359E-07 / Inv polynomial coefficient, axis 1             
AP_2_1  = -1.1367083921002E-09 / Inv polynomial coefficient, axis 1             
AP_3_0  = -1.3066890117205E-10 / Inv polynomial coefficient, axis 1             
BP_ORDER=                    3 / Inv polynomial order, axis 2                   
BP_0_0  = -1.8162994641013E-05 / Inv polynomial coefficient, axis 2             
BP_0_1  = -1.3659800485951E-07 / Inv polynomial coefficient, axis 2             
BP_0_2  = 5.04294536598353E-08 / Inv polynomial coefficient, axis 2             
BP_0_3  = 2.08190125014730E-10 / Inv polynomial coefficient, axis 2             
BP_1_0  = 2.32341117048019E-07 / Inv polynomial coefficient, axis 2             
BP_1_1  = -6.3443477767157E-08 / Inv polynomial coefficient, axis 2             
BP_1_2  = 1.30422066111061E-10 / Inv polynomial coefficient, axis 2             
BP_2_0  = 2.75974486529842E-07 / Inv polynomial coefficient, axis 2             
BP_2_1  = 3.60335982649415E-11 / Inv polynomial coefficient, axis 2             
BP_3_0  = 1.55121683602205E-10 / Inv polynomial coefficient, axis 2             
RA      =    54.78885815432049                                                  
DEC     =    11.85325565554055                                                  
COMMENT ***************************                                             
COMMENT        PHOTOMETRY                                                       
COMMENT ***************************                                             
FOVX    =   0.2889955555555556 / Image horizontal axis Fiel of View(deg)        
FOVY    =   0.2889955555555556 / Image vertical axis Fiel of View(deg)          
ZP      =                  0.0 / Zero point                                     
EZP     =                  0.0 / Zero point's error                             
FWHM    =    3.942845582962036 / Full width                                     
EFWHM   =   0.4234123826026917 / Full width's error                             
M_LIM   =                  0.0 / Limit magnitude (3 sigmas)                     
M_SKY   =   0.5923062153966254 / Sky's magnitud                                 
SKY     =    10.81001853942871 / Sky flux                                       
ESKY    =    9.739238739013672 / Sky's flux error                               
SCALE   =                0.508 / Image scale in arcsec                          
CCW     =   -146.1187789098986 / Field rotation                                 
AP_SNR  =                   49 / Opening radius maximising snr                  
AP_FLUX =                   49 / Opening radius maximising flux                 
END                                                                             
    """

    _header1 = astropy.io.fits.Header.fromstring(_header1_str, sep="\n")

    def _split_target_name(self, s):
        result = ""
        prev_type = None

        for char in s:
            current_type = "letter" if char.isalpha() else "number"

            # Check if there's a transition from letter to number or vice versa
            if prev_type is not None and prev_type != current_type:
                result += " "

            result += char
            prev_type = current_type

        return result

    def test_get_target_ephemeris(self):
        _site_lat = +28.300332
        _site_lon = -16.512206
        _site_elev = 2390
        _target_name = "Ceres"
        _date_obs = 2458133.33546
        _ephems = (9.434766666666667, 27.97686)

        ephems = get_target_ephemeris(
            _target_name, _date_obs, _site_lat, _site_lon, _site_elev
        )

        self.assertEquals(ephems, _ephems)

    def test_get_target_ephemeris_location(self):
        _site_lat = -34.6037
        _site_lon = -58.3816
        _site_elev = 50
        _target_name = "Ceres"
        _date_obs = 2458133.33546
        _ephems = (9.434723333333332, 27.97789)

        ephems = get_target_ephemeris(
            _target_name, _date_obs, _site_lat, _site_lon, _site_elev
        )

        self.assertEquals(ephems, _ephems)

    def test_get_target_ephemeris_target(self):
        _site_lat = +28.300332
        _site_lon = -16.512206
        _site_elev = 2390
        _target_name = "Mars Barycenter"
        _date_obs = 2458133.33546
        _ephems = (15.355653333333334, -17.6265)

        ephems = get_target_ephemeris(
            _target_name, _date_obs, _site_lat, _site_lon, _site_elev
        )

        self.assertEquals(ephems, _ephems)

    def test_get_target_ephemeris_date(self):
        _site_lat = +28.300332
        _site_lon = -16.512206
        _site_elev = 2390
        _target_name = "Ceres"
        _date_obs = 2459000.0  # Cambia la fecha de observación
        _ephems = (22.944357333333336, -17.19894)

        ephems = get_target_ephemeris(
            _target_name, _date_obs, _site_lat, _site_lon, _site_elev
        )

        self.assertEquals(ephems, _ephems)

    def test_px_to_wcs_h1(self):
        x = 100
        y = 150
        wcs_coords = px_to_wcs(x, y, self._header1)

        expected_coords = [np.array(54.74860556759758), np.array(12.027688383745517)]

        self.assertListEqual(
            [float(wcs_coords[0]), float(wcs_coords[1])], expected_coords
        )

    def test_px_to_wcs_h2(self):
        x = 100
        y = 150
        wcs_coords = px_to_wcs(x, y, self._header2)

        expected_coords = [np.array(66.15202567431312), np.array(26.385933695156663)]

        self.assertListEqual(
            [float(wcs_coords[0]), float(wcs_coords[1])], expected_coords
        )

    def test_wcs_to_px_h1(self):
        cat = {"RA": 54.74875915682997, "DEC": 12.027625065441386}
        wcs_coords = wcs_to_px(cat, self._header1)

        expected_pix = [np.array(101), np.array(149)]

        self.assertListEqual([int(wcs_coords[0]), int(wcs_coords[1])], expected_pix)

    def test_wcs_to_px_h2(self):
        cat = {"RA": 66.15202567431312, "DEC": 26.385933695156663}
        wcs_coords = wcs_to_px(cat, self._header2)

        expected_pix = [np.array(100), np.array(150)]

        self.assertListEqual([int(wcs_coords[0]), int(wcs_coords[1])], expected_pix)

    def test_get_target_ra_dec_without_moving_h1(self):
        target_name = self._split_target_name(self._header1["OBJECT"])
        moving_target = False
        ra_dec_dict = None
        result = get_target_ra_dec(
            target_name, moving_target, self._header1, ra_dec_dict
        )
        expected_result = tuple([self._header1["POINTRA"], self._header1["POINTDEC"]])

        self.assertTupleEqual(result, expected_result)

    def test_get_target_ra_dec_without_moving_h2(self):
        target_name = self._split_target_name(self._header2["OBJECT"])
        moving_target = False
        ra_dec_dict = None
        result = get_target_ra_dec(
            target_name, moving_target, self._header2, ra_dec_dict
        )
        expected_result = tuple([self._header2["POINTRA"], self._header2["POINTDEC"]])

        self.assertTupleEqual(result, expected_result)

    def test_get_target_ra_dec_moving_h1(self):
        target_name = self._split_target_name(self._header1["OBJECT"])
        moving_target = True
        ra_dec_dict = None  # This parameter is not needed for ephemeris lookup
        result = get_target_ra_dec(
            target_name, moving_target, self._header1, ra_dec_dict
        )
        expected_result = tuple([self._header1["POINTRA"], self._header1["POINTDEC"]])

        tolerance = 0.001

        self.assertTrue(abs(result[0] - expected_result[0]) < tolerance)
        self.assertTrue(abs(result[1] - expected_result[1]) < tolerance)

    def test_get_target_ra_dec_moving_h2(self):
        target_name = self._split_target_name(self._header2["OBJECT"])
        moving_target = True
        ra_dec_dict = None  # This parameter is not needed for ephemeris lookup
        result = get_target_ra_dec(
            target_name, moving_target, self._header2, ra_dec_dict
        )

        expected_result = tuple([self._header2["POINTRA"], self._header2["POINTDEC"]])

        tolerance = 0.1

        self.assertTrue(abs(result[0] - expected_result[0]) < tolerance)
        self.assertTrue(abs(result[1] - expected_result[1]) < tolerance)

    def test_get_target_ra_dec_moving_precalculated_ra_dec_h1(self):
        target_name = self._split_target_name(self._header1["OBJECT"])
        moving_target = True
        # Assuming ra_dec_dict is pre-calculated and available
        ra_dec_dict = {
            "time_index": [],  # Replace with the actual list of time objects
            "ra": [],  # Replace with the actual list of RA values
            "dec": [],  # Replace with the actual list of Dec values
        }
        result = get_target_ra_dec(
            target_name, moving_target, self._header1, ra_dec_dict
        )
        expected_result = []

        self.assertEquals(result, expected_result)

    def test_get_target_ra_dec_moving_precalculated_ra_dec_h2(self):
        target_name = self._split_target_name(self._header2["OBJECT"])
        moving_target = True
        # Assuming ra_dec_dict is pre-calculated and available
        ra_dec_dict = {
            "time_index": [],  # Replace with the actual list of time objects
            "ra": [],  # Replace with the actual list of RA values
            "dec": [],  # Replace with the actual list of Dec values
        }
        result = get_target_ra_dec(
            target_name, moving_target, self._header2, ra_dec_dict
        )
        expected_result = []

        self.assertEquals(result, expected_result)

    def test_get_target_pix_h1(self):
        ra = 3.6523833841657742
        dec = 11.851682228745984
        wcs_coords = get_target_pix(ra, dec, self._header1)

        expected_pix = [np.array(1011), np.array(1044)]

        self.assertListEqual([int(wcs_coords[0]), int(wcs_coords[1])], expected_pix)

    def test_get_target_pix_h2(self):
        ra = 4.423111699254016
        dec = 26.56427818078933

        wcs_coords = get_target_pix(ra, dec, self._header2)

        expected_pix = [np.array(5342), np.array(3747)]

        self.assertListEqual([int(wcs_coords[0]), int(wcs_coords[1])], expected_pix)

    def test_get_ccw_h1(self):
        result = get_ccw(self._header1)
        expected_result = -146.11877890989848
        self.assertEquals(result, expected_result)

    def test_get_ccw_h2(self):
        result = get_ccw(self._header2)
        expected_result = 100.01936844282955
        self.assertEquals(result, expected_result)

    def test_get_if_header_already_post_processed_h1_pho(self):
        result = get_if_header_already_post_processed(self._header1, "PHOTOMETRY")
        self.assertTrue(result)

    def test_get_if_header_already_post_processed_h1_ast(self):
        result = get_if_header_already_post_processed(self._header1, "ASTROMETRY")
        self.assertTrue(result)

    def test_get_if_header_already_post_processed_h2_pho(self):
        result = get_if_header_already_post_processed(self._header2, "PHOTOMETRY")
        self.assertTrue(result)

    def test_get_if_header_already_post_processed_h2_ast(self):
        result = get_if_header_already_post_processed(self._header2, "ASTROMETRY")
        self.assertTrue(result)

    def test_cat_input_from_header_1(self):
        result = cat_input_from_header(self._header1)
        expected_result = (
            SkyCoord(54.78885815 * u.deg, 11.85325566 * u.deg, frame="icrs"),
            0.8174028682644279,
            "SDSSg",
            0.508,
        )

        self.assertTrue(result[0].separation(expected_result[0]).arcsecond < 0.0001)
        self.assertEqual(result[1], expected_result[1])
        self.assertEqual(result[2], expected_result[2])
        self.assertEqual(result[3], expected_result[3])

    def test_cat_input_from_header_2(self):
        result = cat_input_from_header(self._header2)
        expected_result = (
            SkyCoord(66.42826274 * u.deg, 26.6216114 * u.deg, frame="icrs"),
            1.5842334595383938,
            "Lum",
            0.142,
        )

        self.assertTrue(result[0].separation(expected_result[0]).arcsecond < 0.0001)
        self.assertEqual(result[1], expected_result[1])
        self.assertEqual(result[2], expected_result[2])
        self.assertEqual(result[3], expected_result[3])
