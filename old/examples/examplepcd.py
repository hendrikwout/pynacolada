import pynacolada as pcd
import numpy as np
import datetime as dt
import sciproc as sp
from time import sleep
from Scientific.IO import NetCDF

# purpose: A short tutorial of using pynacolada.pcd(). The example should work with COSMO-CLM data output.
# author: Hendrik Wouters <hendrik.wouters@ees.kuleuven.be>

print('example 1: calculate the mean temperature of entire vertical columns (per lat-lon grid cell)')
fnin ='/media/URB_UIP_2/data/rcm/0.009_20020101_berlin/cclm/out01/lffd2002050100.nc'
print('reading:',fnin)
fin = NetCDF.NetCDFFile(fnin,'r')
datin =  [{'file': fin, \
           'varname': 'T', \
          },\
         ]

fnout = '/media/URB_UIP_2/data/rcm/0.009_20020101_berlin/cclm/anatest0.nc'

fout = NetCDF.NetCDFFile(fnout,'w')
datout = [{'file': fout, \
           'varname': 'T'}]
print(' the function definition: take the mean along the axis. Note that we need to preserve the "amount" of dimensions of the output data compared to the input data.')

func = lambda x: np.array([np.mean(x,axis=0 )])
dnamsel = ['level',]
pcd.pcd(func,dnamsel,datin,datout)

fout.close()
fin.close()

print("previous example didn't go very fast! We try again with the 'example 1', in which we use the appenddim-option, and just generating exactly the same output! It results in only one call of the function, instead of 38025 times in a some python loop")
sleep(5)
print("example 1: bis... calculate the mean temperature of entire vertical columns (per lat-lon grid cell)")
fnin ='/media/URB_UIP_2/data/rcm/0.009_20020101_berlin/cclm/out01/lffd2002050100.nc'
print('reading:',fnin)
fin = NetCDF.NetCDFFile(fnin,'r')
datin =  [{'file': fin, \
           'varname': 'T', \
          },\
         ]

fnout = '/media/URB_UIP_2/data/rcm/0.009_20020101_berlin/cclm/anatest1.nc'

fout = NetCDF.NetCDFFile(fnout,'w')
datout = [{'file': fout, \
           'varname': 'T'}]

func = lambda x: np.array([np.mean(x,axis=0 )])
dnamsel = ['level',]
# here is the appenddim option -> it result in only one call of the function, instead of 38025- times!
pcd.pcd(func,dnamsel,datin,datout,appenddim=True)

fout.close()
fin.close()
sleep(5)
print("That was much faster isn't it?")

print('example 2: calculate the mean temperature of the first 10 layers (per lat-lon grid cell)')
fnin ='/media/URB_UIP_2/data/rcm/0.009_20020101_berlin/cclm/out01/lffd2002050100.nc'
print('reading:',fnin)
fin = NetCDF.NetCDFFile(fnin,'r')
datin =  [{'file': fin, \
           'varname': 'T', \
# we specify a 'subspace' using the 'dsel' option
           'dsel': {'level' : range(30,40,1)}, \
          },\
         ]

fnout = '/media/URB_UIP_2/data/rcm/0.009_20020101_berlin/cclm/anatest2.nc'

fout = NetCDF.NetCDFFile(fnout,'w')
datout = [{'file': fout, \
           'varname': 'T'}]
func = lambda x: np.array([np.mean(x,axis=0 )])
dnamsel = ['level',]
pcd.pcd(func,dnamsel,datin,datout,appenddim=True)

fout.close()
fin.close()

print(' example 3: calculate the scalar horizontal mean wind speed of the first 10 layers ')
fnin ='/media/URB_UIP_2/data/rcm/0.009_20020101_berlin/cclm/out01/lffd2002050100.nc'
print('reading:',fnin)
fin = NetCDF.NetCDFFile(fnin,'r')
datin =  [{'file': fin, \
           'varname': 'U', \
           'dsel': {'level' : range(30,40,1)}, \
# The U and V wind speed components are not exactly on the same grid so have other (but similar) coordinates ! Therefore, we need to define dimension aliases (thus preventing pcd from generating a huge dataset)!
           'daliases': { 'srlat':'rlat', 'srlon':'rlon' },\
          },\
# a second variable input definition
          {'file': fin, \
           'varname':'V', \
           'dsel': {'level' : range(30,40,1)},
# Same here as for the U component....
           'daliases': { 'srlat':'rlat', 'srlon':'rlon' },\
           }\
         ]

fnout = '/media/URB_UIP_2/data/rcm/0.009_20020101_berlin/cclm/anatest3.nc'

fout = NetCDF.NetCDFFile(fnout,'w')
datout = [{'file': fout, \
           'varname': 'u'}]
func = lambda U,V: np.array([np.mean(np.sqrt(U**2+V**2),axis=0 )])
dnamsel = ['level',]
pcd.pcd(func,dnamsel,datin,datout,appenddim=True)
# Note also that the calculation is not entirely correct because the U and V loacations are shifted by 1/2-gridcell!

fout.close()
fin.close()


print("example 4: now we come to it's real power. Do the same as before, but for an hourly time series. ")

# the timeseries: just a list of all the files in the right order
dts = sp.dtrange(dt.datetime(2002,5,1),dt.datetime(2002,7,1),dt.timedelta(hours=1))
path = '/media/URB_UIP_2/data/rcm/0.009_20020101_berlin/cclm/out01/'
dtfiles = [path+'lffd'+dt.datetime.strftime(e,"%Y%m%d%H")+'.nc' for e in dts]

datin =  [{'file': dtfiles, \
           'varname': 'U', \
           'dsel': {'level' : range(30,40,1)}, \
           'daliases': { 'srlat':'rlat', 'srlon':'rlon' },\
          },\
          {'file': dtfiles, \
           'varname':'V', \
           'dsel': {'level' : range(30,40,1)},
           'daliases': { 'srlat':'rlat', 'srlon':'rlon' },\
           }\
         ]

fnout = '/media/URB_UIP_2/data/rcm/0.009_20020101_berlin/cclm/anatest4.nc'

fout = NetCDF.NetCDFFile(fnout,'w')
datout = [{'file': fout, \
           'varname': 'u'}]

# we write the datetimes to the netcdf file (we could do it also afterwards after the pcd.pcd()-call)
pcd.ncwritedatetime(fout,dts,tunits='days', refdat=dt.datetime(2002,1,1))

# function definition:
func = lambda U,V: np.array([np.mean(np.sqrt(U**2+V**2),axis=0 )])
dnamsel = ['level',]
pcd.pcd(func,dnamsel,datin,datout,appenddim=True)

fout.close(); print 'output data written to:',fnout
print('and enjoy watching your processed data with ncview!')


