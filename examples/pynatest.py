#!/usr/bin/env python

# Example 1: calculate the mean scalar wind speed of the first 10 layers
import pynacolada as pcl
from Scientific.IO import NetCDF
import numpy as np

fnin = '/home/hendrik/data/belgium_aq/rcm/aq09/stage1/int2lm/laf2009010100_urb_ahf.nc'
#print fnin
# fobjin = open(fnin,'rb')
fin = NetCDF.NetCDFFile(fnin,'r')
fnout = 'laf2009010100_out.nc'
#print fnout
# fobjout = open(fnout,'wb+''rlat')
fout = NetCDF.NetCDFFile(fnout,'w')
# input data definitions
datin =  [{'file': fin, \
           'varname': 'U', \
           'dsel': {'level' : range(30,40,1)}, \
           'daliases': { 'srlat':'rlat', 'srlon':'rlon' },\
          },\
          {'file': fin, \
           'varname':'V', \
           'dsel': {'level' : range(30,40,1)},
           'daliases': { 'srlat':'rlat', 'srlon':'rlon' },\
           }\
         ]
# output data definitions
datout = [{'file': fout, \
           'varname': 'u'}]
# function definition:
func = lambda U,V: np.array([np.mean(np.sqrt(U**2+V**2),axis=0 )])
dnamsel = ['level',]
pcl.pcl(func,dnamsel,datin,datout,appenddim=True)
print 'output data written to:',fnout
