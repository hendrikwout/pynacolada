import pynacolada as pcd
import numpy as np
import datetime as dt
from Scientific.IO import NetCDF

csvfile = open('data/example.csv', 'r')
ncfile = NetCDF.NetCDFFile('data/example.nc','w')

pcd.csv2netcdf(csvfile,ncfile,sep=',',formatlist=[[dt.datetime,'%m/%d/%Y %H:%M']]+[[np.double,np.nan]]*15,refdat=dt.datetime(2009,1,1,0,0), tunits='hours')
ncfile.close(); print 'data written to: ',ncfile
