#!/usr/bin/env python

# extract datetimes from netcdf file
from Scientific.IO import NetCDF
from pynacolada import ncgetdatetime

ncread = NetCDF.NetCDFFile('/home/hendrik/data/toulouse_project/Sites/TLcenter/outputterra/17wt01/data-extra.nc')
print ncgetdatetime(ncread)

ncread.close()
