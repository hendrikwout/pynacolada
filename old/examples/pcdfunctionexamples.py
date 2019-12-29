
import pynacolada as pcd
import numpy as np
import datetime as dt
import sciproc as sp
from Scientific.IO import NetCDF

# purpose: More examples for pynacolada.pcd() defining more advanced functions

# example 1: calculate running time-average of a dataset.

# Note: this function is included in sciproc ('from sciproc import avgglide'), but here it is explicitly given as an example
def avgglide(x,avlen):
    xout = np.zeros_like(x)
    xwork = np.zeros_like(x)
    if np.mod(avlen,2) == 0:
        xwork[0] = np.nan
        xwork[:-1] = (x[1:] + x[:-1])/2.
    else:
        xwork = x
    lenxout = xout.shape[0]
    avlen2 = int(avlen/2.)
    xout[:avlen2] = np.nan
    xout[-avlen2:] = np.nan
    for i in range(0,avlen,1):
        xout[avlen2:(-avlen+avlen2)] = xout[avlen2:(-avlen+avlen2)] + xwork[i:(-avlen+i)]
    return xout/avlen

# now, apply weekly cycle to a netcdf file
ncin = NetCDF.NetCDFFile('input.nc','r')
ncglide = NetCDF.NetCDFFile('output_glide.nc','w')
# we can cycle through the netcdf variables that have the 'time'-dimension
for evar in ncin.variables:
    if ((evar not in ncin.dimensions) & ('time' in ncin.variables[evar].dimensions):
        pcd.pcd(lambda x: avgglide(x,168),\
                ('time',),\
                [{'file' : ncin, 'varname': evar}],\
                [{'file' : ncglide, 'varname': evar}])
ncglide.close(); print('data written to:', ncglide)
ncin.close()


