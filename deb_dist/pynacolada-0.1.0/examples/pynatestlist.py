import pynacolada as pcl
from Scientific.IO import NetCDF
import os
import numpy as np
import pylab as pl

fnins = ['/home/hendrik/data/belgium_aq/rcm/aq09/stage1/int2lm/laf2009010100_urb_ahf.nc' , '/home/hendrik/data/belgium_aq/rcm/aq09/stage1/int2lm/laf2009010100_urb_ahf2.nc']
#print fnin
# fobjin = open(fnin,'rb')
# fin = NetCDF.NetCDFFile(fnin,'r')
fnout = '/home/hendrik/data/belgium_aq/rcm/aq09/stage1/int2lm/laf2009010100_urb_ahf3.nc'
os.system('rm '+fnout)
#print fnout
# fobjout = open(fnout,'wb+''rlat')
fout = NetCDF.NetCDFFile(fnout,'w')
datin =  [[fnins,'T'],[fnins,'rlat']]
datout = [[fout,'T'],]
# selection of function dimension input
func = lambda x, y: (np.array([np.mean(x,axis=0)],dtype=np.float32) ,) # *(1.+np.zeros(x.shape))
dnamsel = ['time',]
pcl.pcl(func,dnamsel,datin,datout,predim = 'time')

fout.close()


# fig = pl.figure()
# fout = NetCDF.NetCDFFile(fnout,'r')
# pl.imshow(fout.variables['T'][:].squeeze())
# fig.show()
# fout.close()
