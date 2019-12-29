import pynacolada as pcd
from Scientific.IO import NetCDF
import os
import numpy as np
import pylab as pl

fnins = ['/home/hendrik/data/belgium_aq/rcm/aq09/stage1/int2lm/laf2009010100_urb_ahf.nc' , '/home/hendrik/data/belgium_aq/rcm/aq09/stage1/int2lm/laf2009010100_urb_ahf2.nc']
#print fnin
# fobjin = open(fnin,'rb')
# fin = NetCDF.NetCDFFile(fnin,'r')
fout = NetCDF.NetCDFFile('/home/hendrik/data/belgium_aq/rcm/aq09/stage1/int2lm/laf2009010100_urb_ahf3.nc','w')
#os.system('rm '+fnout)
#print fnout
# fobjout = open(fnout,'wb+''rlat')
datin =  [{'file':fnins,'varname':'T','predim':'time'},\
          {'file':fnins,'varname':'rlat','predim':'time'}]
datout = [{'file':fout,'varname':'T'},]
# selection of function dimension input
func = lambda x, y: (np.array([np.mean(x,axis=0)],dtype=np.float32) ,) # *(1.+np.zeros(x.shape))
dnamsel = ['time',]
pcd.pcd(func,dnamsel,datin,datout,appenddim=True)

fout.close();print('output file written to:',fout )


# fig = pl.figure()
# fout = NetCDF.NetCDFFile(fnout,'r')
# pl.imshow(fout.variables['T'][:].squeeze())
# fig.show()
# fout.close()
