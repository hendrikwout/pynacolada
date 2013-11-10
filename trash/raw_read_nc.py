import scipy.io as io
import numpy as np


self = io.netcdf.netcdf_file('/home/hendrik/data/global/AHF_2005_2.5min.nc','r')
self.fp.seek(0)
magic = self.fp.read(3)
self.__dict__['version_byte'] = np.fromstring(self.fp.read(1), '>b')[0]

# Read file headers and set data.
# stolen from scipy: /usr/lib/python2.7/dist-packages/scipy/io/netcdf.py
self._read_numrecs()
self._read_dim_array()
self._read_gatt_array()
# self._read_var_array()
header = self.fp.read(4)
begin = 0
count = self._unpack_int()
vars = []
for ivar in range(count):
    vars.append(self._read_var())

