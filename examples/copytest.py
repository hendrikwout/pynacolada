import pynacolada as pcd
from Scientific.IO import NetCDF
fin = NetCDF.NetCDFFile('/home/hendrik/data/belgium_aq/rcm/aq09/stage1/int2lm/laf2009010100_urb_ahf.nc','r')
fout = NetCDF.NetCDFFile('/home/hendrik/data/belgium_aq/rcm/aq09/stage1/int2lm/laf2009010100_urb_ahf_test.nc','w')
pcd.nccopyvariable(fin,fout,'lat',copyvalues=True)
pcd.nccopyvariable(fin,fout,'lon',copyvalues=True)
pcd.nccopyvariable(fin,fout,'rlon',copyvalues=True)
pcd.nccopyvariable(fin,fout,'rlat',copyvalues=True)
pcd.nccopyvariable(fin,fout,'T',copyvalues=True)
pcd.nccopyvariable(fin,fout,'U',copyvalues=True)
pcd.nccopyvariable(fin,fout,'V',copyvalues=True)
pcd.nccopyattrfile(fin,fout)
fin.close()
fout.close()

