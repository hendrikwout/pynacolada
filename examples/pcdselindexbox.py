import pynacolada as pcd
from Scientific.IO import NetCDF

# this is a similar functionality as the cdo's 'selindexbox' option.

# cdo gave an error for this dataset. It seems that cdo cannot process all grid types:
# $ cdo selindexbox,20,90,30,100 /home/hendrik/data/belgium_aq/rcm/aq09/stage1/int2lm/laf2009010100_urb_ahf.nc /home/hendrik/data/belgium_aq/rcm/aq09/stage1/int2lm/laf2009010100_urb_ahf_sel.nc
# cdo selindexbox: Unsupported grid type: generic
# 
# cdo selindexbox (Abort): Unsupported grid type!

# on the contrary, this functionality now works with pynacolada.pcd():  they are no restrictions on the grid type.

fin = NetCDF.NetCDFFile('/home/hendrik/data/belgium_aq/rcm/aq09/stage1/int2lm/laf2009010100_urb_ahf.nc','r')
fout = NetCDF.NetCDFFile('/home/hendrik/data/belgium_aq/rcm/aq09/stage1/int2lm/laf2009010100_urb_ahf_sel.nc','w')

dnamsel = ['rlon','rlat']
func = lambda x: x[20:90,30:100]
for evar in fin.variables:
    if ( ('rlat' in fin.variables[evar].dimensions) & \
        ('rlon' in fin.variables[evar].dimensions)):
        datin =  [{'file':fin,'varname':evar,}]
        datout = [{'file':fout,'varname':evar},]
        pcd.pcd(func,dnamsel,datin,datout,appenddim=True)

fout.close();print('output file written to:',fout )
fin.close()
# Unfortunatly, It not superfast. The reason is that the 'preperations' in pcd needs to be redone
# for each variable.

