import pynacolada as pcd
from Scientific.IO import NetCDF

# this is a similar functionality as the cdo's 'selindexbox' option.

# cdo gave an error for this dataset. It seems that cdo cannot process all grid types:
# $ cdo selindexbox,20,90,30,100 /home/hendrik/data/belgium_aq/rcm/aq09/stage1/int2lm/laf2009010100_urb_ahf.nc /home/hendrik/data/belgium_aq/rcm/aq09/stage1/int2lm/laf2009010100_urb_ahf_sel.nc
# cdo selindexbox: Unsupported grid type: generic
# 
# cdo selindexbox (Abort): Unsupported grid type!

# on the contrary, this functionality now works with pynacolada.pcd():  they are no restrictions on the grid type.

# example 1: crop the domain for the variable 'T'
fin = NetCDF.NetCDFFile('/home/hendrik/data/belgium_aq/rcm/aq09/stage1/int2lm/laf2009010100_urb_ahf.nc','r')
fout = NetCDF.NetCDFFile('/home/hendrik/data/belgium_aq/rcm/aq09/stage1/int2lm/laf2009010100_urb_ahf_sel.nc','w')

func = lambda x: x[20:90,30:100]
dnamsel = ['rlon','rlat']
evar='T'

print ('processing variable: ',evar)
datin =  [{'file':fin,'varname':evar,}]
datout = [{'file':fout,'varname':evar},]
pcd.pcd(func,dnamsel,datin,datout,appenddim=True)

fout.close();print('output file written to:',fout )
fin.close()
# Unfortunatly, It not superfast. The reason is that the 'preperations' in pcd needs to be redone
# for each variable.

# example 2: crop the domain to all variables that involve the lat-lon grid, and just copy any other variable
fin = NetCDF.NetCDFFile('/home/hendrik/data/belgium_aq/rcm/aq09/stage1/int2lm/laf2009010100_urb_ahf.nc','r')
fout = NetCDF.NetCDFFile('/home/hendrik/data/belgium_aq/rcm/aq09/stage1/int2lm/laf2009010100_urb_ahf_sel2.nc','w')

func = lambda x: x[20:90,30:100]
dnamsel = [['rlon','srlon'],['rlat','srlat']]
for evar in fin.variables:
    # apply the domain selection on all variables that involve these dimensions
    procvar = True
    for ednamsel in dnamsel:
        found = False
        if type(ednamsel).__name__ == 'list':
            for eednamsel in ednamsel:
                if eednamsel in fin.variables[evar].dimensions:
                    found = True
        elif ednamsel in fin.variables[evar].dimensions:
            found = True
        if found == False:
            procvar = False

    if procvar:
        print ('processing variable: ',evar)
        datin =  [{'file':fin,'varname':evar,}]
        datout = [{'file':fout,'varname':evar},]
        pcd.pcd(func,dnamsel,datin,datout,appenddim=True)
    else:
        if evar not in fin.dimensions:
            print ('copying variable: ',evar)
            pcd.nccopyvariable(fin,fout,evar,copyvalues=True)

fout.close();print('output file written to:',fout )
fin.close()
# Unfortunatly, It not superfast. The reason is that the 'preperations' in pcd needs to be redone
# for each variable.

