import pickle
import pylab as pl
from operator import itemgetter
import scipy.io as io
import numpy as np
import sys 

# to do next:
# output to netcdf
# implement function passing: done
# implement multiple variables: output variable
# implement multiple variable arrays
# implement wildcard at the end
# implement warning messages:
# maxicecubesize should be in (mega/kilo)bytes

class SomeError(Exception):
     def __init__(self, value):
         self.value = value
     def __str__(self):
         return repr(self.value)

def nctypecode(dtype):
    # purose: netcdf-typecode from array-dtype
    if ((dtype == dtype('float32')) or (dtype == 'float32')):
        return 'f'
    elif ((dtype == dtype('float64')) or (dtype == 'float64')):
        return 'd'
    elif ((dtype == dtype('int32')) or (dtype == 'int32')):
        return 'i'
    elif ((dtype == dtype('int64')) or (dtype == 'int64')):
        return 'l'

def ncdtype(typecode):
    # purose: get array-dtype from netcdf-typecode
    if typecode == 'f':
        return np.dtype('float32')
    elif typecode == 'd':
        return np.dtype('float64')
    elif typecode == 'i':
        return dtype('int32')
    elif typecode == 'l':
        return dtype('int64')

def rwicecube(filestream,shp,refiter,dimiter,dimpos,refnoiter,dimnoiter,icecube,vtype,vsize,voffset,rwchsize,mode):
    """
    read or write data icecube from binary data and put it in an array
    filestream: binary file reference
    shp: shape of the filestream
    refiter: reference to dimensions over which no slice is performed
    pos: current index position of the non-sliced dimensions
    """
    
    # e.g. shp = (200,100,50,50,20)
    #      refiter = (1,3,4)
    #      dimpos = (5,10,9)
    
    # extend so that structured arrays are read at once
    
    lennoiter = long(1)
    for irefnoiter,erefnoiter in enumerate(refnoiter):
        lennoiter = lennoiter*dimnoiter[irefnoiter]

    fpos = 0
    # e.g. fpos = (9)+ 20*(10) + 50*50*20*(5)
    for idimpos,edimpos in enumerate(dimpos):
        curadd = edimpos
        #e.g. if edimpos == (5): curadd = 50*50*20*(5)

        # exclude trivial special case of only 1 iteration step
        # --> in that case fpos is just zero.
        if refiter != [-1]:
            if ((refiter[idimpos] + 1) < len(shp)):
                for i in range(refiter[idimpos] + 1,len(shp)) :
                    curadd = curadd * shp[i]
    
        fpos = fpos + curadd

    # Initialize (for reading) or prepare (for writing) icecube array
    if mode == 'read':
        icecube = np.zeros((lennoiter,),dtype=vtype)*np.nan

    elif mode == 'write':
        icecube = np.reshape(icecube,(lennoiter,))

    dimnoiterpos = [0]*len(dimnoiter)
    # print icecube,dimnoiterpos
    j = 0
    while j < lennoiter:
        fposicecube = fpos
        for idimpos,edimpos in enumerate(dimnoiterpos):
            curadd = edimpos
            # e.g. fposicecube = (1)*52
            # e.g. fposicecube = (9)+ 20*(10) + 50*50*20*(5)
            if ((refnoiter[idimpos] + 1) < len(shp)):
                for i in range(refnoiter[idimpos] + 1,len(shp)) :
                    curadd = curadd * shp[i]
    
            fposicecube = fposicecube + curadd
    
        filestream.seek(voffset+vsize*fposicecube)
        
        if mode == 'read':
            icecube[j:(j+rwchsize)] = np.fromfile(filestream,dtype=vtype,count=rwchsize)
        elif mode == 'write':
            filestream.seek(vsize*fposicecube)
            filestream.write(icecube[j:(j+rwchsize)])

    
        # go to next data strip 
        if dimnoiterpos != []:
            # rwchsize: allow reading of chunks for the inner dimensions
            dimnoiterpos[-1] = dimnoiterpos[-1] + rwchsize
            for idimidx,edimidx in enumerate(reversed(dimnoiterpos)):
                if idimidx > 0:
                    while dimnoiterpos[idimidx] >= dimnoiter[idimidx]:
                        dimnoiterpos[idimidx-1] = dimnoiterpos[idimidx-1] + 1
                        dimnoiterpos[idimidx] -= dimnoiter[idimidx]
        j = j+rwchsize
    
    icecube.shape = dimnoiter
    if mode == 'read':
        return icecube

def readicecubeps(fstream,shp,refiter,dimiter,dimiterpos,refnoiter,dimnoiter,vtype,vsize,voffset,rwchsize):
    """ 
    read an icecube by sorting the indices (highest at the back).
    perform an in-memory Post Swap of dimensions (very fast) to compensate for the sorting.
    we allow reading in chunks according to the inner dimensions. They will be mostly there because we allow an max-icecubesize
    """

    icecube =rwicecube(fstream,shp,refiter,dimiter,dimiterpos,sorted(refnoiter),sorted(dimnoiter),None,vtype,vsize,voffset,rwchsize,'read') 

    trns = zip(*sorted(zip(refnoiter,range(len(refnoiter))),key=itemgetter(0,1)))[1]

    # build the 'inverse permutation' operator for tranposition before writeout
    inv = range(len(trns))
    for itrns, etrns in enumerate(trns):
        inv[etrns] = itrns
    return np.transpose(icecube,inv)

def writeicecubeps(fstream,shp,refiter,dimiter,dimiterpos,refnoiter,dimnoiter,data,vtype,vsize,voffset,rwchsize):
    """ 
    write an icecube and perform an in-memory Post Swap of dimensions before (very fast)
    hereby, we acquire the order of the icecube dimensions
    """
    trns = zip(*sorted(zip(refnoiter,range(len(refnoiter))),key=itemgetter(0,1)))[1]
    rwicecube(fstream,shp,refiter,dimiter,dimiterpos,sorted(refnoiter),sorted(dimnoiter),np.transpose(data,trns),vtype,vsize,voffset,rwchsize,'write') 




# fin = open('/home/hendrik/data/belgium_aq/rcm/aq09/stage1/aurorabc/hour16_beleuros.bin','r')
# shp = (4,36,52);vtype=np.float32; vsize=4; voffset=0


def sizeout(shp,func,refapplyout):
    lenapply = long(1)
    dimapply = []
    for irefapplyout in range(len(refapplyout)):
        dimapply.append(shp[refapplyout[irefapplyout]])
        lenapply = lenapply*dimapply[-1]

    dimapplyout = np.array(func(np.zeros(dimapply))).shape

    if (len(dimapplyout)!=len(dimapply)):
        raise SomeError("The number of input dimensions (= "+str(len(dimapply))+") and output dimensions (= "+str(len(dimapplyout))+") resulting from the specified function need to be the same.")

    return dimapplyout,lenapply

def shake(fin,shp,func,refapplyout,fout,refiter=None,maxicecubesize=100000):
    """
    purpose 
    -------

    swap specified dimensions to the back efficiently in a specified order

    input parameters
    ----------------

    fin: binary file input stream
    fout: binary file output stream
    shp: shape of the data stream
    refapplyout: dimensions over which the function is applied
    refiter (optional): reference to dimensions that are swapped to the front. The order of those indices are
    taken into account.  Of not specified, it is guessed from the residual dimensions (defined in shp) that are not in refnoiter
    """
    dimapplyout,lenapply = sizeout(shp,func,refapplyout)

    # we want to read the data in chunks (icecubes) as big as possible. In the first place, the data chunks contain of course the dimensions on which the functions are applied. Afterwards, the chunk dimensions is extended (in the outer(!) direction) to make the icecubes bigger.
    # refnoiter: reference to dimensions that are swapped to the back. In any case, this needs to include all refapplyouts. Data in these dimensions are read in icecubes. The order of those indices are taken into account. We also add in front those dimensions that can be read at once (still needs to be tested!). 

    # the total length of the numpy array as IO memory buffer ('icecubes'). The programm will try to read this in chunks (cfr. rwchunksize- as large as possible. An in-memory transposition may be applied after read or before writing.
    lennoiter = 1
    # the total length of data passed to function 
    lenapply = 1
    # the dimension of data passed to function 
    dimapply = []
    # the dimension of the IO buffer array
    dimnoiter = []
    refnoiter = []
    for erefapplyout in refapplyout:
        # refapplyout.append(erefapplyout)
        dimapply.append(shp[erefapplyout])
        lenapply = lenapply*shp[erefapplyout]

        refnoiter.append(erefapplyout)
        dimnoiter.append(shp[erefapplyout])
        lennoiter = lennoiter*shp[erefapplyout]

    if lenapply > maxicecubesize:
        print 'Warning, the function data input length of',lenapply,' (dimensions: ',dimapply,') exceeds the maximum icecubesize of '+str(maxicecubesize)+'.' 
    else:
        idim = len(shp)-1
        edim = shp[idim]
        while ((idim >= 0) & ((lennoiter*edim) < maxicecubesize)):
            if (idim not in refnoiter):
                refnoiter.insert(0,idim)
                dimnoiter.insert(0,edim)
                lennoiter = lennoiter*edim
                # print 'yeeps',idim,edim,refnoiter,dimnoiter,lennoiter, maxicecubesize
            idim = idim - 1 
            edim = shp[idim]
        print 'Icecubesize is: ',lennoiter,dimnoiter,refnoiter


    # refnoiterout = list(refnoiter)
    dimnoiterout = list(dimnoiter)
    dimnoiterout[(len(dimnoiterout) - len(dimapply)):] = list(dimapplyout)

    lennoiterout = 1
    for edimnoiterout in dimnoiterout:
        lennoiterout = lennoiterout*edimnoiterout


    lenapplyout = long(1)
    for edim in dimapplyout:
        lenapplyout = lenapplyout*edim

    dimiter = []
    leniter = long(1)

    # guess from residual dimensions that are not in refnoiter
    if refiter == None:
        refiter = []
        for ishp,eshp in enumerate(shp):
            if ishp not in refnoiter:
                refiter.append(ishp)
    for erefiter in refiter:
        dimiter.append(shp[erefiter])
        leniter = leniter*dimiter[-1]

    # the trivial case of only one iteration
    if dimiter == []:
        dimiter = [1]
        dimiterpos = [0]
        refiter = [-1]
    else:
        dimiterpos = [0]*len(refiter)

    shpout = [None]*len(shp)
    if refiter != [-1]:
        for irefiter,erefiter in enumerate(refiter):
            shpout[erefiter] = dimiter[irefiter]

    for irefnoiter,erefnoiter in enumerate(refnoiter):
        shpout[erefnoiter] = dimnoiterout[irefnoiter]

    # get the maximum size of continuous data chunks for more efficient IO
    rwchunksize = 1
    rwchunksizeout = 1

    ishp = len(shp)-1
    while ishp in refnoiter:
        rwchunksize = rwchunksize*shp[ishp]
        rwchunksizeout = rwchunksizeout*shpout[ishp]
        ishp = ishp-1

    print ' rwchunksize:   ',rwchunksize
    print ' rwchunksizeout:',rwchunksizeout

    print str(0)+'/'+str(leniter),
    for j in range(leniter):
        # reading icecube, rearranged in the order of dimensions specified by refnoiter
        dataicecube = np.array(readicecubeps(fin,shp,refiter,dimiter,dimiterpos,refnoiter,dimnoiter,vtype,vsize,voffset,rwchunksize),dtype=vtype).ravel()
        # temporary store output in a array-buffer
        dataicecubeout = np.zeros((lennoiterout,),dtype=vtype)

        # crush the ice
        # refnoiter = (6 ,7 ,8 ,4 ,5)
        # dimiter      = (30,20,15,20,15)
        # refapplyout  =       (8 ,4 ,5)

        # get the dimensions of the buffer over which we iterate
        # we know that the function are applied along dimensions that are at the inner data
        dimnoapply = []
        lennoapply = long(1)
        for irefnoiter in range(len(refnoiter)-len(refapplyout)):
            dimnoapply.append(dimnoiter[irefnoiter])
            lennoapply = lennoapply*dimnoapply[-1]

        if dimnoapply == []:
            dimnoapply = [1]

        dimnoapplypos = [0]*len(dimnoapply)
        for k in range(lennoapply):
            # actually, this is just the end of the file output already written
            apos    = 0
            # e.g. apos = (9)+ 20*(10) + 50*50*20*(5)
            for idimpos,edimpos in enumerate(dimnoapplypos):
                curadd    = edimpos
                #e.g. if edimpos == (5): curadd = 50*50*20*(5)
                if ((idimpos + 1) < len(refnoiter)):
                    for i in range(idimpos + 1,len(refnoiter)) :
                        curadd    = curadd    * dimnoiter[i]
                        # curaddout = curaddout * dimnoiteroutref[i]
                apos    = apos    + curadd

            aposout    = 0
            # e.g. aposout = (9)+ 20*(10) + 50*50*20*(5)
            
            for idimpos,edimpos in enumerate(dimnoapplypos):
                curadd    = edimpos
                #e.g. if edimpos == (5): curadd = 50*50*20*(5)
                if ((idimpos + 1) < len(refnoiter)):
                    for i in range(idimpos + 1,len(refnoiter)) :
                        curadd    = curadd    * dimnoiterout[i]
                        # curaddout = curaddout * dimnoiteroutref[i]
                aposout    = aposout    + curadd

            hunk = dataicecube[apos:(apos+lenapply)]
            hunk.shape = dimapply

            # apply the function
            hunkout = np.array(func(hunk)) #np.array((np.zeros(hunk.shape) + 1)*np.mean(hunk),dtype=vtype)
            dataicecubeout[aposout:(aposout+lenapplyout)] = np.array(hunkout[:].ravel(),dtype=vtype)

            # go to next data slice  
            dimnoapplypos[-1] = dimnoapplypos[-1] + 1
            for idimidx,edimidx in enumerate(reversed(dimnoapplypos)):
                # # alternative (makes 'dimiter' redundant)
                # if dimiterpos[idimidx] == shp[refiter[idimidx]]:
                if idimidx > 0:
                    if dimnoapplypos[idimidx] == dimnoapply[idimidx]:
                       dimnoapplypos[idimidx-1] = dimnoapplypos[idimidx-1] + 1
                       dimnoapplypos[idimidx] = 0
        
        dataicecubeout.shape = dimnoiterout
        writeicecubeps(fout,shpout,refiter,dimiter,dimiterpos,refnoiter,dimnoiterout,dataicecubeout,vtype,vsize,voffset,rwchunksizeout)

        # go to next data slice  
        dimiterpos[-1] = dimiterpos[-1] + 1
        for idimidx,edimidx in enumerate(reversed(dimiterpos)):
            # # alternative (makes 'dimiter' redundant)
            # if dimiterpos[idimidx] == shp[refiter[idimidx]]:
            if dimiterpos[idimidx] == dimiter[idimidx]:
                if idimidx > 0:
                    dimiterpos[idimidx-1] = dimiterpos[idimidx-1] + 1
                    dimiterpos[idimidx] = 0
        sys.stdout.write ('\b'*(len(str(j)+'/'+str(leniter))+1))
        sys.stdout.write (str(j+1)+'/'+str(leniter))

def ncvartypeoffset(ncfile,var):
    """ purpose: get binary data type and offset of a variable in netcdf file
        unfortunately, getting these properties are not explicitely implemented in scipy, but most of this code is stolen from scipy: /usr/lib/python2.7/dist-packages/scipy/io/netcdf.py
        ncfile is a scipy.io.netcdf.netcdf_file
        var variable we want to calculate the offset from
    """
    oripos=ncfile.fp.tell()
    ncfile.fp.seek(0)
    magic = ncfile.fp.read(3)
    ncfile.__dict__['version_byte'] = np.fromstring(ncfile.fp.read(1), '>b')[0]
    
    # Read file headers and set data.
    ncfile._read_numrecs()
    ncfile._read_dim_array()
    ncfile._read_gatt_array()
    header = ncfile.fp.read(4)
    count = ncfile._unpack_int()
    vars = []
    for ic in range(count):
        vars.append(list(ncfile._read_var()))
    var = 'QV'
    ivar =  np.where(np.array(vars) == var)[0][0]
    ncfile.fp.seek(oripos)
    return vars[ivar][6] , vars[ivar][7]

var = 'QV'
ncfile = io.netcdf.netcdf_file('/home/hendrik/data/belgium_aq/rcm/aq09/stage1/int2lm/laf2009010100_urb_ahf.nc','r')
shp = ncfile.variables[var].shape 
vsize = ncfile.variables[var].itemsize() 
vtype, voffset = ncvartypeoffset(ncfile,var)
fin = ncfile.fp
refapplyout= (1,)

fout = io.netcdf.netcdf_file('/home/hendrik/data/belgium_aq/rcm/aq09/stage1/int2lm/laf2009010100_urb_ahf2.nc','w')



# here




        


    

# fout = open('/home/hendrik/data/belgium_aq/rcm/aq09/stage1/aurorabc/hour16_beleuros2.bin','wb')
# def readicecube(filestream,shp,refiter,dimpos,refnoiter=None):
# testdat = readicecubeps(      fin,       shp,(1,),    (2,),refnoiter=(1,0))

# def shake(fin,shp,refapplyout,fout,vtype,vsize,voffset,refiter=None,maxicecubesize=10000):
# shake(      fin,shp,(1,2),offset,vtype,vsize,voffset,refiter=None,maxicecubesize=10000)


func = lambda x: [np.mean(x)] # *(1.+np.zeros(x.shape))

shake(fin,shp,func,refapplyout,fout,refiter=None,maxicecubesize=1000000000)

fout.close()
fin.close()
fread = open('/home/hendrik/data/global/test.bin','r')

ipol = 0

nx = shp[3]
ny = shp[2]
nz = 1# shp[0]
iz = 0

fig = pl.figure()
fread.seek((ipol*nz + iz)*ny*nx*vsize,0)
field = np.fromfile(fread,dtype=vtype,count=ny*nx)
field.shape = (ny,nx)
pl.imshow(field)
fig.show()
fread.close()
