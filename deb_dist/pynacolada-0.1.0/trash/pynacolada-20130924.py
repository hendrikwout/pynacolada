import numpy as np
import pickle
from numpy import fromfile
import pylab as pl
from operator import itemgetter
import scipy.io as io
import numpy as np

# print readicecube(fin,(1,35,52),(5,),(3,))
def readicecube(filestream,shp,dimiterref,dimpos,icecube,mode='read',dimnoiterref=None):
    """
    read data icecube from binary data and put it in an array
    filestream: binary file reference
    shp: shape of the filestream
    dimiterref: reference to dimensions over which no slice is performed
    pos: current index position of the non-sliced dimensions
    """
    
    # e.g. shp = (200,100,50,50,20)
    #      dimiterref = (1,3,4)
    #      dimpos = (5,10,9)
    
    # extend so that structured arrays are read at once
    
    dimiter = []
    dimnoiter = []
    lennoiter = long(1)
    for i in range(len(shp)):
        if i in dimiterref:
            dimiter.append(shp[i])
    if dimnoiterref == None:
        dimnoiterref = []
        for i in range(len(shp)):
            if i not in dimiterref:
                dimnoiterref.append(i)
                dimnoiter.append(shp[i])
                lennoiter = lennoiter*shp[i]
    # the following is not really needed for application, but we implement it for debugging
    else:
        for idimnoiterref,edimnoiterref in enumerate(dimnoiterref):
            dimnoiter.append(shp[edimnoiterref])
            lennoiter = lennoiter*shp[edimnoiterref]

    # print 'lennoiter',shp,dimnoiterref,dimiterref,lennoiter
    
    fpos = 0
    # e.g. fpos = (9)+ 20*(10) + 50*50*20*(5)
    for idimpos,edimpos in enumerate(dimpos):
        curadd = edimpos
        #e.g. if edimpos == (5): curadd = 50*50*20*(5)

        # exclude trivial special case of only 1 iteration step
        # --> in that case fpos is just zero.
        if dimiterref != [-1]:
            if ((dimiterref[idimpos] + 1) < len(shp)):
                for i in range(dimiterref[idimpos] + 1,len(shp)) :
                    curadd = curadd * shp[i]
    
        fpos = fpos + curadd
    # print fpos,dimnoiterref,lennoiter
    
    
    # e.g. dimnoiterref = (0,2)
    #      dimnoiterpos = (5,20)
    #      j = based on (0,2) and (5,20)
    
    
    # Initialize (for reading) or prepare (for writing) icecube array
    if mode == 'read':
        icecube = np.zeros((lennoiter,))*np.nan
    elif mode == 'write':
        # print lennoiter
        # print icecube.shape, dimnoiter # should be the same
        icecube = np.reshape(icecube,(lennoiter,))
        # print dataout

    # get the maximum size of continuous data chunks for more efficient IO
    rwchunksize = 1
    idimnoiterref = len(dimnoiterref) - 1
    ishp = len(shp)-1
    while ishp == dimnoiterref[idimnoiterref]:
        # print ishp,idimnoiterref,dimnoiterref[idimnoiterref],shp[ishp],dimnoiter[idimnoiterref]
        rwchunksize = rwchunksize * shp[ishp]
        # # or
        # rwchunksize = rwchunksize * dimnoiter[idimnoiterref]
        idimnoiterref - 1
        ishp = ishp -1


    dimnoiterpos = [0]*len(dimnoiter)
    # print icecube,dimnoiterpos
    j = 0
    while j < lennoiter:
        fposicecube = fpos
        for idimpos,edimpos in enumerate(dimnoiterpos):
            curadd = edimpos
            # e.g. fposicecube = (1)*52
            # e.g. fposicecube = (9)+ 20*(10) + 50*50*20*(5)
            if ((dimnoiterref[idimpos] + 1) < len(shp)):
                for i in range(dimnoiterref[idimpos] + 1,len(shp)) :
                    curadd = curadd * shp[i]
    
            fposicecube = fposicecube + curadd
    
        filestream.seek(4*fposicecube)

        if mode == 'read':
            #print 'test',j,rwchunksize,j+rwchunksize,icecube.shape
            icecube[j:(j+rwchunksize)] = fromfile(filestream,dtype='float32',count=rwchunksize)
        elif mode == 'write':
            filestream.write(np.array(icecube[j:(j+rwchunksize)],dtype='float32'))

        #print 'reading icecube with length / position: ', fposicecube,'/',1,icecube[j]
        # print j, dimnoiterpos,fposicecube,j == fposicecube,icecube[j]
    
        # go to next data strip 
        if dimnoiterpos != []:
            dimnoiterpos[-1] = dimnoiterpos[-1] + rwchunksize 
            for idimidx,edimidx in enumerate(reversed(dimnoiterpos)):
                while dimnoiterpos[idimidx] >= dimnoiter[idimidx]:
                    #print idimidx,dimnoiter[idimidx]
                    if idimidx >= 0:
                        dimnoiterpos[idimidx-1] = dimnoiterpos[idimidx-1] + 1
                        dimnoiterpos[idimidx] -= dimnoiter[idimidx]
        j = j+rwchunksize
    
    icecube.shape = dimnoiter
    if mode == 'read':
        return icecube

def readicecubeps(fstream,shp,dimiterref,dimiterpos,dimnoiterref,mode='read'):
    """ 
    read an icecube and perform an in-memory Post Swap of dimensions (very fast)
    hereby, we acquire the order of the icecube dimensions
    """
    icecube =readicecube(fstream,shp,dimiterref,dimiterpos,None) 
    # print 'shape',icecube,icecube.shape
    if mode=='read':
        # print icecube.shape,zip(*sorted(zip(dimnoiterref,range(len(dimnoiterref))),key=itemgetter(0,1)))
        if dimnoiterref == None:
            return icecube
        else:
            return np.transpose(icecube,zip(*sorted(zip(dimnoiterref,range(len(dimnoiterref))),key=itemgetter(0,1)))[1])
def writeicecubeps(fstream,shp,dimiterref,dimiterpos,dimnoiterref,data):
    """ 
    write an icecube and perform an in-memory Post Swap of dimensions before (very fast)
    hereby, we acquire the order of the icecube dimensions
    """
    # print 'shape',icecube,icecube.shape
    # print icecube.shape,zip(*sorted(zip(dimnoiterref,range(len(dimnoiterref))),key=itemgetter(0,1)))
    # if dimnoiterref == None:
    #     return icecube
    # else:
    #     return np.transpose(icecube,zip(*sorted(zip(dimnoiterref,range(len(dimnoiterref))),key=itemgetter(0,1)))[1])
    trns = zip(*sorted(zip(dimnoiterref,range(len(dimnoiterref))),key=itemgetter(0,1)))[1]
    # build the 'inverse permutation operator'
    inv = range(len(trns))
    for itrns, etrns in enumerate(trns):
        inv[etrns] = itrns

    # print 'hello2',data.shape
    dataout = np.array(np.transpose(data[:],inv),dtype='float32')
    # print 'hello3',dataout.shape
    readicecube(fstream,shp,dimiterref,dimiterpos,icecube=dataout,mode='write') 


# self = io.netcdf.netcdf_file('/home/hendrik/data/belgium_aq/rcm/aq09/stage2/int2lm/laf2009010100_urb_ahf.nc','r')
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
for ivars in range(count):
    vars.append(self._read_var())

ioffset = vars[0][7]
shp = (4320,8640)
dtype = vars[0][4]; mp = 8 ; vardtype = np.float64 # double!

fin = self.fp
fout = open('/home/hendrik/data/global/test.bin','wb')
# fin = open('/home/hendrik/data/belgium_aq/rcm/aq09/stage1/aurorabc/hour16_beleuros.bin','r')
# fout = open('/home/hendrik/data/belgium_aq/rcm/aq09/stage1/aurorabc/hour16_beleuros2.bin','wb')
# def readicecube(filestream,shp,dimiterref,dimpos,dimnoiterref=None):
# testdat = readicecubeps(      fin,       shp,(1,),    (2,),dimnoiterref=(1,0))

# def shake(fin,shp,dimapplyref,fout,dimiterref=None,maxicecubesize=10000):
# shake(      fin,shp,(1,2),dimiterref=None,maxicecubesize=10000)

dimapplyref = (0,)
dimiterref = None
maxicecubesize=100


# def shake(fin,shp,dimapplyref,dimiterref=None,maxicecubesize=10000):
for tt in range(1): 
    """
    purpose 
    -------

    swap specified dimensions to the back efficiently in a specified order

    input parameters
    ----------------

    fin: binary file input stream
    fout: binary file output stream
    shp: shape of the data stream
    dimapplyref: dimensions over which the function is applied
    dimiterref (optional): reference to dimensions that are swapped to the front. The order of those indices are
    taken into account.  Of not specified, it is guessed from the residual dimensions (defined in shp) that are not in dimnoiterref
    """

    lenapply = 1
    dimapply = []

    # we want to read the data in chunks (icecubes) as big as possible. In the first place, the data chunks contain of course the dimensions on which the functions are applied. Afterwards, the chunk dimensions is extended (in the outer(!) direction) to make the icecubes bigger.
    # dimnoiterref: reference to dimensions that are swapped to the back. In any case, this needs to include all dimapplyrefs. Data in these dimensions are read in icecubes. The order of those indices are taken into account
    dimnoiterref = []
    dimnoiter = []
    lennoiter = 1
    for edimapplyref in dimapplyref:
        # dimapplyref.append(edimapplyref)
        dimapply.append(shp[edimapplyref])
        lenapply = lenapply*shp[edimapplyref]

        dimnoiterref.append(edimapplyref)
        dimnoiter.append(shp[edimapplyref])
        lennoiter = lennoiter*shp[edimapplyref]

    if lenapply > maxicecubesize:
        print 'Warning, the function data input length of',lenapply,' (dimensions: ',dimapply,') exceeds the maximum icecubesize of '+str(maxicecubesize)+'.' 
    else:
        for idim,edim in reversed(list(enumerate(shp))):
            if (edim not in dimnoiter):
                if ((lennoiter*edim) < maxicecubesize):
                    dimnoiterref.insert(0,idim)
                    dimnoiter.insert(0,edim)
                    lennoiter = lennoiter*edim


    dimiter = []
    leniter = long(1)

    # guess from residual dimensions that are not in dimnoiterref
    if dimiterref == None:
        dimiterref = []
        for ishp,eshp in enumerate(shp):
            if ishp not in dimnoiterref:
                dimiterref.append(ishp)
    for edimiterref in dimiterref:
        dimiter.append(shp[edimiterref])
        leniter = leniter*dimiter[-1]

    # the trivial case of only one iteration
    if dimiter == []:
        dimiter = [1]
        dimiterpos = [0]
        dimiterref = [-1]
    else:
        dimiterpos = [0]*len(dimiterref)

    shpout = []
    if dimiterref != [-1]:
        for edimiterref in dimiterref:
            shpout.append(shp[edimiterref])

    for edimnoiterref in dimnoiterref:
        shpout.append(shp[edimnoiterref])
    # # or
    # for ishp,eshp in enumerate(shp):
    #     if ishp not in dimiterref: 
    #         shpout.append(shp[i])

    for j in range(leniter):
        print j,'/',leniter

        # actually, this is just the end of the file output already written
        fpos = 0
        # e.g. fpos = (9)+ 20*(10) + 50*50*20*(5mp)
        for idimpos,edimpos in enumerate(dimiterpos):
            curadd = edimpos
            #e.g. if edimpos == (5): curadd = 50*50*20*(5)
            if ((idimpos + 1) < len(shpout)):
                for i in range(idimpos + 1,len(shpout)) :
                    curadd = curadd * shpout[i]
        
            fpos = fpos + curadd

        # read data from file
        fin.seek(ioffset + mp*fpos)
        
        # reading icecube, rearranged in the order of dimensions specified by dimnoiterref
        dataicecube = np.array(readicecubeps(fin,shp,dimiterref,dimiterpos,dimnoiterref),dtype='float32').ravel()
        dataicecubeout = np.zeros(dataicecube.shape,dtype='float32')

        # crush the ice

        # dimnoiterref = (6 ,7 ,8 ,4 ,5)
        # dimiter      = (30,20,15,20,15)
        # dimapplyref  =       (8 ,4 ,5)


        lenapply = long(1)
        dimapply = []
        for idimapplyref in range(len(dimapplyref)):
            dimapply.append(shp[dimapplyref[idimapplyref]])
            lenapply = lenapply*dimapply[-1]

        # # guess from residual dimensions that are not in dimnoiterref
        # if dimiterref == None:
        #     dimiterref = []
        #     for ishp,eshp in enumerate(shp):
        #         if ishp not in dimnoiterref:
        #             dimiterref.append(ishp)

        # we know that the function apply dimensions are at the inner data
        dimnoapply = []
        lennoapply = long(1)
        for idimnoiterref in range(len(dimnoiterref)-len(dimapplyref)):
            dimnoapply.append(shp[dimnoiterref[idimnoiterref]])
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
                curaddout = edimpos
                #e.g. if edimpos == (5): curadd = 50*50*20*(5)
                if ((idimpos + 1) < len(dimnoiterref)):
                    for i in range(idimpos + 1,len(dimnoiterref)) :
                        curadd    = curadd    * dimnoiterref[i]
                        # curaddout = curaddout * dimnoiteroutref[i]
            
                apos    = apos    + curadd

            hunk = dataicecube[apos:(apos+lenapply)]
            hunk.shape = dimapply

            hunkout = (np.zeros(hunk.shape) + 1)*np.mean(hunk)
            dataicecubeout[apos:(apos+lenapply)] = hunkout[:].ravel()

            # go to next data slice  
            dimnoapplypos[-1] = dimnoapplypos[-1] + 1
            for idimidx,edimidx in enumerate(reversed(dimnoapplypos)):
                # # alternative (makes 'dimiter' redundant)
                # if dimiterpos[idimidx] == shp[dimiterref[idimidx]]:
                if dimnoapplypos[idimidx] == dimapply[idimidx]:
                    if idimidx > 0:
                        dimnoapplypos[idimidx-1] = dimnoapplypos[idimidx-1] + 1
                        dimnoapplypos[idimidx] = 0
        
        # print "hello",dataicecubeout.shape, dimnoiter
        dataicecubeout.shape = dimnoiter
        writeicecubeps(fout,shp,dimiterref,dimiterpos,dimnoiterref,dataicecubeout)

        print dimiterpos
        # go to next data slice  
        dimiterpos[-1] = dimiterpos[-1] + 1
        for idimidx,edimidx in enumerate(reversed(dimiterpos)):
            # # alternative (makes 'dimiter' redundant)
            # if dimiterpos[idimidx] == shp[dimiterref[idimidx]]:
            if dimiterpos[idimidx] == dimiter[idimidx]:
                if idimidx > 0:
                    dimiterpos[idimidx-1] = dimiterpos[idimidx-1] + 1
                    dimiterpos[idimidx] = 0

    #print leniter
# 
# def swapindcs(fin,shp,dimnoiterref,fout,dimiterref=None):
#     """
#     purpose 
#     -------
# 
#     swap specified dimensions to the back efficiently in a specified order
# 
#     input parameters
#     ----------------
# 
#     fin: binary file input stream
#     fout: binary file output stream
#     shp: shape of the filestream
#     dimnoiterref: reference to dimensions that are swapped to the back. Data in these dimensions are treated as icecubes. The order of those indices are taken into account
#     dimiterref (optional): reference to dimensions that are swapped to the front. The order of those indices are
#     taken into account.  Of not specified, it is guessed from the residual dimensions (defined in shp) that are not in dimnoiterref
#     """
# 
#     dimiter = []
#     leniter = long(1)
# 
#     # guess from residual dimensions that are not in dimnoiterref
#     if dimiterref == None:
#         dimiterref = []
#         for ishp,eshp in enumerate(shp):
#             if ishp not in dimnoiterref:
#                 dimiterref.append(ishp)
#     for edimiterref in dimiterref:
#         dimiter.append(shp[edimiterref])
#         leniter = leniter*dimiter[-1]
# 
#     dimiterpos = [0]*len(dimiter)
# 
# 
#     shpout = []
#     for edimiterref in dimiterref:
#         shpout.append(shp[edimiterref])
# 
#     for edimnoiterref in dimnoiterref:
#         shpout.append(shp[edimnoiterref])
#     # # or
#     # for ishp,eshp in enumerate(shp):
#     #     if ishp not in dimiterref: 
#     #         shpout.append(shp[i])
# 
#     for j in range(leniter):
# 
#         # actually, this is just the end of the file output already written
#         fposout = 0
#         # e.g. fposout = (9)+ 20*(10) + 50*50*20*(5)
#         for idimpos,edimpos in enumerate(dimiterpos):
#             curadd = edimpos
#             #e.g. if edimpos == (5): curadd = 50*50*20*(5)
#             if ((idimpos + 1) < len(shpout)):
#                 for i in range(idimpos + 1,len(shpout)) :
#                     curadd = curadd * shpout[i]
#         
#             fposout = fposout + curadd
# 
#         # drop data to file in reordered way
#         fout.seek(4*fposout)
#         np.array(readicecubeps(fin,shp,dimiterref,dimiterpos,dimnoiterref),dtype='float32').tofile(fout)
#         # go to next data slice  
#         dimiterpos[-1] = dimiterpos[-1] + 1
#         for idimidx,edimidx in enumerate(reversed(dimiterpos)):
#             # # alternative (makes 'dimiter' redundant)
#             # if dimiterpos[idimidx] == shp[dimiterref[idimidx]]:
#             if dimiterpos[idimidx] == dimiter[idimidx]:
#                 if idimidx > 0:
#                     dimiterpos[idimidx-1] = dimiterpos[idimidx-1] + 1
#                     dimiterpos[idimidx] = 0
#     print leniter
# 
# 
# def outerloop(fin,shp,dimiterref):
#     """
#     loop over the dimensions over which we want to iterate and that are within the icecubes
#     filestream: binary file refence
#     shp: shape of the filestream
#     dimiterref: reference to dimensions over which no slice is performed
#     """
# 
#     dimiter = []
#     leniter = long(1)
#     for edimiterref in dimiterref:
#         dimiter.append(shp[edimiterref])
#         leniter = leniter*dimiter[-1]
# 
#     dimiterpos = [0]*len(dimiter)
# 
#     for j in range(leniter):
#         print readicecube(fin,shp,dimiterref,dimiterpos)
# 
#         # go to next data slice  
#         dimiterpos[-1] = dimiterpos[-1] + 1
#         for idimidx,edimidx in enumerate(reversed(dimiterpos)):
#             # # alternative (makes 'dimiter' redundant)
#             # if dimiterpos[idimidx] == shp[dimiterref[idimidx]]:
#             if dimiterpos[idimidx] == dimiter[idimidx]:
#                 if idimidx > 0:
#                     dimiterpos[idimidx-1] = dimiterpos[idimidx-1] + 1
#                     dimiterpos[idimidx] = 0

fout.close()
fin.close()
fread = open('/home/hendrik/data/global/test.bin','r')


ipol = 0

nx = shp[1]
ny = shp[0]
nz = 0
iz = 0

fig = pl.figure()
fread.seek((ipol*nz + iz)*ny*nx*4,0)
field = fromfile(fread,dtype='float32',count=int(ny*nx/2))
field.shape = (int(ny/2),nx)
pl.imshow(field)
pl.show()

# pl.imshow(testdat)
# fig.show()
# fread.close()
fread.close()
