import pickle
import pylab as pl
from operator import itemgetter
import scipy.io as io
import numpy as np
import sys 

# next:
# implement function passing
# implement multiple variables
# implement multiple variable arrays
# implement wildcard at the end
# implement warning messages:
# - the rwchunksize

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
    # purpose: get array-dtype from netcdf-typecode
    if typecode == 'f':
        return np.dtype('float32')
    elif typecode == 'd':
        return np.dtype('float64')
    elif typecode == 'i':
        return dtype('int32')
    elif typecode == 'l':
        return dtype('int64')

# print rwicecube(fin,(1,35,52),(5,),(3,))
def rwicecube(filestream,shp,dimiterref,dimiter,dimpos,dimnoiterref,dimnoiter,icecube,vtype,vsize,voffset,rwchsize,mode):
    """
    read or write data icecube from binary data and put it in an array
    filestream: binary file reference
    shp: shape of the filestream
    dimiterref: reference to dimensions over which no slice is performed
    pos: current index position of the non-sliced dimensions
    """
    
    # e.g. shp = (200,100,50,50,20)
    #      dimiterref = (1,3,4)
    #      dimpos = (5,10,9)
    
    # extend so that structured arrays are read at once
    
    # dimiter = []
    # dimnoiter = []
    lennoiter = long(1)
    # for i in range(len(shp)):
    #     if i in dimiterref:
    #         dimiter.append(shp[i])
    # if dimnoiterref == None:
    #     dimnoiterref = []
    #     for i in range(len(shp)):
    #         if i not in dimiterref:
    #             dimnoiterref.append(i)
    #             dimnoiter.append(shp[i])
    #             lennoiter = lennoiter*shp[i]
    # # the following is not really needed for application, but we implement it for debugging
    # else:
    for idimnoiterref,edimnoiterref in enumerate(dimnoiterref):
        # dimnoiter.append(shp[edimnoiterref])
        lennoiter = lennoiter*dimnoiter[idimnoiterref]

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
    # print 'fpos',fpos,dimiterref,dimiter,leniter

    
    # e.g. dimnoiterref = (0,2)
    #      dimnoiterpos = (5,20)
    #      j = based on (0,2) and (5,20)
    
    
    # print 'lennoiter:', lennoiter
    # Initialize (for reading) or prepare (for writing) icecube array
    if mode == 'read':
        icecube = np.zeros((lennoiter,),dtype=vtype)*np.nan

    elif mode == 'write':
        # print lennoiter
        # print icecube.shape, dimnoiter # should be the same
        icecube = np.reshape(icecube,(lennoiter,))
        # print dataout

    # get the maximum size of continuous data chunks for more efficient IO

    # # zou goed moeten zijn.
    # found = False
    # idimnoiterref = 0
    # while ((found == False) & (idimnoiterref < len(dimnoiterref))):
    #     cont = True
    #     for ishp in range(len(shp) - (len(dimnoiterref) - idimnoiterref),len(shp)):
    #         if ishp in dimnoiterref[idimnoiterref:]:
    #             cont == True
    #         else:
    #             cont == False
    #         if cont == True: found = idimnoiterref
    #     idimnoiterref = idimnoiterref+1

    # print 'found',found,dimnoiterref[found]
    # if found != False:
    #     for ishp in range(dimnoiterref[found],len(shp)): 
    #         rwchunksize = rwchunksize * shp[ishp]
    #         rwchunksizeout = rwchunksizeout * shpout[ishp]


    # print 'rwchunksize',rwchunksize
    # while dimnoiterref[idimnoiterref] in range(ishp,len(ishp)):
    #     # print ishp,idimnoiterref,dimnoiterref[idimnoiterref],shp[ishp],dimnoiter[idimnoiterref]
    #     rwchunksize = rwchunksize * shp[ishp]
    #     # # or
    #     # rwchunksize = rwchunksize * dimnoiter[idimnoiterref]
    #     idimnoiterref = idimnoiterref - 1
    #     ishp = ishp -1

    # # get the maximum size of continuous data chunks for more efficient IO
    # rwchunksize = 1
    # idimnoiterref = len(dimnoiterref) - 1
    # ishp = len(shp)-1
    # while dimnoiterref[idimnoiterref] in range(ishp,len(ishp)):
    #     # print ishp,idimnoiterref,dimnoiterref[idimnoiterref],shp[ishp],dimnoiter[idimnoiterref]
    #     rwchunksize = rwchunksize * shp[ishp]
    #     # # or
    #     # rwchunksize = rwchunksize * dimnoiter[idimnoiterref]
    #   #  idimnoiterref = idimnoiterref - 1
    #     ishp = ishp -1
    # print 'rwchunksize',rwchunksize

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
    
        filestream.seek(voffset+vsize*fposicecube)
        
        if mode == 'read':
            # rwchsize=rwchunksize
            #print 'test',j,rwchunksize,j+rwchunksize,icecube.shape,voffset,vsize,fposicecube,voffset+vsize*fposicecube
            icecube[j:(j+rwchsize)] = np.fromfile(filestream,dtype=vtype,count=rwchsize)
            # print '_rwchunksize',rwchunksize,icecube[j:(j+rwchunksize)].shape,rwchunksize*vsize
            # icecube[j:(j+1)] = np.fromstring(filestream.read(vsize),dtype=vtype)
        elif mode == 'write':
            # rwchsize=rwchunksizeout
            filestream.seek(vsize*fposicecube)
            # print vtype
            filestream.write(icecube[j:(j+rwchsize)])

        #print 'reading icecube with length / position: ', fposicecube,'/',1,icecube[j]
        # print j, dimnoiterpos,fposicecube,j == fposicecube,icecube[j]
    
        # go to next data strip 
        if dimnoiterpos != []:
            # rwchsize: allow reading of chunks for the inner dimensions
            dimnoiterpos[-1] = dimnoiterpos[-1] + rwchsize
            for idimidx,edimidx in enumerate(reversed(dimnoiterpos)):
                if idimidx > 0:
                    while dimnoiterpos[idimidx] >= dimnoiter[idimidx]:
                    #print idimidx,dimnoiter[idimidx]
                        dimnoiterpos[idimidx-1] = dimnoiterpos[idimidx-1] + 1
                        dimnoiterpos[idimidx] -= dimnoiter[idimidx]
        j = j+rwchsize
    
    icecube.shape = dimnoiter
    if mode == 'read':
        return icecube

def readicecubeps(fstream,shp,dimiterref,dimiter,dimiterpos,dimnoiterref,dimnoiter,vtype,vsize,voffset,rwchsize):
    """ 
    read an icecube by sorting the indices (highest at the back).
    perform an in-memory Post Swap of dimensions (very fast) to compensate for the sorting.
    we allow reading in chunks according to the inner dimensions. They will be mostly there because we allow an max-icecubesize
    """
    # print 'trns:',zip(*sorted(zip(dimnoiterref,range(len(dimnoiterref))),key=itemgetter(0,1)))[1]

    icecube =rwicecube(fstream,shp,dimiterref,dimiter,dimiterpos,sorted(dimnoiterref),sorted(dimnoiter),None,vtype,vsize,voffset,rwchsize,'read') 
    # print 'shape',icecube.shape
    # print 'shape tr',np.transpose(icecube,zip(*sorted(zip(dimnoiterref,range(len(dimnoiterref))),key=itemgetter(0,1)))[1]).shape
    print icecube.shape,zip(*sorted(zip(dimnoiterref,range(len(dimnoiterref))),key=itemgetter(0,1)))[1]

    trns = zip(*sorted(zip(dimnoiterref,range(len(dimnoiterref))),key=itemgetter(0,1)))[1]
    # print 'test write',data.shape,trns

    # build the 'inverse permutation' operator for tranposition before writeout
    inv = range(len(trns))
    for itrns, etrns in enumerate(trns):
        inv[etrns] = itrns
    return np.transpose(icecube,inv)

def writeicecubeps(fstream,shp,dimiterref,dimiter,dimiterpos,dimnoiterref,dimnoiter,data,vtype,vsize,voffset,rwchsize):
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
    # print 'test write',data.shape,trns

    # build the 'inverse permutation' operator for tranposition before writeout

    # print 'test trans',np.transpose(data,inv).shape
    # print 'hello2',data.shape
    # print 'hello3',dataout.shape
    rwicecube(fstream,shp,dimiterref,dimiter,dimiterpos,sorted(dimnoiterref),sorted(dimnoiter),np.transpose(data,trns),vtype,vsize,voffset,rwchsize,'write') 


self = io.netcdf.netcdf_file('/home/hendrik/data/belgium_aq/rcm/aq09/stage2/int2lm/laf2009010100_urb_ahf.nc','r')
# self = io.netcdf.netcdf_file('/home/hendrik/data/global/AHF_2005_2.5min.nc','r')
self.fp.seek(0)
magic = self.fp.read(3)
self.__dict__['version_byte'] = np.fromstring(self.fp.read(1), '>b')[0]

# Read file headers and set data.
# stolen from scipy: /usr/lib/python2.7/dist-packages/scipy/io/netcdf.py
self._read_numrecs()
self._read_dim_array()
self._read_gatt_array()
header = self.fp.read(4)
count = self._unpack_int()

vars = []
for ic in range(count):
    vars.append(list(self._read_var()))

var = 'T'
ivar =  np.where(np.array(vars) == var)[0][0]

fin = self.fp; 
shp = self.variables[var].shape; vtype = vars[ivar][6]; vsize = self.variables[var].itemsize(); voffset = long(vars[ivar][7])

# shp = self.variables[var].shape; vtype = 'float32'; vsize = 4; voffset = vars[ivar][7]


# fin = open('/home/hendrik/data/belgium_aq/rcm/aq09/stage1/aurorabc/hour16_beleuros.bin','r')
# shp = (4,36,52);vtype=np.float32; vsize=4; voffset=0


fout = open('/home/hendrik/data/global/test.bin','wb')
# fout = open('/home/hendrik/data/belgium_aq/rcm/aq09/stage1/aurorabc/hour16_beleuros2.bin','wb')
# def readicecube(filestream,shp,dimiterref,dimpos,dimnoiterref=None):
# testdat = readicecubeps(      fin,       shp,(1,),    (2,),dimnoiterref=(1,0))

# def shake(fin,shp,dimapplyref,fout,vtype,vsize,voffset,dimiterref=None,maxicecubesize=10000):
# shake(      fin,shp,(1,2),offset,vtype,vsize,voffset,dimiterref=None,maxicecubesize=10000)

dimapplyref = (0,1,)
dimiterref = None
maxicecubesize=100000000
func = lambda x: [[np.mean(x)]]# *(1.+np.zeros(x.shape))

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

    # the total length of data passed to function 
    lenapply = 1
    # the dimension of data passed to function 
    dimapply = []

    # we want to read the data in chunks (icecubes) as big as possible. In the first place, the data chunks contain of course the dimensions on which the functions are applied. Afterwards, the chunk dimensions is extended (in the outer(!) direction) to make the icecubes bigger.
    # dimnoiterref: reference to dimensions that are swapped to the back. In any case, this needs to include all dimapplyrefs. Data in these dimensions are read in icecubes. The order of those indices are taken into account. We also add in front those dimensions that can be read at once (still needs to be tested!). 
    dimnoiterref = []

    # the total length of the numpy array as IO memory buffer ('icecubes'). The programm will try to read this in chunks (cfr. rwchunksize- as large as possible. An in-memory transposition may be applied after read or before writing.
    lennoiter = 1
    # the dimension of the IO buffer array
    dimnoiter = []
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
        idim = len(shp)-1
        edim = shp[idim]
        while ((idim >= 0) & ((lennoiter*edim) < maxicecubesize)):
            if (idim not in dimnoiterref):
                dimnoiterref.insert(0,idim)
                dimnoiter.insert(0,edim)
                lennoiter = lennoiter*edim
                # print 'yeeps',idim,edim,dimnoiterref,dimnoiter,lennoiter, maxicecubesize
            idim = idim - 1 
            edim = shp[idim]
        print 'Icecubesize is: ',lennoiter,dimnoiter,dimnoiterref






    lenapply = long(1)
    dimapply = []
    for idimapplyref in range(len(dimapplyref)):
        dimapply.append(shp[dimapplyref[idimapplyref]])
        lenapply = lenapply*dimapply[-1]

    dimapplyout = np.array(func(np.zeros(dimapply))).shape
    # dimnoiterrefout = list(dimnoiterref)
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

    # guess from residual dimensions that are not in dimnoiterref
    if dimiterref == None:
        dimiterref = []
        for ishp,eshp in enumerate(shp):
            if ishp not in dimnoiterref:
                dimiterref.append(ishp)
    for edimiterref in dimiterref:
        dimiter.append(shp[edimiterref])
        leniter = leniter*dimiter[-1]

    #print dimiter
    # the trivial case of only one iteration
    if dimiter == []:
        dimiter = [1]
        dimiterpos = [0]
        dimiterref = [-1]
    else:
        dimiterpos = [0]*len(dimiterref)

    shpout = [None]*len(shp)
    if dimiterref != [-1]:
        for idimiterref,edimiterref in enumerate(dimiterref):
            shpout[edimiterref] = dimiter[idimiterref]

    for idimnoiterref,edimnoiterref in enumerate(dimnoiterref):
        # print 'hello',idimnoiterref,edimnoiterref,dimnoiterout[idimnoiterref]
        shpout[edimnoiterref] = dimnoiterout[idimnoiterref]

    rwchunksize = 1
    rwchunksizeout = 1

    ishp = len(shp)-1
    while ishp in dimnoiterref:
        rwchunksize = rwchunksize*shp[ishp]
        rwchunksizeout = rwchunksizeout*shpout[ishp]
        ishp = ishp-1

    print ' rwchunksize   ',rwchunksize
    print ' rwchunksizeout',rwchunksizeout

    # # or
    # for ishp,eshp in enumerate(shp):
    #     if ishp not in dimiterref: 
    #         shpout.append(shp[i])

    for j in range(leniter):
        if j>0: sys.stdout.write ('\b'*(len(str(j-1)+'/'+str(leniter))+1))
        print str(j)+'/'+str(leniter),
        # print j,leniter,dimnoiterref,dimapplyref
        # actually, this is just the end of the file output already written

        # # read data from file
        # fin.seek(voffset + vsize*fpos)
        
        # reading icecube, rearranged in the order of dimensions specified by dimnoiterref
        dataicecube = np.array(readicecubeps(fin,shp,dimiterref,dimiter,dimiterpos,dimnoiterref,dimnoiter,vtype,vsize,voffset,rwchunksize),dtype=vtype).ravel()

        dataicecubeout = np.zeros((lennoiterout,),dtype=vtype)

        # crush the ice

        # dimnoiterref = (6 ,7 ,8 ,4 ,5)
        # dimiter      = (30,20,15,20,15)
        # dimapplyref  =       (8 ,4 ,5)



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
            dimnoapply.append(dimnoiter[idimnoiterref])
            lennoapply = lennoapply*dimnoapply[-1]

        if dimnoapply == []:
            dimnoapply = [1]

        dimnoapplypos = [0]*len(dimnoapply)
        

        for k in range(lennoapply):
            # print j,k
            # actually, this is just the end of the file output already written
            apos    = 0
            # e.g. apos = (9)+ 20*(10) + 50*50*20*(5)
            for idimpos,edimpos in enumerate(dimnoapplypos):
                curadd    = edimpos
                #e.g. if edimpos == (5): curadd = 50*50*20*(5)
                if ((idimpos + 1) < len(dimnoiterref)):
                    for i in range(idimpos + 1,len(dimnoiterref)) :
                        curadd    = curadd    * dimnoiter[i]
                        # curaddout = curaddout * dimnoiteroutref[i]
            
                apos    = apos    + curadd


            aposout    = 0
            # e.g. aposout = (9)+ 20*(10) + 50*50*20*(5)
            for idimpos,edimpos in enumerate(dimnoapplypos):
                curadd    = edimpos
                #e.g. if edimpos == (5): curadd = 50*50*20*(5)
                if ((idimpos + 1) < len(dimnoiterref)):
                    for i in range(idimpos + 1,len(dimnoiterref)) :
                        curadd    = curadd    * dimnoiterout[i]
                        # curaddout = curaddout * dimnoiteroutref[i]
            
                aposout    = aposout    + curadd

            hunk = dataicecube[apos:(apos+lenapply)]
            hunk.shape = dimapply

            # apply the function
            hunkout = np.array(func(hunk)) #np.array((np.zeros(hunk.shape) + 1)*np.mean(hunk),dtype=vtype)
            # print 'hunk   ',apos, hunk.shape, lenapply
            # print 'hunkout',aposout, hunkout.shape, lenapplyout
            dataicecubeout[aposout:(aposout+lenapplyout)] = np.array(hunkout[:].ravel(),dtype=vtype)
            # print aposout, aposout+lenapplyout,lenapplyout,dataicecubeout

            # go to next data slice  
            dimnoapplypos[-1] = dimnoapplypos[-1] + 1
            for idimidx,edimidx in enumerate(reversed(dimnoapplypos)):
                # # alternative (makes 'dimiter' redundant)
                # if dimiterpos[idimidx] == shp[dimiterref[idimidx]]:
                if idimidx > 0:
                    if dimnoapplypos[idimidx] == dimnoapply[idimidx]:
                       dimnoapplypos[idimidx-1] = dimnoapplypos[idimidx-1] + 1
                       dimnoapplypos[idimidx] = 0
        
        # print "hello",dataicecubeout.shape, dimnoiter
        dataicecubeout.shape = dimnoiterout
        # print dataicecubeout
        writeicecubeps(fout,shpout,dimiterref,dimiter,dimiterpos,dimnoiterref,dimnoiterout,dataicecubeout,vtype,vsize,voffset,rwchunksizeout)

        #print dimiterpos
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

nx = shpout[3]
ny = shpout[2]
nz = 1# shp[0]
iz = 0

fig = pl.figure()
fread.seek((ipol*nz + iz)*ny*nx*vsize,0)
field = np.fromfile(fread,dtype=vtype,count=ny*nx)
field.shape = (ny,nx)
pl.imshow(field)
fig.show()

# pl.imshow(testdat)
# fig.show()
# fread.close()
fread.close()
