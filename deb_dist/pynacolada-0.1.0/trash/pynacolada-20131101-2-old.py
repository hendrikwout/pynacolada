import pickle
import pylab as pl
from operator import itemgetter
import scipy.io as io
import numpy as np
import sys 
from operator import mul



class SomeError(Exception):
     def __init__(self, value):
         self.value = value
     def __str__(self):
         return repr(self.value)

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

    refnoitersort,trns,dimnoitersort = zip(*sorted(zip(refnoiter,range(len(refnoiter)),dimnoiter),key=itemgetter(0,1)))
    icecube =rwicecube(fstream,shp,refiter,dimiter,dimiterpos,refnoitersort,dimnoitersort,None,vtype,vsize,voffset,rwchsize,'read') 


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
    refnoitersort,trns,dimnoitersort = zip(*sorted(zip(refnoiter,range(len(refnoiter)),dimnoiter),key=itemgetter(0,1)))
    rwicecube(fstream,shp,refiter,dimiter,dimiterpos,refnoitersort,dimnoitersort,np.transpose(data,trns),vtype,vsize,voffset,rwchsize,'write') 


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

func = lambda x,y,z,u: np.array([[[np.mean(x)]],[[np.mean(x)]]]) # *(1.+np.zeros(x.shape))

#sizeout(shp,func,dimapplyref)
 
# extract shpout from shp[:],dimref[:],dimapplyref[:],func

adimsin = ((365,40,100,300),\
           (200,40),\
           (365,40,100),\
           (1,200,100),\
           )
avtypeoutspec = [None]
adnamsin = (('time','z','lat','lon'),\
         ('lon','z'),\
         ('time','z','lat'),\
         ('time','lon','lat'))
dnamsel = ('lon','time','t')
maxicecubesize=100000000

# construction of the output dimensions
dimsout = [] # maximum length of an output dimension
dnamout = []
idimsout = 0

for idnamsin,ednamsin in enumerate(adnamsin):
    for idnam,ednam in reversed(list(enumerate(ednamsin))):
        if ednam not in dnamout:
            dnamout.insert(0,ednam)
            print ednam
            if ednam not in dnamsel:
                dimsout.insert(0,adimsin[idnamsin][idnam])
                print ednam
            else:
                # In this case, wait for assigning the output dimensions. This actually depends on the specified function
                dimsout.insert(0,None)
        else:
            if ((adimsin[idnamsin][idnam] != 1) & (dimsout[dnamout.index(ednam)] != 1) & \

                # we allow non-equal dimension lengths, as long as the dimension is covered/captured by the function
                # maybe still allow non-equal dimension length not covered by the function????
                (dimsout[dnamout.index(ednam)] != None) & \
                (adimsin[idnamsin][idnam] != dimsout[dnamout.index(ednam)])):
                raise SomeError("The corresponding output dnamensions (index: "+str(dnamout.index(ednam))+") of the input variable "+str(idnamsin)+ " "+ str(idnam)+ " "+" have a different length and not equal to 1.")
            else:
                if (dimsout[dnamout.index(ednam)] != None):
                    dimsout[dnamout.index(ednam)] = max(dimsout[dnamout.index(ednam)],adimsin[idnamsin][idnam])
print 'Output dimensions: ', zip(dnamout,dimsout)
idnam = 0

# ad the missing dimensions selected for the function
for idnamsel,ednamsel in enumerate(dnamsel):
    if ednamsel not in dnamout:
        dnamout.insert(idnam,ednamsel)
        dimsout.insert(idnam,None) # to be defined from the function
        idnam = idnam+1 # moet dit ook hier niet boven geimplementeerd worden?
    else:
        idnam = dnamout.index(ednam)+1

# copy adnams
adnams = list([])
for idnamsin,ednamsin in enumerate(adnamsin):
    adnams.append(list(ednamsin))

for idnams,ednams in enumerate(adnams):
    idnam = 0
    for idnamout,ednamout in enumerate(dnamout):
        if ednamout not in ednams:
            ednams.insert(idnam,ednamout)
            idnam = idnam + 1
        else:
            idnam = ednams.index(ednamout) + 1

adims = []
arefs = []
for idims,edims in enumerate(adimsin):
    arefs.append(list([]))
    adims.append(list([]))
    # dnamout.index()

for idims,edims in enumerate(adimsin):
    for idim,edim in enumerate(dimsout):
        arefs[idims].append(dnamout.index(adnams[idims][idim]))
        if dnamout[arefs[idims][-1]] in adnamsin[idims]:
            adims[idims].append(adimsin[idims][adnamsin[idims].index(dnamout[arefs[idims][-1]])])
        else:
            adims[idims].append(1)

adims = np.array(adims,dtype=np.int32)
arefs = np.array(arefs,dtype=np.int32)

# adimssw: the input dimensions (like adims), but ordered according to the output dimensions
adimssw = np.zeros_like(arefs)
arefssw = np.zeros_like(arefs)
for irefs,erefs in enumerate(arefs):
    for iref,eref in enumerate(arefs[irefs]):
        adimssw[irefs,arefs[irefs][iref]] = adims[irefs,iref]
        arefssw[irefs,arefs[irefs][iref]] = iref

refapplyout = []
for idnamsel,ednamsel in enumerate(dnamsel):
    refapplyout.append(dnamout.index(ednamsel))


# adimapplyin: the input dimensions of the function based on the refapplyout

# arefapply = [list([])]*len(arefs)
# adimapplyin = np.array([list([None]*len(refapplyout))]*len(arefs))
adimapplyin = np.zeros((len(arefs),len(refapplyout)),dtype='int32')

for irefapplyout,erefapplyout in enumerate(refapplyout):
    for idims,edims in enumerate(adims):
        adimapplyin[idims,irefapplyout] = adims[idims][np.where(arefs[idims] == erefapplyout)[0][0]]


dummydat = []
avtypeout = []
for idimapply,edimapply in enumerate(adimapplyin):
    dummydat.append(np.zeros(edimapply))

ddout =  func(*dummydat)
if (type(ddout).__name__ != 'list'):
    ddout = list([ddout])

for iddout in range(len(ddout)):
    ddout[iddout] = np.array(ddout[iddout])

    if (len(np.array(ddout[iddout]).shape) != len(adimapplyin[iddout])):
        raise SomeError('The amount of input ('+str(len(adimapplyin[iddout]))+') and output dimensions ('+str(len(ddout[iddout].shape))+') of function  is not the same')

    # determine datatype (dtype) output
    if avtypeoutspec[iddout] != None:
        # adopt the one from the output file
        avtypeout.append(avtypeout[iddout])
    else: 
        # otherwise adopt the one from the function output
        avtypeout.append(ddout[iddout].dtype)

adimsout = []
adimapplyout = []
for iddout,eddout in enumerate(ddout):
    adimsout.append(list(dimsout))
    adimapplyout.append([None]*len(refapplyout))
    print iddout

# arefout = [0,1,2,3,4]
adimapplyout = []
alenapplyout = []
for idimsout,edimsout in enumerate(adimsout):
    adimapplyout.append([])
    for irefapplyout,erefapplyout in enumerate(refapplyout):
        adimsout[idimsout][erefapplyout] = ddout[idimsout].shape[irefapplyout]
        adimapplyout[idimsout].append(ddout[idimsout].shape[irefapplyout]) # adimsout[idimsout][arefs[idims].index(erefapplyout)]
    alenapplyout.append(reduce(mul,adimapplyout[idimsout]))

    # we want to read the data in chunks (icecubes) as big as possible. In the first place, the data chunks contain of course the dimensions on which the functions are applied. Afterwards, the chunk dimensions is extended (in the outer(!) direction) to make the icecubes bigger.
    # refnoiter: reference to dimensions that are swapped to the back. In any case, this needs to include all refapplyouts. Data in these dimensions are read in icecubes. The order of those indices are taken into account. We also add in front those dimensions that can be read at once (still needs to be tested!). 

    # the total length of the numpy array as IO memory buffer ('icecubes'). The programm will try to read this in chunks (cfr. rwchunksize- as large as possible. An in-memory transposition may be applied after read or before writing.
    # the dimension of the IO buffer array

    # the total length of data passed to function 
alenapply = []
alennoiter = []

# important remark!!! we consider that the number of input dimensions of the function is equal to the number of output dimensions!!!!!
# (a)refapply = (a)refapplyout

# adimapply: the dimensions of data to be read in chunks, ordered
adimnoiter = []
for idimsswapply,edimsswapply in enumerate(adimapplyin):
    alenapply.append( reduce(mul,edimsswapply))
    adimnoiter.append([])
alennoiter = np.array(alenapply)

refnoiterout = []
for irefapplyout,erefapplyout in enumerate(refapplyout):
    refnoiterout.append(int(erefapplyout))
for irefapplyout,erefapplyout in enumerate(refapplyout):
    for idimsswapply,edimsswapply in enumerate(adimapplyin):
        adimnoiter[idimsswapply].append(int(adimapplyin[idimsswapply,irefapplyout]))

# for now:
#   adimnoiter = adimapplyin, but will be appended below
#   refnoiterout = refapplyout
# we now will try to read the data in even larger icecubes!


if (max(alennoiter) > maxicecubesize):
    print 'Warning, one of the function data input lengths "',alennoiter,'" (dimensions: ',adimapplyin,') exceeds the maximum icecubesize of '+str(maxicecubesize)+'.' 
else:
    # we try will to read the data in even larger icecubes!
    idim = arefs.shape[1]-1
    # emaxdim = max(adimssw[:,idim])
    while ((idim >= 0) & ((max(alennoiter)*max(adimssw[:,idim])) < maxicecubesize)):
        print idim
        if (idim not in refnoiterout):
            print 'idim',idim
            refnoiterout.insert(0,int(idim))
            for idimsapply,emaxdimsapply in enumerate(adimapplyin):
                adimnoiter[idimsapply].insert(0,adimssw[idimsapply,idim])
                alennoiter[idimsapply] = alennoiter[idimsapply] * adimssw[idimsapply,idim]
            # print 'yeeps',idim,emaxdim,refnoiterout,dimnoiter,lennoiter, maxicecubesize
        idim = idim - 1 
print 'Icecubesizes are: ',alennoiter #,dimnoiter,refnoiterout

for idimsout,edimsout in enumerate(dimsout):
    dimsout[idimsout] = max(np.array(adimsout)[:,idimsout])

dimnoiterout = []     
for irefnoiterout,erefnoiterout in enumerate(refnoiterout):
    dimnoiterout.append(dimsout[erefnoiterout])
    
dimiter = []

# guess from residual dimensions that are not in refnoiterout
refiter = None
if refiter == None:
    refiter = []
    for idimsout,edimsout in enumerate(dimsout):
        if idimsout not in refnoiterout:
            refiter.append(idimsout)

adimiter = []
adimiterpos = []
aleniter = []
for idimssw,edimssw in enumerate(adimssw):
    adimiter.append([])
    adimiterpos.append([])
    aleniter.append(1)
    for erefiter in refiter:
        adimiter[idimssw].append(int(adimssw[idimssw][erefiter]))
        aleniter[idimssw] = aleniter[idimssw]*adimiter[idimssw][-1]
    # the trivial case of only one iteration
    if adimiter[idimssw] == []:
        adimiter[idimssw].append(1)
        adimiterpos[idimssw].append(0)
    else:
        adimiterpos[idimssw].append([0]*len(refiter))

adimiterout = []
# adimiterposout = []
aleniterout = []
for idimsout,edimsout in enumerate(adimsout):
    adimiterout.append([])
    # adimiterposout.append([])
    aleniterout.append(1)
    for erefiter in refiter:
        adimiterout[idimsout].append(int(adimsout[idimsout][erefiter]))
        aleniterout[idimsout] = aleniterout[idimsout]*adimiterout[idimsout][-1]
    # the trivial case of only one iteration
    if adimiterout[idimsout] == []:
        adimiterout[idimsout].append(1)
    #    adimiterposout[idimsout].append(0)
    #else:
    #    adimiterposout[idimsout].append([0]*len(refiter))

for idims,edims in enumerate(adims):
    if refiter == []:
        refiter = [-1]

arefsiter = []
for irefs,erefs in enumerate(arefs):
    arefsiter.append([])
    for iref,eref in enumerate(refiter):
        if eref != -1:
            arefsiter[irefs].append(arefssw[irefs][eref])

arefsnoiter = []
for irefs,erefs in enumerate(arefs):
    arefsnoiter.append([])
    for iref,eref in enumerate(refnoiterout):
        if eref != -1:
            arefsnoiter[irefs].append(arefssw[irefs][eref])

arefsapply = []
for irefs,erefs in enumerate(arefs):
    arefsapply.append([])
    for iref,eref in enumerate(refapplyout):
        if eref != -1:
            arefsapply[irefs].append(arefssw[irefs][eref])

lennoiterout = reduce(mul,dimnoiterout)

# maximum of both input and output dimensions for the iteration
dimitermax =np.zeros(np.array(adimiter).shape[1],dtype=np.int32)
for idimitermax,edimitermax in enumerate(dimitermax):
    for idimiter,edimiter in enumerate(adimiter):
        dimitermax[idimitermax] = max(dimitermax[idimitermax],adimiter[idimiter][idimitermax])

# maximum of both input and output dimensions for the iteration
    for idimiterout,edimiterout in enumerate(adimiterout):
        dimitermax[idimitermax] = max(dimitermax[idimitermax],adimiterout[idimiterout][idimitermax])

lenitermax = reduce(mul,dimitermax)
dimiterpos = [0]*len(dimitermax)

# short overview:
# # arefssw: the references of the output dimensions (adimssw) to the data dimensions
# # arefs: the references of the data dimensions (adims) to the output dimensions
# # arefsiter: refences of the looping dimensions to the data dimensions
# # arefsnoiter: refences of the non-looping dimensions to the data dimensions

# get the maximum size of continuous data chunks for more efficient IO
rwchunksize = [1]*len(arefsnoiter)
for idims in range(len(arefsnoiter)):
    idim = len(adimnoiter[idims])
    while ((idim in arefsnoiter[idims]) & (idim >= 0)):
        # The inner dimensions just have to be referenced so not in correct order.  We know that they will be read in the correct order in the end
        rwchunksize[idims] = rwchunksize[idims]*adims[idims][idim]
        idim = idim - 1

rwchunksizeout = [1]*len(adimsout)
idim = len(dimnoiterout)
while ((idim in refnoiterout) & (idim >= 0)):
    # The inner dimensions just have to be referenced and not in correct order.  We know that they will be read in the correct order in the end
    for idimsout,edimsout in enumerate(adimsout):
        rwchunksizeout[idimsout] = rwchunksizeout[idimsout]*adimsout[idimsout][idim]
    idim = idim - 1

adimnoiterout = []
alennoiterout = []
for idimsout,edimsout in enumerate(adimsout):
    adimnoiterout.append([])
    for iref,eref in enumerate(refnoiterout):
        adimnoiterout[idimsout].append(adimsout[idimsout][eref])
    alennoiterout.append(reduce(mul,adimnoiterout[idimsout]))
# get the dimensions of the buffer over which we iterate
# we know that the function are applied along dimensions that are at the inner data
adimnoapply = []
alennoapply = []
for irefs,erefs in enumerate(arefs):
    adimnoapply.append([])
    alennoapply.append(1)
    for irefnoiterout in range(len(arefsnoiter[irefs])-len(refapplyout)):
        adimnoapply[irefs].append(adimnoiter[irefs][irefnoiterout])
        alennoapply[irefs] =alennoapply[irefs]*adimnoapply[irefs][-1]

    if adimnoapply[irefs] == []:
        adimnoapply[irefs] = [1]

adimnoapplyout = []
alennoapplyout = []
for idimsout in range(len(adimsout)):
    adimnoapplyout.append([])
    alennoapplyout.append(1)
    for irefnoiterout in range(len(refnoiterout)-len(refapplyout)):
        adimnoapplyout[idimsout].append(adimnoiterout[idimsout][irefnoiterout])
        alennoapplyout[idimsout] = alennoapplyout[idimsout]*adimnoapplyout[idimsout][-1]

    if adimnoapplyout[idimsout] == []:
        adimnoapplyout[idimsout] = [1]

dimnoapply = [1]*len(adimnoapply[1])
for idimnoapply in range(len(dimnoapply)):
    for idims,edims in enumerate(adimnoapply):
        dimnoapply[idimnoapply] = max(dimnoapply[idimnoapply],adimnoapply[idims][idimnoapply])
    for idims,edims in enumerate(adimnoapplyout):
        dimnoapply[idimnoapply] = max(dimnoapply[idimnoapply],adimnoapplyout[idims][idimnoapply])
lennoapply = reduce(mul,dimnoapply)

dimnoapplypos = [0]*len(dimnoapply)

print str(0)+'/'+str(lenitermax),
for j in range(lenitermax):
    # reading icecube, rearranged in the order of dimensions specified by refnoiterout
    dataicecube = []
    for ilennoiter,elennoiter in enumerate(alennoiter):
         #dataicecube.append(np.array(readicecubeps(fin,adims[irefs],arefsiter[irefs],adimiter[irefs],adimiterpos[irefs],arefsnoiter[irefs],adimnoiter[irefs],vtype,vsize[irefs],voffset[irefs],rwchunksize[irefs]),dtype=vtype).ravel()
         dataicecube.append(np.zeros((elennoiter,))) 
         #np.array(readicecubeps(fin,adims[0],arefsiter[0],adimiter[0],adimiterpos[0],arefsnoiter[0],adimnoiter[0],vtype,vsize,voffset,rwchunksize),dtype=vtype).ravel()
    # temporary store output in a array-buffer
    
    dataicecubeout = []
    for ilennoiterout,elennoiterout in enumerate(alennoiterout):
        dataicecubeout.append(np.zeros((elennoiterout,),dtype=avtypeout[ilennoiterout]))

    # crush the ice
    # refnoiterout = (6 ,7 ,8 ,4 ,5)
    # dimiter      = (30,20,15,20,15)
    # refapplyout  =       (8 ,4 ,5)

    for k in range(lennoapply):
        # actually, this is just the end of the file output already written
        ahunk = []
        for irefs, erefs in enumerate(arefs):
            pos = 0
            # e.g. pos = (9)+ 20*(10) + 50*50*20*(5)
            for idimpos,edimpos in enumerate(dimnoapplypos):
                curadd    = np.mod(edimpos,adimnoapply[irefs][idimpos])
                #e.g. if edimpos == (5): curadd = 50*50*20*(5)
                if ((idimpos + 1) < len(arefsnoiter[irefs])):
                    for i in range(idimpos + 1,len(arefsnoiter[irefs])) :
                                                # here, we assume that the dimensions of the chunk are already in the order considered by adimsnoiter(out) etc. (cfr. preceeded transposition in readicecubeps)
                        curadd    = curadd    * adimnoiter[irefs][i]
                        # curaddout = curaddout * dimnoiteroutref[i]
                pos    = pos    + curadd
            ahunk.append(dataicecube[irefs][pos:(pos+alenapply[irefs])])
            ahunk[irefs].shape = adimapplyin[irefs]

        # apply the function
        ahunkout = np.array(func(*ahunk)) #np.array((np.zeros(hunk.shape) + 1)*np.mean(hunk),dtype=vtype)
        if (type(ahunkout).__name__ != 'list'): # tbi: nog te bekijken of dit wel de handigste voorwaarde is!
            ahunkout = list([ahunkout])

        for ihunkout in range(len(ahunkout)):
            ahunkout[ihunkout] = np.array(ahunkout[ihunkout])
            # e.g. posout = (9)+ 20*(10) + 50*50*20*(5)
            posout    = 0
            for idimpos,edimpos in enumerate(dimnoapplypos):
                curadd    = np.mod(edimpos,adimnoapplyout[ihunkout][idimpos])
                #e.g. if edimpos == (5): curadd = 50*50*20*(5)
                if ((idimpos + 1) < len(refnoiterout)):
                    for i in range(idimpos + 1,len(refnoiterout)) :
                                                # here, we assume that the idims are in the intended order (cfr. subsequent transposition in writeicecubeps)
                        curadd    = curadd    * dimnoiterout[i]
                        # curaddout = curaddout * dimnoiteroutref[i]
                posout    = posout    + curadd

            dataicecubeout[ihunkout][posout:(posout+alenapplyout[ihunkout])] = np.array(ahunkout[ihunkout].ravel(),dtype=avtypeout[ihunkout])

        # go to next data slice  
        dimnoapplypos[-1] = dimnoapplypos[-1] + 1
        for idimidx,edimidx in enumerate(reversed(dimnoapplypos)):
            # # alternative (makes 'dimiter' redundant)
            # if dimiterpos[idimidx] == shp[refiter[idimidx]]:
            if idimidx > 0:
                if dimnoapplypos[idimidx] == dimnoapply[idimidx]:
                   dimnoapplypos[idimidx-1] = dimnoapplypos[idimidx-1] + 1
                   dimnoapplypos[idimidx] = 0
    
    for idimsout in range(len(dataicecubeout)):
        dataicecubeout[idimsout].shape = dimnoiterout
        #print dataicecubeout[idimsout].shape
    # for idimsout in range(len(adimsout)):
    #     writeicecubeps(fout[idimsout],\
    #                    adimsout[idimsout],\
    #                    arefsnoiter[idimsout],\
    #                    adimiterout[idimsout],\
    #                    dimiterposout[idimsout],\
    #                    arefnoiterout[idimsout],\
    #                    adimnoiterout[idimsout],\
    #                    dataicecubeout[idimsout],\
    #                    vtype[idimsout],\
    #                    vsize[idimsout],\
    #                    voffset[idimsout],\
    #                    rwchunksizeout[idimsout])

    # go to next data slice  
    dimiterpos[-1] = dimiterpos[-1] + 1
    for idimidx,edimidx in enumerate(reversed(dimiterpos)):
        # # alternative (makes 'dimiter' redundant)
        # if dimiterpos[idimidx] == shp[refiter[idimidx]]:
        if dimiterpos[idimidx] == dimitermax[idimidx]:
            if idimidx > 0:
                dimiterpos[idimidx-1] = dimiterpos[idimidx-1] + 1
                dimiterpos[idimidx] = 0
    sys.stdout.write ('\b'*(len(str(j)+'/'+str(lenitermax))+1))
    sys.stdout.write (str(j+1)+'/'+str(lenitermax))


