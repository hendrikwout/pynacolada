import os
import pickle
import pylab as pl
from operator import itemgetter
import netcdf 
import numpy as np
import sys 
from operator import mul
from ncdftools import nccopydimension
from Scientific.IO import NetCDF
from array import array
import struct


def nctypecode(dtype):
    # purose: netcdf-typecode from array-dtype
    if ((dtype == np.dtype('float32')) or (np.dtype == 'float32')):
        return 'f'
    elif ((dtype == np.dtype('float64')) or (np.dtype == 'float64')):
        return 'd'
    elif ((dtype == np.dtype('int32')) or (np.dtype == 'int32')):
        return 'i'
    elif ((dtype == np.dtype('int64')) or (np.dtype == 'int64')):
        return 'l'

class SomeError(Exception):
     def __init__(self, value):
         self.value = value
     def __str__(self):
         return repr(self.value)

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
        curadd = np.mod(edimpos,dimiter[idimpos])
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
            curadd = np.mod(edimpos,dimnoiter[idimpos])
            # e.g. fposicecube = (1)*52
            # e.g. fposicecube = (9)+ 20*(10) + 50*50*20*(5)
            if ((refnoiter[idimpos] + 1) < len(shp)):
                for i in range(refnoiter[idimpos] + 1,len(shp)) :
                    curadd = curadd * shp[i]
    
            fposicecube = fposicecube + curadd
    
        
        if mode == 'read':
            filestream.seek(voffset+vsize*fposicecube)
            temp = np.fromfile(filestream,dtype='='+vtype[1],count=rwchsize)
            temp.byteswap(True)
            icecube[j:(j+rwchsize)] = temp
        elif mode == 'write':
            filestream.seek(voffset+vsize*fposicecube)
            fpointout.seek(voffset+vsize*fposicecube)
            # filestream.seek(voffset+vsize*fposicecube)
            testdata[fposicecube:(fposicecube+rwchsize)] = np.array(icecube[j:(j+rwchsize)],dtype=vtype[1])

            # little = struct.pack('>'+'d'*len(icecube[j:(j+rwchsize)]), *icecube[j:(j+rwchsize)]) 
            # # Seek to offset based on piece index 
            # #print little
            # filestream.write(little)

            # filestream.write(np.array(icecube[j:(j+rwchsize)],dtype=vtype))
            # # np.array(icecube[j:(j+rwchsize)],dtype=vtype[1]).byteswap().tofile(filestream)
            temp = np.array(icecube[j:(j+rwchsize)],dtype='>d')
            filestream.write(temp)
            fpointout.write(temp)
            # # print temp
            # # filestream.write(temp[:])
            # # little = struct.pack('<'+'B'*len(temp), *temp) 
            # # print icecube.byteswap().dtype
            # # print voffset, vsize, fposicecube, vtype, rwchsize, icecube.dtype# ,icecube[j:(j+rwchsize)]
    
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


def writeicecubeps(fstream,shp,refiter,dimiter,dimiterpos,refnoiter,dimnoiter,data,vtype,vsize,voffset,rwchsize):
    """ 
    write an icecube and perform an in-memory Post Swap of dimensions before (very fast)
    hereby, we acquire the order of the icecube dimensions
    """
    refnoitersort,trns,dimnoitersort = zip(*sorted(zip(refnoiter,range(len(refnoiter)),dimnoiter),key=itemgetter(0,1)))
    rwicecube(fstream,shp,refiter,dimiter,dimiterpos,refnoitersort,dimnoitersort,np.transpose(data,trns),vtype,vsize,voffset,rwchsize,'write') 

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



fnin = '/home/hendrik/data/belgium_aq/rcm/aq09/stage1/int2lm/laf2009010100_urb_ahf.nc'
print fnin
# fobjin = open(fnin,'rb')
fin = NetCDF.NetCDFFile(fnin,'r')
fnout = '/home/hendrik/data/belgium_aq/rcm/aq09/stage1/int2lm/laf2009010100_urb_ahf2.nc'
os.system('rm '+fnout)
print fnout
# fobjout = open(fnout,'wb+')
fout = NetCDF.NetCDFFile(fnout,'w')

fnpointout = '/home/hendrik/data/belgium_aq/rcm/aq09/stage1/int2lm/laf2009010100_urb_ahf4.nc'
os.system('rm '+fnpointout)
print fnpointout
# fobjout = open(fnpointout,'wb+')
fpointout = open(fnpointout,'w')

# we kunnen eens proberen om een variabele aan te maken met een vooraf gespecifieerde dimensie!

datin =  [[fin,'QV'],[fin,'rlat']]
datout = [[fout,'QV'],[fout,'TEST']]
# adtypeoutspec = [None,None] # to be obtained automatically from the data output stream (if it already exists)

# selection of function dimension input
func = lambda x, y: (np.array([[[np.mean(x)]],[[np.mean(x)]]],dtype=np.float) , np.array([[[np.mean(x)]],[[np.mean(x)]]],dtype=np.float)) # *(1.+np.zeros(x.shape))
dnamsel = ('rlon','time','t')

# obtain definitions of the variable stream input
vsdin = [] # input variable stream definitions
for idatin,edatin in enumerate(datin):
    # read in scipy.netcdf mode to obtain varariable offsets
    
    # obtain file name from open netcdf!! very nasty!!!
    ncfn =  str(datin[idatin][0])[19:(str(datin[idatin][0]).index("'",19))]

    nctemp = netcdf.netcdf_file(ncfn,'r')
    # nctemp = datin[idatin][0]

    vsdin.append(dict())
    vsdin[idatin]['dnams'] = []
    for idim,edim in enumerate(nctemp.variables[datin[idatin][1]].dimensions):
        vsdin[idatin]['dnams'].append(str(edim))
    vsdin[idatin]['dims'] = list(nctemp.variables[datin[idatin][1]].shape)
    vsdin[idatin]['itemsize'] = nctemp.variables[datin[idatin][1]].itemsize()
    vsdin[idatin]['dtype']  =   nctemp.variables[datin[idatin][1]]._dtype
    vsdin[idatin]['voffset']  = nctemp.variables[datin[idatin][1]]._voffset
    nctemp.close()

# obtain definitions of the variable stream output
vsdout = [] # input variable stream definitions
for idatout,edatout in enumerate(datout):
    vsdout.append(dict())
    if edatout[1] in edatout[0].variables:
        vsdout[idatout]['dnams'] = []
        for idim,edim in enumerate(datout[idatout][0].variables[datout[idatout][1]].dimensions):
            vsdout[idatout]['dnams'].append(str(edim))

        vsdout[idatout]['dims'] = list(datout[idatout][0].variables[datout[idatout][1]].shape)
        vsdout[idatout]['itemsize'] = datout[idatout][0].variables[datout[idatout][1]].itemsize()
        vsdout[idatout]['dtype']=     datout[idatout][0].variables[datout[idatout][1]]._dtype
        vsdout[idatout]['voffset'] =  datout[idatout][0].variables[datout[idatout][1]]._voffset
    else:
        # the variable doesn't exists (we will create it afterwards)
        vsdout[idatout]['dnams'] = None
        vsdout[idatout]['dims'] = None
        vsdout[idatout]['itemsize'] = None
        vsdout[idatout]['dtype'] = None


# collecting the involved dimensions (will be considered as the standard output dimensions)
dnamsstd = [] # standard output dimensions: list of all output dimensions: this is collected from the input dimensions, the output dimensions and the selected/processed dimensions
dimsstd = [] # maximum length of an output dimension
idimsstd = 0

for ivsdin,evsdin in enumerate(vsdin):
    dnaminlast = None
    index = 0
    for idnam,ednam in reversed(list(enumerate(evsdin['dnams']))):
        if ednam not in dnamsstd:
            # In dnamsstd, ednam should be just after the dimensions preceding ednams in dnams  
            # # actually, we also want that, in dnamsstd, ednam should be just before the dimensions succeeding ednams in dnams. Sometimes, this is not possible at the same time. But it will be the case if that is possible when applying one of the criteria
            index = 0
            # print 'dnamsstd: ', evsdin,dnamsstd
            for idnam2,ednam2 in enumerate(dnamsstd):
                # print ednam,ednam2,idnam2,evsdin['dnams'][0:idnam2+1]
                if ednam2 in evsdin['dnams'][0:(idnam+1)]:
                    # print index
                    index = max(index,dnamsstd.index(ednam2) + 1)

            dnamsstd.insert(index,ednam)
            if ednam not in dnamsel:
                dimsstd.insert(index,int(vsdin[ivsdin]['dims'][idnam]))
            else:
                # In this case, wait for assigning the output dimensions. This actually depends on the specified function
                dimsstd.insert(index,None)
        else:
            if ((vsdin[ivsdin]['dims'][idnam] != 1) & (dimsstd[dnamsstd.index(ednam)] != 1) & \
                # we allow non-equal dimension lengths, as long as the dimension is covered/captured by the function
                # maybe still allow non-equal dimension length not covered by the function????
                (dimsstd[dnamsstd.index(ednam)] != None) & \
                (vsdin[ivsdin]['dims'][idnam] != dimsstd[dnamsstd.index(ednam)])):
                raise SomeError("The corresponding output dnamensions (index: "+str(dnamsstd.index(ednam))+") of the input variable "+str(ivsdin)+ " "+ str(idnam)+ " "+" have a different length and not equal to 1.")
            else:
                # None means it's considered by the function
                if (dimsstd[dnamsstd.index(ednam)] != None):
                    dimsstd[dnamsstd.index(ednam)] = max(dimsstd[dnamsstd.index(ednam)],vsdin[ivsdin]['dims'][idnam])

print 'Preliminary output dimensions: ', zip(dnamsstd,dimsstd)

idnam = 0
# add the missing dimensions selected for the function
for idnamsel,ednamsel in enumerate(dnamsel):
    if ednamsel not in dnamsstd:
        dnamsstd.insert(idnam,ednamsel)
        dimsstd.insert(idnam,None) # to be defined from the function
        idnam = idnam+1 # moet dit ook hier niet boven geimplementeerd worden?
    else:
        idnam = dnamsstd.index(ednam)+1


# adimsstd: list the specific output dimensions
# if function dimension: data output dimension should be the same as the function output dimension, but this should be checked afterwards.
# if not function dimension:
# # look what's the output dimension like. If the dimension is not in the output variable, we add a dummy 1-dimension
# we need to create/list adimsstd also before!! And then append them with the missing dimensions, as dummy 1-dimensions. If that is not sufficient, we will just get an error message.


# get references to the standard output dimensions on which the function is applied
refdfuncstd = []
for idnamsel,ednamsel in enumerate(dnamsel):
    refdfuncstd.append(dnamsstd.index(ednamsel))

# all output dimensions are now collected...
# add the standard output dimensions that are missing in each seperate input variable  as a dummy 1-dimension
for ivsdin,evsdin in enumerate(vsdin):
    idnam = 0
    for idnamsstd,ednamsstd in enumerate(dnamsstd):
        if ednamsstd not in vsdin[ivsdin]['dnams']:
            vsdin[ivsdin]['dnams'].insert(idnam,ednamsstd)
            vsdin[ivsdin]['dims'].insert(idnam,1)
            idnam = idnam + 1
        else:
            idnam = vsdin[ivsdin]['dnams'].index(ednamsstd) + 1


# do the same for the data output variables

# # vsdin[ivsdin]['refdstd']: references of data stream dimensions (vsdin[..]['dnams'] to the standard dimensions (dnamsstd) 
for ivsdin,evsdin in enumerate(vsdin):
    vsdin[ivsdin]['refdstd']= list([])
    for idim,edim in enumerate(vsdin[ivsdin]['dnams']):
        vsdin[ivsdin]['refdstd'].append(dnamsstd.index(edim))

for ivsdout,evsdout in enumerate(vsdout):
    if vsdout[ivsdout]['dnams'] == None:
        vsdout[ivsdout]['dnams'] = dnamsstd

# adimfuncin: the input dimensions of the function based on the refdfuncstd


# adimfuncin: the dimensions of the function input
adimfuncin = np.zeros((len(vsdin),len(refdfuncstd)),dtype='int32') - 1
alenfuncin = []

for ivsdout in range(len(vsdout)):
    if vsdout[ivsdout]['dnams'] == None:
        vsdout[ivsdout]['dnams'] == dnamsstd

# vsdout[..]['refdstd']: references of data stream dimensions (vsdout[..]['dnams'] to the standard dimensions (dnamsstd) 
for ivsdout,evsdout in enumerate(vsdout):
    vsdout[ivsdout]['refdstd'] = list([])
    for idim,edim in enumerate(vsdout[ivsdout]['dnams']):
        vsdout[ivsdout]['refdstd'].append(dnamsstd.index(edim))

# arefdfuncout: references of the function dimensions to the data output stream dimensions
arefdfuncout = []
for ivsdout,evsdout in enumerate(vsdout):
    arefdfuncout.append([])
    for idnamsel,ednamsel in enumerate(dnamsel):
        arefdfuncout[ivsdout].append(vsdout[ivsdout]['dnams'].index(ednamsel))
        # is arefdfuncout[ivsdout][irefdfuncout] == vsdout[ivsdout]['refdstd'].index(erefdfuncstd) ???

# arefdfuncin: references of the function dimensions to the data input stream dimensions
arefdfuncin = []
for ivsdin,evsdin in enumerate(vsdin):
    arefdfuncin.append([])
    for idnamsel,ednamsel in enumerate(dnamsel):
        arefdfuncin[ivsdin].append(vsdin[ivsdin]['dnams'].index(ednamsel))

# to do next:::...
for ivsdin,evsdin in enumerate(vsdin):
    for irefdfuncstd,erefdfuncstd in enumerate(refdfuncstd):
        adimfuncin[ivsdin,irefdfuncstd] = evsdin['dims'][vsdin[ivsdin]['refdstd'].index(erefdfuncstd)]
    alenfuncin.append(reduce(mul,adimfuncin[ivsdin]))

# 'probe' function output dimensions
dummydat = []
for ivsdin,evsdin in enumerate(vsdin):
    dummydat.append(np.zeros(adimfuncin[ivsdin]))
ddout =  func(*dummydat)
if (type(ddout).__name__ == 'tuple'):
    ddout = list(ddout)
if (type(ddout).__name__ != 'list'):
    ddout = list([ddout])

# obtain output data type. If not specified, we obtain it from the function output.
# meanwhile, check whether the number of input dimensions are the same as the number of output dimensions.

if len(ddout) != len(vsdout):
    raise SomeError('the amount of output variables in from '+ str(func) + ' ('+str(len(ddout))+') is not the same as specified ('+str(len(vsdout))+')')


for iddout in range(len(ddout)):
    if type(ddout[iddout] ) != np.ndarray: 
        ddout[iddout] = np.array(ddout[iddout])

    if (len(np.array(ddout[iddout]).shape) != len(adimfuncin[iddout])):
        raise SomeError('The amount of input ('+str(len(adimfuncin[iddout]))+') and output dimensions ('+str(len(ddout[iddout].shape))+') of function  is not the same')

    if vsdout[iddout]['dims'] == None:
        vsdout[iddout]['dims'] = dimsstd
        # overwrite dimensions with the function output dimensions
        for irefdfuncout,erefdfuncout in enumerate(arefdfuncout[iddout]):
            vsdout[iddout]['dims'][erefdfuncout] = ddout[iddout].shape[irefdfuncout]

    if vsdout[iddout]['dtype'] == None:
        # output netcdf variable does not exist... creating
        # why does this needs to be little endian????
        vsdout[iddout]['dtype'] = '>'+nctypecode(ddout[iddout].dtype)

        # try to copy dimension from data input
        for idim,edim in enumerate(vsdout[iddout]['dnams']):
            if edim not in datout[iddout][0].dimensions:
                dimensionfound = False
                idatin = 0
                # try to copy the dimension from the input data
                while ((not dimensionfound) & (idatin < (len(datin) ))):
                    if edim in datin[idatin][0].dimensions:
                        if (vsdout[iddout]['dims'][idim] == datin[idatin][0].dimensions[edim]):
                            print datin[idatin][0],datout[iddout][0], edim
                            nccopydimension(datin[idatin][0],datout[iddout][0], edim) 
                            dimensionfound = True
                    idatin = idatin + 1
                if dimensionfound == False:
                    datout[iddout][0].createDimension(edim,vsdout[iddout]['dims'][idim])

        datout[iddout][0].createVariable(datout[iddout][1],vsdout[iddout]['dtype'][1],tuple(vsdout[iddout]['dnams']))
        # we should check this at the time the dimensions are not created
        if (vsdout[iddout]['dims'] != list(datout[iddout][0].variables[datout[iddout][1]].shape)):
            raise SomeError("dimensions of output file ( "+str(vsdout[iddout]['dims'])+"; "+ str(vsdout[iddout]['dnams'])+") do not correspond with intended output dimension "+str(datout[iddout][0].variables[datout[iddout][1]].shape)+"; "+str(datout[iddout][0].variables[datout[iddout][1]].dimensions))


for idatin,edatin in enumerate(datin):
    # obtain file pointer!! very nasty!!
    ncfn =  str(datin[idatin][0])[19:(str(datin[idatin][0]).index("'",19))]
    vsdin[idatin]['fp'] = open(ncfn,'r')


for idatout,edatout in enumerate(datout):
    # obtain file pointer!! very nasty!!
    datout[idatout][0].flush()
    ncfn =  str(datout[idatout][0])[19:(str(datout[idatout][0]).index("'",19))]
    vsdout[idatout]['fp'] = open(ncfn,'r+')

    # in order to discover variable offsets
    nctemp = netcdf.netcdf_file(ncfn,'r')
    vsdout[idatout]['itemsize'] = nctemp.variables[datout[idatout][1]].itemsize()
    vsdout[idatout]['voffset'] = nctemp.variables[datout[idatout][1]]._voffset
    nctemp.close()

# # next: check whether the output variable dimensions (if already present) are not too large, otherwise raise error. + Construct final output dimension specs


# to do next:::...
# adimfuncout: the dimensions of the function output
adimfuncout = np.zeros((len(vsdout),len(refdfuncstd)),dtype='int32') - 1
alenfuncout = []
for ivsdout,evsdout in enumerate(vsdout):
    for irefdfuncstd,erefdfuncstd in enumerate(refdfuncstd):
        adimfuncout[ivsdout,irefdfuncstd] = evsdout['dims'][vsdout[ivsdout]['refdstd'].index(erefdfuncstd)]
    # # or ...
    # for irefdfuncout,erefdfuncout in enumerate(arefdfuncout[ivsdout]):
    #     adimfuncout[ivsdout,irefdfuncstd] = evsdout['dims'][erefdfuncout]
    alenfuncout.append(reduce(mul,adimfuncout[ivsdout]))
    # ???arefdfuncout[ivsdout][irefdfuncout] == vsdout[ivsdout]['refdstd'].index(erefdfuncstd)

# make copies of adimfunc*,  alenfunc*, arefdfunc*

# lennoiterstd = list(lenfuncstd)
# dimnoiterstd = list(dimdfuncstd)
refdnoiterstd = list(refdfuncstd)

alendnoiterin = list(alenfuncin)
adimnoiterin = []
arefdnoiterin = []
for ivsdin,evsdin in enumerate(vsdin):
    adimnoiterin.append(list(adimfuncin[ivsdin]))
    arefdnoiterin.append(list(arefdfuncin[ivsdin]))

alendnoiterout = list(alenfuncout)
adimnoiterout = []
arefdnoiterout = []
for ivsdout,evsdout in enumerate(vsdout):
    adimnoiterout.append(list(adimfuncout[ivsdout]))
    arefdnoiterout.append(list(arefdfuncout[ivsdout]))

# arefsin: references of the standard dimensions to the data stream dimensions

arefsin = []
for ivsdin,evsdin in enumerate(vsdin):
    arefsin.append([None]*len(vsdin[ivsdin]['refdstd']))
    # loop over the data stream dimensions

    for irefdstd,erefdstd in enumerate(vsdin[ivsdin]['refdstd']):
        arefsin[ivsdin][erefdstd] = irefdstd

# arefsout: references of the standard dimensions to the data stream dimensions

arefsout = []
for ivsdout,evsdout in enumerate(vsdout):
    arefsout.append([None]*len(vsdout[ivsdout]['refdstd']))
    # loop over the data stream dimensions

    for irefdstd,erefdstd in enumerate(vsdout[ivsdout]['refdstd']):
        arefsout[ivsdout][erefdstd] = irefdstd

dnamselnoiter = list(dnamsel)

# membytes: minimum total memory that will be used. We will the increase usage  when possible/allowed.
membytes = 0
for ivsdin,evsdin in enumerate(vsdin):
    membytes = membytes + alenfuncin[ivsdin] * vsdin[ivsdin]['itemsize']

for ivsdout,evsdout in enumerate(vsdout):
    membytes = membytes + alenfuncout[ivsdout] * vsdout[ivsdout]['itemsize']

maxmembytes = 1000000
if membytes > maxmembytes:
    print 'Warning, used memory ('+str(membytes)+') exceeds maximum memory ('+str(maxmembytes)+').'
else:

    # a temporary copy of alennoiter*
    alendnoiterin_tmp = list(alendnoiterin)
    alendnoiterout_tmp = list(alendnoiterout)
    # we try will to read the data in even larger icecubes to reduce disk access!
    idnam = len(dnamsstd) - 1
    
    cont = True
    while ((idnam >= 0) & (membytes <= maxmembytes) & cont):
    # while loop quite extensive but does what is should-> should be reduced and simplified
        cont = False # only continue to the next loop if idnam+1 (in previous loop) was (inserted) in refdnoiterstd
        if idnam not in refdnoiterstd:
            for ivsdin,evsdin in enumerate(vsdin):
                alendnoiterin_tmp[ivsdin] = alendnoiterin_tmp[ivsdin] *vsdin[ivsdin]['dims'][arefsin[ivsdin][idnam]]
            for ivsdout,evsdout in enumerate(vsdout):
                alendnoiterout_tmp[ivsdout] = alendnoiterout_tmp[ivsdout] *vsdout[ivsdout]['dims'][arefsout[ivsdout][idnam]]

            # recalculate the amount of bytes
            tmpmembytes = 0
            for ivsdin,evsdin in enumerate(vsdin):
                tmpmembytes = tmpmembytes + alendnoiterin_tmp[ivsdin] * vsdin[ivsdin]['itemsize']
            
            for ivsdout,evsdout in enumerate(vsdout):
                tmpmembytes = tmpmembytes + alendnoiterout_tmp[ivsdout] * vsdout[ivsdout]['itemsize']

            print 'tmpmembytes', tmpmembytes, membytes
            # if used memory still below threshold, we add it to the current dimension to the icecubes
            if tmpmembytes <= maxmembytes:
                refdnoiterstd.insert(0,idnam)
                for ivsdin,evsdin in enumerate(vsdin):
                    arefdnoiterin[ivsdin].insert(0, arefsin[ivsdin][idnam])
                    adimnoiterin[ivsdin].insert(0,vsdin[ivsdin]['dims'][arefsin[ivsdin][idnam]])
                    alendnoiterin[ivsdin] = alendnoiterin[ivsdin] *vsdin[ivsdin]['dims'][arefsin[ivsdin][idnam]]
                for ivsdout,evsdout in enumerate(vsdout):
                    arefdnoiterout[ivsdout].insert(0, arefsout[ivsdout][idnam])
                    adimnoiterout[ivsdout].insert(0,vsdout[ivsdout]['dims'][arefsout[ivsdout][idnam]])
                    alendnoiterout[ivsdout] = alendnoiterout[ivsdout] *vsdout[ivsdout]['dims'][arefsout[ivsdout][idnam]]
                dnamselnoiter.insert(0,dnamsstd[idnam])

                # recalculate the amount of bytes
                membytes = 0
                for ivsdin,evsdin in enumerate(vsdin):
                    membytes = membytes + alendnoiterin[ivsdin] * vsdin[ivsdin]['itemsize']
                
                for ivsdout,evsdout in enumerate(vsdout):
                    membytes = membytes + alendnoiterout[ivsdout] * vsdout[ivsdout]['itemsize']

                print 'membytes',membytes
                cont = True
                # if used memory still below threshold, we add it to the current dimension to the icecubes

        else:
            cont = True
        idnam = idnam - 1
                

#        adimnoiterin[ivsdin,irefdnoiterstd] = evsdin['dims'][vsdin[ivsdin]['refdstd'].index(erefdnoiterstd)]
    
    
# arefdfuncin: references of the function dimensions to the data input stream dimensions
# arefdnoiterin: references of the icecube dimensions to the data input stream dimensions
# # vsdin[ivsdin]['refdstd']: references of data stream dimensions (vsdin[..]['dnams'] to the standard dimensions (dnamsstd) 
# dnamselnoiter: references

# guess from residual dimensions that are not in refnoiterin
refditerstd = []
dimiterstd = []
for idim,edim in enumerate(dimsstd):
    if idim not in refdnoiterstd:
        refditerstd.append(idim)
        dimiterstd.append(edim) 

# guess from residual dimensions that are not in refnoiterin
arefditerin = []
adimiterin = []
for ivsdin,evsdin in enumerate(vsdin):
    arefditerin.append([]) 
    adimiterin.append([]) 
    for idim,edim in enumerate(vsdin[ivsdin]['dims']):
        if idim not in arefdnoiterin[ivsdin]:
            arefditerin[ivsdin].append(idim)
            adimiterin[ivsdin].append(edim) 

# guess from residual dimensions that are not in refnoiterin
arefditerout = []
adimiterout = []
for ivsdout,evsdout in enumerate(vsdout):
    arefditerout.append([]) 
    adimiterout.append([]) 
    for idim,edim in enumerate(vsdout[ivsdout]['dims']):
        if idim not in arefdnoiterout[ivsdout]:
            arefditerout[ivsdout].append(idim)
            adimiterout[ivsdout].append(edim) 

dimitermax = []
for iref,eref in enumerate(refditerstd):
    dimitermax.append(1)
    for ivsdin,evsdin in enumerate(vsdin):
        dimitermax[iref] = max(dimitermax[iref],adimiterin[ivsdin][iref])
        print dimitermax[iref], adimiterin[ivsdin][iref]
    for ivsdout,evsdout in enumerate(vsdout):
        dimitermax[iref] = max(dimitermax[iref],adimiterout[ivsdout][iref])


rwchunksizein = [1]*len(vsdin)
for ivsdin,evsdin in enumerate(vsdin):
    idim = len(vsdin[ivsdin]['dims']) -1
    while ((idim in arefdnoiterin[ivsdin]) & (idim >= 0)):
        # The inner dimensions just have to be referenced so not in correct order.  We know that they will be read in the correct order in the end
        rwchunksizein[ivsdin] = rwchunksizein[ivsdin]*vsdin[ivsdin]['dims'][idim]
        idim = idim - 1

rwchunksizeout = [1]*len(vsdout)
for ivsdout,evsdout in enumerate(vsdout):
    idim = len(vsdout[ivsdout]['dims']) -1
    while ((idim in arefdnoiterout[ivsdout]) & (idim >= 0)):
        # The inner dimensions just have to be referenced so not in correct order.  We know that they will be read in the correct order in the end
        rwchunksizeout[ivsdout] = rwchunksizeout[ivsdout]*vsdout[ivsdout]['dims'][idim]
        idim = idim - 1


adimnoapplyout = []
alennoapplyout = []
for ivsdout,evsdout in enumerate(vsdout):
    adimnoapplyout.append([])
    alennoapplyout.append(1)
    for irefdnoiterout in range(len(arefdnoiterout[ivsdout])-len(arefdfuncout[ivsdout])):
        adimnoapplyout[ivsdout].append(adimnoiterout[ivsdout][irefdnoiterout])
        alennoapplyout[ivsdout] =alennoapplyout[ivsdout]*adimnoapplyout[ivsdout][-1]

    if adimnoapplyout[ivsdout] == []:
        adimnoapplyout[ivsdout] = [1]

adimnoapplyin = []
alennoapplyin = []
for ivsdin,evsdin in enumerate(vsdin):
    adimnoapplyin.append([])
    alennoapplyin.append(1)
    for irefdnoiterin in range(len(arefdnoiterin[ivsdin])-len(arefdfuncin[ivsdin])):
        adimnoapplyin[ivsdin].append(adimnoiterin[ivsdin][irefdnoiterin])
        alennoapplyin[ivsdin] =alennoapplyin[ivsdin]*adimnoapplyin[ivsdin][-1]

    if adimnoapplyin[ivsdin] == []:
        adimnoapplyin[ivsdin] = [1]

dimnoapplymax = []
for iref in range(len(arefdnoiterout[ivsdout])-len(arefdfuncout[ivsdout])):
    dimnoapplymax.append(1)
    for ivsdin,evsdin in enumerate(vsdin):
        dimnoapplymax[iref] = max(dimnoapplymax[iref],adimnoapplyin[ivsdin][iref])
        print dimnoapplymax[iref], adimnoapplyin[ivsdin][iref]
    for ivsdout,evsdout in enumerate(vsdout):
        dimnoapplymax[iref] = max(dimnoapplymax[iref],adimnoapplyout[ivsdout][iref])

lennoapplymax = reduce(mul,dimnoapplymax)



testdata = np.zeros(vsdout[0]['dims']).ravel()


lenitermax = reduce(mul,dimitermax)
dimiterpos = [0]*len(dimitermax)
print str(0)+'/'+str(lenitermax),
for j in range(lenitermax):
    # reading icecube, rearranged in the order of dimensions specified by arefnoiterin
    dataicecubein = []
    for ivsdin,evsdin in enumerate(vsdin):
        # dataicecubein.append(np.zeros((elendnoiterin,),dtype=vsdin[ilendnoiterin]['dtype'])) 
        dataicecubein.append(np.array(readicecubeps(\
            vsdin[ivsdin]['fp'],\
            vsdin[ivsdin]['dims'],\
            arefditerin[ivsdin],\
            adimiterin[ivsdin],\
            dimiterpos,\
            arefdnoiterin[ivsdin],\
            adimnoiterin[ivsdin],\
            vsdin[ivsdin]['dtype'],\
            vsdin[ivsdin]['itemsize'],\
            vsdin[ivsdin]['voffset'],\
            rwchunksizein[ivsdin],\
            ), dtype=vsdin[ivsdin]['dtype']).ravel())

    dataicecubeout = []
    for ilendnoiterout,elendnoiterout in enumerate(alendnoiterout):
        dataicecubeout.append(np.zeros((elendnoiterout,),dtype=vsdout[ilendnoiterout]['dtype'][1])) 

    dimnoapplypos = [0]*len(dimnoapplymax)
    for k in range(lennoapplymax):
        # actually, this is just the end of the file output already written
        ahunkin = []
        for ivsdin, evsdin in enumerate(vsdin):
            pos = 0
            # e.g. pos = (9)+ 20*(10) + 50*50*20*(5)
            for idimpos,edimpos in enumerate(dimnoapplypos):
                curadd    = np.mod(edimpos,adimnoapplyin[ivsdin][idimpos])
                #e.g. if edimpos == (5): curadd = 50*50*20*(5)
                if ((idimpos + 1) < len(arefdnoiterin[ivsdin])):
                    for i in range(idimpos + 1,len(arefdnoiterin[ivsdin])) :
                                                # here, we assume that the dimensions of the chunk are already in the order considered by adimsnoiter(out) etc. (cfr. preceeded transposition in readicecubeps)
                        curadd    = curadd    * adimnoiterin[ivsdin][i]
                        # curaddout = curaddout * dimnoiteroutref[i]
                pos    = pos    + curadd
            ahunkin.append(dataicecubein[ivsdin][pos:(pos+alenfuncin[ivsdin])])
            ahunkin[ivsdin].shape = adimfuncin[ivsdin]

        # apply the function

        ahunkout =  func(*ahunkin)
        if (type(ahunkout).__name__ == 'tuple'):
            ahunkout = list(ahunkout)
        if (type(ahunkout).__name__ != 'list'):
            ahunkout = list([ahunkout])

        for ihunkout in range(len(ahunkout)):
            ahunkout[ihunkout] = np.array(ahunkout[ihunkout])
            # e.g. posout = (9)+ 20*(10) + 50*50*20*(5)
            posout    = 0
            for idimpos,edimpos in enumerate(dimnoapplypos):
                curadd    = np.mod(edimpos,adimnoapplyout[ihunkout][idimpos])
                #e.g. if edimpos == (5): curadd = 50*50*20*(5)
                if ((idimpos + 1) < len(arefdnoiterout[ihunkout])):
                    for i in range(idimpos + 1,len(arefdnoiterout[ihunkout])) :
                                                # here, we assume that the idims are in the intended order (cfr. subsequent transposition in writeicecubeps)
                        curadd    = curadd    * adimnoiterout[ihunkout][i]
                        # curaddout = curaddout * dimnoiteroutref[i]
                posout    = posout    + curadd

            dataicecubeout[ihunkout][posout:(posout+alenfuncout[ihunkout])] = np.array(ahunkout[ihunkout].ravel(),dtype=vsdout[ihunkout]['dtype'][1])

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
        dataicecubeout[idimsout].shape = adimnoiterout[idimsout]
        #print dataicecubeout[idimsout].shape


    for ivsdout in range(len(vsdout)):
        # print dataicecubeout[ivsdout].shape,vsdout[ivsdout]
        # print 'ivsdout', ivsdout
        writeicecubeps(\
            vsdout[ivsdout]['fp'],
            vsdout[ivsdout]['dims'],\
            arefditerout[ivsdout],\
            adimiterout[ivsdout],\
            dimiterpos,\
            arefdnoiterout[ivsdout],\
            adimnoiterout[ivsdout],\
            dataicecubeout[ivsdout],\
            vsdout[ivsdout]['dtype'],\
            vsdout[ivsdout]['itemsize'],\
            vsdout[ivsdout]['voffset'],\
            rwchunksizeout[ivsdout])

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

for ivsdin,evsdin in enumerate(vsdin):
    vsdin[ivsdin]['fp'].close()
for ivsdout,evsdout in enumerate(vsdout):
    vsdout[ivsdout]['fp'].close()

import pylab as pl
fout.close()
# fin.close()
# fout = NetCDF.NetCDFFile(fnout,'r')
fout = netcdf.netcdf_file(fnout,'r')
fout.fp.seek(vsdout[0]['voffset'])
# fpointout.seek(vsdout[0]['voffset'])
test = np.fromfile(fout.fp,dtype=vsdout[0]['dtype'],count=reduce(mul,vsdout[0]['dims']))
test.shape = (40,340)
fig = pl.figure()
pl.imshow(test)
fig.show()

fig = pl.figure()
testdata.shape = vsdout[0]['dims']
pl.imshow(testdata[0,:,:,0,1])
fig.show()

fout.close()
fout = NetCDF.NetCDFFile(fnout,'r')

fig = pl.figure()
pl.imshow(fout.variables['QV'][0,:,:,0,0])
fig.show()
fout.close()
