import os
from operator import itemgetter
import numpy as np
from netcdf import netcdf_file
import sys 
from operator import mul
from Scientific.IO import NetCDF

from ncdfextract import nctypecode
from ncdfproc import nccopydimension,nccopyattrvar


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
        lennoiter = lennoiter*len(dimnoiter[irefnoiter])

    fpos = 0
    # e.g. fpos = (9)+ 20*(10) + 50*50*20*(5)
    for idimpos,edimpos in enumerate(dimpos):
        curadd = np.mod(dimiter[idimpos][np.mod(edimpos,len(dimiter[idimpos]))],shp[refiter[idimpos]])
        #e.g. if edimpos == (5): curadd = 50*50*20*(5)

        # exclude trivial special case of only 1 iteration step
        # --> in that case fpos is just zero.
        if refiter > [-1]:
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
    j = 0
    while j < lennoiter:
        fposicecube = fpos
        for idimpos,edimpos in enumerate(dimnoiterpos):
            curadd = np.mod(dimnoiter[idimpos][np.mod(edimpos,len(dimnoiter[idimpos]))],shp[refnoiter[idimpos]])
            # e.g. fposicecube = (1)*52
            # e.g. fposicecube = (9)+ 20*(10) + 50*50*20*(5)
            if ((refnoiter[idimpos] + 1) < len(shp)):
                for i in range(refnoiter[idimpos] + 1,len(shp)) :
                    curadd = curadd * shp[i]
            fposicecube = fposicecube + curadd
        
        filestream.seek(voffset+vsize*fposicecube)
        if mode == 'read':
            temp = np.fromfile(filestream,dtype='>'+vtype[1],count=rwchsize)
            # temp.byteswap(True)
            icecube[j:(j+rwchsize)] = temp
        elif mode == 'write':
            temp = np.array(icecube[j:(j+rwchsize)],dtype='>'+vtype[1])
            filestream.write(temp)
    
        # go to next data strip 
        if dimnoiterpos != []:
            # rwchsize: allow reading of chunks for the inner dimensions
            dimnoiterpos[-1] = dimnoiterpos[-1] + rwchsize
            for idimidx,edimidx in enumerate(reversed(dimnoiterpos)):
                if idimidx > 0:
                    while dimnoiterpos[idimidx] >= len(dimnoiter[idimidx]):
                        dimnoiterpos[idimidx-1] = dimnoiterpos[idimidx-1] + 1
                        dimnoiterpos[idimidx] -= len(dimnoiter[idimidx])
        j = j+rwchsize
    
    icecube.shape = [len(e) for e in dimnoiter]
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

    # print 'shp',shp,refnoitersort,[len(e) for e in dimnoitersort],refiter
    if str(type(fstream))[7:11] == 'list':
	# this statement assumes that the first dimension of the datastream represents the 'listing' of the files
        if refnoitersort[0] == 0:
            # print([len(fstream)]+list([len(e) for e in dimnoitersort[1:]]))
            icecube = np.zeros([len(fstream)]+list([len(e) for e in dimnoitersort[1:]]))
#            icecube = np.zeros([len(fstream)]+list(dimnoitersort[1:]))
            refiterred = list(refiter); refiterred = [e -1 for e in refiterred]
            refnoitersortred = list(refnoitersort[1:]); refnoitersortred = [e - 1 for e in refnoitersortred]
            dimnoitersortred = list(dimnoitersort[1:])
            shpred = list(shp[1:])
     #        print 'shpred',shpred,refnoitersortred,[len(e) for e in dimnoitersortred],refiterred
            for ifn,efn in enumerate(fstream):
                tfile = open(fstream[ifn],'r')
                icecube[ifn] =rwicecube(tfile,shpred,refiterred,dimiter,dimiterpos,refnoitersortred,dimnoitersortred,None,vtype,vsize,voffset,rwchsize,'read') 
                tfile.close() 
        elif 0 in refiter:
            irefiter = refiter.index(0)
            ifn = dimiterpos[irefiter]
            tfile = open(fstream[ifn],'r')
            refiterred = list(refiter); refiterred.pop(irefiter); refiterred = [ e-1 for e in refiterred]
            dimiterred = list(dimiter); dimiterred.pop(irefiter)
            dimiterposred = list(dimiterpos);dimiterposred.pop(irefiter)
            shpred = list(shp[1:])
            refnoitersortred = list(refnoiter); refnoitersortred = [e-1 for e in refnoitersortred]
            icecube =rwicecube(tfile,shpred,refiterred,dimiterred,dimiterposred,refnoitersortred,dimnoitersort,None,vtype,vsize,voffset,rwchsize,'read') 
            tfile.close()
    else:
        icecube =rwicecube(fstream,shp,refiter,dimiter,dimiterpos,refnoitersort,dimnoitersort,None,vtype,vsize,voffset,rwchsize,'read') 


    # build the 'inverse permutation' operator for tranposition before writeout
    inv = range(len(trns))
    for itrns, etrns in enumerate(trns):
        inv[etrns] = itrns
    return np.transpose(icecube,inv)

# we kunnen eens proberen om een variabele aan te maken met een vooraf gespecifieerde dimensie!

def pcd(func,dnamsel,datin,datout,appenddim = False, predim = None,maxmembytes = 10000000):
    """ process binary data in order of specified dimensions
      func: the function to be used
    dnamsel: the dimensions on which the function needs to apply
    datin: list of input [file,variable]-pairs
    datin: list of output [file,variable]-pairs
    
      Warning! for now output variables will be re-opened in write mode!
    """
    
    # obtain definitions of the variable stream input
    vsdin = [] # input variable stream definitions
    for idatin,edatin in enumerate(datin):
        # read in scipy.netcdf mode to obtain varariable offsets
        
        vsdin.append(dict())
        if str(type(datin[idatin]['file']))[7:11] == 'list':
            ncfn =  datin[idatin]['file'][0]
            nctemp = netcdf_file(ncfn,'r')
            vsdin[idatin]['dnams'] = []
            for idim in range(len(nctemp.variables[datin[idatin]['varname']].dimensions)):
                edim = nctemp.variables[datin[idatin]['varname']].dimensions[idim]
                if 'daliases' in datin[idatin]:
                    if edim in datin[idatin]['daliases']:
                        edim = datin[idatin]['daliases'][edim]

                vsdin[idatin]['dnams'].append(str(edim))

	    print '1dnams', idatin,vsdin[idatin]['dnams']
            # this statement could be a problem when defining daliases!!!
            if ((nctemp.variables[datin[idatin]['varname']].shape[0] == 1) & ((predim == None) | (nctemp.variables[datin[idatin]['varname']].dimensions[0] == predim))):
                # we expand the first dimension
                vsdin[idatin]['dims'] = [len(datin[idatin]['file'])]+list(nctemp.variables[datin[idatin]['varname']].shape[1:])

            else: 
                # we add a new dimension # warning! uncatched problems can occur when predim0 already exists
                if predim == None:
                    predim = 'predim0'
                    idimextra = 0
                    while predim in vsdin[idatin]['dnams']:
                        predim = 'predim'+str(idimextra)
                        idimextra = idimextra + 1
                vsdin[idatin]['dnams'].insert(0,predim)
                vsdin[idatin]['dims'] = [len(datin[idatin]['file'])]+list(nctemp.variables[datin[idatin]['varname']].shape[:])
	    print '2dnams', idatin,vsdin[idatin]['dnams']
		
        else:
            # we assume a netcdf file
            if str(type(datin[idatin]['file']))[7:17]  == 'NetCDFFile':
                # obtain file name from open netcdf!! very nasty!!!
                ncfn =  str(datin[idatin]['file'])[19:(str(datin[idatin]['file']).index("'",19))]
            # we assume a file name
            elif str(type(datin[idatin]['file']))[7:10] == 'str':
                ncfn = datin[idatin]['file']
            else:
                raise SomeError("Input file "+ str(datin[idatin]) + " ("+str(idatin)+")  could not be recognized.")
            nctemp = netcdf_file(ncfn,'r')
            vsdin[idatin]['dnams'] = []
            for idim in range(len(nctemp.variables[datin[idatin]['varname']].dimensions)):
                edim = nctemp.variables[datin[idatin]['varname']].dimensions[idim]
                if 'daliases' in datin[idatin]:
                    if edim in datin[idatin]['daliases']:
                        edim = datin[idatin]['daliases'][edim]
                vsdin[idatin]['dnams'].append(str(edim))
            vsdin[idatin]['dims'] = list(nctemp.variables[datin[idatin]['varname']].shape)

        vsdin[idatin]['itemsize'] = nctemp.variables[datin[idatin]['varname']].itemsize()
        vsdin[idatin]['dtype']  =   nctemp.variables[datin[idatin]['varname']]._dtype
        vsdin[idatin]['voffset']  = nctemp.variables[datin[idatin]['varname']]._voffset

        # dselmarker
        vsdin[idatin]['dsel'] = [False]*len(vsdin[idatin]['dims'])
        if 'dsel' in datin[idatin]:
            # cropped dimensions
            for edcrop in datin[idatin]['dsel']:
                if edcrop in vsdin[idatin]['dnams']:
                    vsdin[idatin]['dsel'][vsdin[idatin]['dnams'].index(edcrop)] = list(datin[idatin]['dsel'][edcrop]) 
                else:
                   print("Warning, dimension '"+ str(edcrop) + "' not in netcdf variable '"+ncfn+ "("+datin[idatin]['varname']+")'.")
        nctemp.close()
    
    # obtain definitions of the variable stream output
    vsdout = [] # input variable stream definitions
    for idatout,edatout in enumerate(datout):
        vsdout.append(dict())
        if edatout['varname'] in edatout['file'].variables:
            vsdout[idatout]['dnams'] = []
            for idim,edim in enumerate(datout[idatout]['file'].variables[datout[idatout]['varname']].dimensions):
                vsdout[idatout]['dnams'].append(str(edim))
    
            vsdout[idatout]['dims'] = list(datout[idatout]['file'].variables[datout[idatout]['varname']].shape)
             # dselmarker
            vsdout[idatout]['dsel'] = [False]*len(vsdout[idatout])
            vsdout[idatout]['itemsize'] = datout[idatout]['file'].variables[datout[idatout]['varname']].itemsize()
            vsdout[idatout]['dtype']=     datout[idatout]['file'].variables[datout[idatout]['varname']]._dtype
            vsdout[idatout]['voffset'] =  datout[idatout]['file'].variables[datout[idatout]['varname']]._voffset
        else:
            # the variable doesn't exists (we will create it afterwards)
            vsdout[idatout]['dnams'] = None
            vsdout[idatout]['dims'] = None
            # dselmarker
            vsdout[idatout]['dsel'] = None
            vsdout[idatout]['itemsize'] = None
            vsdout[idatout]['dtype'] = None
    
    # collecting the involved dimensions (will be considered as the standard output dimensions)
    dnamsstd = [] # standard output dimensions: list of all output dimensions: this is collected from the input dimensions, the output dimensions and the selected/processed dimensions
    dimsstd = [] # maximum length of an output dimension
    idimsstd = 0
    
    for ivsdin,evsdin in enumerate(vsdin):
        dnaminlast = None
        idx = 0
        for idnam,ednam in reversed(list(enumerate(evsdin['dnams']))):
            if ednam not in dnamsstd:
                # In dnamsstd, ednam should be just after the dimensions preceding ednams in dnams  
                # # actually, we also want that, in dnamsstd, ednam should be just before the dimensions succeeding ednams in dnams. Sometimes, this is not possible at the same time. But it will be the case if that is possible when applying one of the criteria
                idx = 0
                for idnam2,ednam2 in enumerate(dnamsstd):
                    if ednam2 in evsdin['dnams'][0:(idnam+1)]:
                        idx =  max(idx  ,dnamsstd.index(ednam2) + 1)
    
                dnamsstd.insert(idx  ,ednam)
                if ednam not in dnamsel:
                    # dselmarker
                    if vsdin[ivsdin]['dsel'][idnam] != False: # type(edim).__name__ == 'list':
                        dimsstd.insert(idx  ,int(len(vsdin[ivsdin]['dsel'][idnam]))) 
                    else:
                        dimsstd.insert(idx  ,int(vsdin[ivsdin]['dims'][idnam]))
                else:
                    # In this case, wait for assigning the output dimensions. This actually depends on the specified function
                    dimsstd.insert(idx  ,None)
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
                        # dselmarker
                        if vsdin[ivsdin]['dsel'][idnam] != False:
                            dimsstd[dnamsstd.index(ednam)] = max(dimsstd[dnamsstd.index(ednam)],len(vsdin[ivsdin]['dsel'][idnam]))
                        else:
                            dimsstd[dnamsstd.index(ednam)] = max(dimsstd[dnamsstd.index(ednam)],vsdin[ivsdin]['dims'][idnam])
    
    
    idnam = 0

    idnam = len(dnamsstd)
    # add the missing dimensions selected for the function
    for idnamsel,ednamsel in reversed(list(enumerate(dnamsel))):
        if ednamsel not in dnamsstd:
            dnamsstd.insert(idnam,ednamsel)
            dimsstd.insert(idnam,None) # to be defined from the function
            idnam = idnam # moet dit ook hier niet boven geimplementeerd worden?
        else:
            idnam = dnamsstd.index(ednam)+1
    
    print "dimsstd", dimsstd
    
    # dimsstd: list the specific output dimensions
    # if function dimension: data output dimension should be the same as the function output dimension, but this should be checked afterwards.
    # if not function dimension:
    # # look what's the output dimension like. If the dimension is not in the output variable, we add a dummy 1-dimension
    # we need to create/list adimsstd also before!! And then append them with the missing dimensions, as dummy 1-dimensions. If that is not sufficient, we will just get an error message.
    
    
    # get references to the standard output dimensions on which the function is applied
    refdfuncstd = []
    for idnamsel,ednamsel in enumerate(dnamsel):
        refdfuncstd.append(dnamsstd.index(ednamsel))
    
    # all standard output dimensions (cfr. dimsstd, dnamsstd) are now collected...
    # add the standard output dimensions that are missing in each seperate input variable  as a dummy 1-dimension
    for ivsdin,evsdin in enumerate(vsdin):
        idnam = 0
	# the dimension of the list should always the first dimension! This is assumed in the rwicecube (see statement 'if refnoitersort[0] == 0:')
        if type(datin[idatin]['file']).__name__  == 'list':
            idnam = 1

        for idnamsstd,ednamsstd in enumerate(dnamsstd):
            if ednamsstd not in vsdin[ivsdin]['dnams']:
                vsdin[ivsdin]['dnams'].insert(idnam,ednamsstd)
                vsdin[ivsdin]['dims'].insert(idnam,1)
                vsdin[ivsdin]['dsel'].insert(idnam,False)
                # dselmarker
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
    # adimfuncin = np.zeros((len(vsdin),len(refdfuncstd)),dtype='int32') - 1
    
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
    
        

    adimfuncin = []
    alendfuncin = []
    for ivsdin,evsdin in enumerate(vsdin):
        alendfuncin.append(1)
        adimfuncin.append([])
        for irefdfuncstd,erefdfuncstd in enumerate(refdfuncstd):
            # dselmarker
            # edim = evsdin['dims'][vsdin[ivsdin]['refdstd'].index(erefdfuncstd)]
            if evsdin['dsel'][vsdin[ivsdin]['refdstd'].index(erefdfuncstd)] != False:
                adimfuncin[ivsdin].append(list(evsdin['dsel'][vsdin[ivsdin]['refdstd'].index(erefdfuncstd)]))
            else:
                adimfuncin[ivsdin].append(range(evsdin['dims'][vsdin[ivsdin]['refdstd'].index(erefdfuncstd)]))
            alendfuncin[ivsdin] = alendfuncin[ivsdin]*len(adimfuncin[ivsdin][irefdfuncstd])
    
    # 'probe' function output dimensions
    dummydat = []
    for ivsdin,evsdin in enumerate(vsdin):
        dummydat.append(np.zeros([len(e) for e in adimfuncin[ivsdin]]))
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
    
        if (len(np.array(ddout[iddout]).shape) != len(adimfuncin[ivsdin])):
            raise SomeError('The amount of input ('+str(len(adimfuncin[ivsdin]))+') and output dimensions ('+str(len(ddout[iddout].shape))+') of function  is not the same')
    
        if vsdout[iddout]['dims'] == None:
            vsdout[iddout]['dims'] = list(dimsstd)
            vsdout[iddout]['dsel'] = [False]*len(vsdout[iddout]['dims']) # Added for consistency with input dimensions. As output dimension, it is not really used.
            # overwrite dimensions with the function output dimensions
            for irefdfuncout,erefdfuncout in enumerate(arefdfuncout[iddout]):
                vsdout[iddout]['dims'][erefdfuncout] = ddout[iddout].shape[irefdfuncout]
    
        if vsdout[iddout]['dtype'] == None:
            # output netcdf variable does not exist... creating
            # why does this needs to be little endian????
            vsdout[iddout]['dtype'] = '>'+nctypecode(ddout[iddout].dtype)
    
            # try to copy dimension from data input
            for idnams,ednams in enumerate(vsdout[iddout]['dnams']):
                if ednams not in datout[iddout]['file'].dimensions:
                    dimensionfound = False
                    idatin = 0
                    while ((not dimensionfound) & (idatin < (len(datin) ))):
                        templopen = False
                        # try to copy the dimension from the input data
                        if str(type(datin[idatin]['file']))[7:17]  == 'NetCDFFile':
                            nctemplate = datin[idatin]['file']
                        elif str(type(datin[idatin]['file']))[7:10]  == 'str':
                            nctemplate = NetCDF.NetCDFFile(datin[idatin]['file'],'r')
                            templopen = True
                        elif str(type(datin[idatin]['file']))[7:11]  == 'list':
                            nctemplate = NetCDF.NetCDFFile(datin[idatin]['file'][0],'r')
                            templopen = True
                        else:
                            raise SomeError("Input file "+ str(datin[idatin]) + " ("+str(idatin)+")  could not be recognized.")
                        if ednams in nctemplate.dimensions:
                            # dselmarker

                            if vsdin[idatin]['dsel'][idnams] != False: # type(edim).__name__ == 'list':
                                if (vsdout[iddout]['dims'][idnams] == len(vsdin[idatin]['dsel'][idnams])):
                                    datout[iddout]['file'].createDimension(ednams,vsdout[iddout]['dims'][idnams])
                                    if ednams in nctemplate.variables:
                                        datout[iddout]['file'].createVariable(ednams,nctemplate.variables[ednams].typecode(),(ednams,))
                                        datout[iddout]['file'].variables[ednams][:] = np.array(nctemplate.variables[ednams][:])[vsdin[idatin]['dsel'][idnams]]
                                        nccopyattrvar(nctemplate,datout[iddout]['file'],varin=ednams,)
                                    dimensionfound = True
                            else:

                                if (vsdout[iddout]['dims'][idnams] == nctemplate.dimensions[ednams]):
                                    nccopydimension(nctemplate,datout[iddout]['file'], ednams) 
                                    dimensionfound = True


                        if templopen:
                            nctemplate.close()
                        idatin = idatin + 1
                    if dimensionfound == False:
                        datout[iddout]['file'].createDimension(ednams,vsdout[iddout]['dims'][idnams])
                    # if  a template file needed to be opened, close it again

            # check here whether the output dimensions are ok , otherwise make alternative dimensions
            dnams = []
            for idim,edim in enumerate(vsdout[iddout]['dims']):
                if (vsdout[iddout]['dims'][idim] == datout[iddout]['file'].dimensions[vsdout[iddout]['dnams'][idim]]):
                    dnams.append(vsdout[iddout]['dnams'][idim])
                else:
                    # if (vsdout[iddout]['dims'][idim] == 1):
                    #     # if dimension is one, just remove it
                    #     vsdout[iddout]['dims'].pop(idim)
                    # else:
                    dnams.append(vsdout[iddout]['dnams'][idim])
                    # else, make a new alternative dimension with a similar name
                    dnamidx = -1
                    while (dnams[idim] in datout[iddout]['file'].dimensions):
                        dnamidx = dnamidx + 1
                        dnams[idim] = str(vsdout[iddout]['dnams'][idim])+'_'+str(dnamidx)
                    datout[iddout]['file'].createDimension(dnams[idim],vsdout[iddout]['dims'][idim])
            vsdout[iddout]['dnams'] = dnams


            datout[iddout]['file'].createVariable(datout[iddout]['varname'],vsdout[iddout]['dtype'][1],tuple(vsdout[iddout]['dnams']))
            # we should check this at the time the dimensions are not created
            if (vsdout[iddout]['dims'] != list(datout[iddout]['file'].variables[datout[iddout]['varname']].shape)):
                raise SomeError("dimensions of output file ( "+str(vsdout[iddout]['dims'])+"; "+ str(vsdout[iddout]['dnams'])+") do not correspond with intended output dimension "+str(datout[iddout]['file'].variables[datout[iddout]['varname']].shape)+"; "+str(datout[iddout]['file'].variables[datout[iddout]['varname']].dimensions))
    
    
    for idatin,edatin in enumerate(datin):
        if str(type(datin[idatin]['file']))[7:17]  == 'NetCDFFile':
            # obtain file pointer!! very nasty!!
            ncfn =  str(datin[idatin]['file'])[19:(str(datin[idatin]['file']).index("'",19))]
            vsdin[idatin]['fp'] = open(ncfn,'r')
        elif str(type(datin[idatin]['file']))[7:10]  == 'str':
            ncfn = datin[idatin]['file']
            vsdin[idatin]['fp'] = open(ncfn,'r')
        elif str(type(datin[idatin]['file']))[7:11]  == 'list':
            # !!!!!if series/list of file names, then the file poniters will open when at read-time
            ncfn = datin[idatin]['file']
            vsdin[idatin]['fp'] = datin[idatin]['file']
        else:
            raise SomeError("Input file "+ str(datin[idatin]) + " ("+str(idatin)+")  could not be recognized.")
    
    for idatout,edatout in enumerate(datout):
        # obtain file pointer!! very nasty!!
        datout[idatout]['file'].flush()
        ncfn =  str(datout[idatout]['file'])[19:(str(datout[idatout]['file']).index("'",19))]
        vsdout[idatout]['fp'] = open(ncfn,'r+')
    
        # in order to discover variable offsets
        nctemp = netcdf_file(ncfn,'r')
        vsdout[idatout]['itemsize'] = nctemp.variables[datout[idatout]['varname']].itemsize()
        vsdout[idatout]['voffset'] = nctemp.variables[datout[idatout]['varname']]._voffset
        nctemp.close()
    
    # # next: check whether the output variable dimensions (if already present) are not too large, otherwise raise error. + Construct final output dimension specs
    


    adimfuncout = [[None]*len(refdfuncstd)]*(len(vsdout))
    alendfuncout = []
    for ivsdout,evsdout in enumerate(vsdout):
        alendfuncout.append(1)
        for irefdfuncstd,erefdfuncstd in enumerate(refdfuncstd):
            # dselmarker
            # edim = evsdout['dims'][vsdout[ivsdout]['refdstd'].outdex(erefdfuncstd)]
            if evsdout['dsel'][vsdout[ivsdout]['refdstd'].index(erefdfuncstd)] != False:
                adimfuncout[ivsdout][irefdfuncstd] = evsdout['dsel'][vsdout[ivsdout]['refdstd'].index(erefdfuncstd)]
            else:
                adimfuncout[ivsdout][irefdfuncstd] = range(evsdout['dims'][vsdout[ivsdout]['refdstd'].index(erefdfuncstd)])
            alendfuncout[ivsdout] = alendfuncout[ivsdout]*len(adimfuncout[ivsdout][irefdfuncstd])


    
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
    
    print "dnamsstd", dnamsstd
    if appenddim == True:
        membytes = 0
        # dsellen = len(dnamsel)
        # a temporary copy of alenfunc*
        alendfuncin_tmp = list(alendfuncin)
        alendfuncout_tmp = list(alendfuncout)
        # we try will to read the data in even larger icecubes to reduce disk access!
        idnam = len(dnamsstd) - 1
        
        cont = True
        maxarefdfunc = len(refdfuncstd) 
        while ((idnam >= maxarefdfunc) & (membytes <= maxmembytes) & cont):
        # while loop quite extensive but does what is should-> should be reduced and simplified
            cont = False # only continue to the next loop if idnam+1 (in previous loop) was (inserted) in refdfuncstd

            ednam = dnamsstd[idnam]
            if idnam not in refdfuncstd:
                for ivsdin,evsdin in enumerate(vsdin):
                    # dselmarker
                    if vsdin[ivsdin]['dsel'][arefsin[ivsdin][idnam]] != False:
                        alendfuncin_tmp[ivsdin] = alendfuncin_tmp[ivsdin] *len(vsdin[ivsdin]['dsel'][arefsin[ivsdin][idnam]])
                    else:
                        alendfuncin_tmp[ivsdin] = alendfuncin_tmp[ivsdin] *vsdin[ivsdin]['dims'][arefsin[ivsdin][idnam]]
                for ivsdout,evsdout in enumerate(vsdout):
                    alendfuncout_tmp[ivsdout] = alendfuncout_tmp[ivsdout] *vsdout[ivsdout]['dims'][arefsout[ivsdout][idnam]]
    
                # recalculate the amount of bytes
                tmpmembytes = 0
                for ivsdin,evsdin in enumerate(vsdin):
                    tmpmembytes = tmpmembytes + alendfuncin_tmp[ivsdin] * vsdin[ivsdin]['itemsize']
                
                for ivsdout,evsdout in enumerate(vsdout):
                    tmpmembytes = tmpmembytes + alendfuncout_tmp[ivsdout] * vsdout[ivsdout]['itemsize']
    
                # if used memory still below threshold, we add it to the current dimension to the icecubes
                if tmpmembytes <= maxmembytes:
                    refdfuncstd.insert(maxarefdfunc,idnam)
                    for ivsdin,evsdin in enumerate(vsdin):

                       # arefdfuncin: references of the function dimensions to the data input stream dimensions
                        arefdfuncin[ivsdin].insert(maxarefdfunc, arefsin[ivsdin][idnam])
                        if vsdin[ivsdin]['dsel'][arefsin[ivsdin][idnam]] != False:
                            adimfuncin[ivsdin].insert(maxarefdfunc,vsdin[ivsdin]['dsel'][arefsin[ivsdin][idnam]])
                            alendfuncin[ivsdin] = alendfuncin[ivsdin] *len(vsdin[ivsdin]['dsel'][arefsin[ivsdin][idnam]])
                        else:
                            adimfuncin[ivsdin].insert(maxarefdfunc,range(vsdin[ivsdin]['dims'][arefsin[ivsdin][idnam]]))
                            alendfuncin[ivsdin] = alendfuncin[ivsdin] *vsdin[ivsdin]['dims'][arefsin[ivsdin][idnam]]
                    for ivsdout,evsdout in enumerate(vsdout):
                        arefdfuncout[ivsdout].insert(maxarefdfunc, arefsout[ivsdout][idnam])
                        adimfuncout[ivsdout].insert(maxarefdfunc,range(vsdout[ivsdout]['dims'][arefsout[ivsdout][idnam]]))
                        alendfuncout[ivsdout] = alendfuncout[ivsdout] *vsdout[ivsdout]['dims'][arefsout[ivsdout][idnam]]
                    #dnamsel.insert(dsellen,dnamsstd[idnam])
    
                    # recalculate the amount of bytes
                    membytes = 0
                    for ivsdin,evsdin in enumerate(vsdin):
                        membytes = membytes + alendfuncin[ivsdin] * vsdin[ivsdin]['itemsize']
                    
                    for ivsdout,evsdout in enumerate(vsdout):
                        membytes = membytes + alendfuncout[ivsdout] * vsdout[ivsdout]['itemsize']
    
                    cont = True
                    # if used memory still below threshold, we add it to the current dimension to the icecubes
    
            else:
                cont = True
            idnam = idnam - 1

    refdnoiterstd = list(refdfuncstd)
    
    alendnoiterin = list(alendfuncin)
    adimnoiterin = []
    arefdnoiterin = []
    for ivsdin,evsdin in enumerate(vsdin):
        adimnoiterin.append(list(adimfuncin[ivsdin]))
        arefdnoiterin.append(list(arefdfuncin[ivsdin]))
    
    alendnoiterout = list(alendfuncout)
    adimnoiterout = []
    arefdnoiterout = []
    for ivsdout,evsdout in enumerate(vsdout):
        adimnoiterout.append(list(adimfuncout[ivsdout]))
        arefdnoiterout.append(list(arefdfuncout[ivsdout]))

    #dnamselnoiter = list(dnamsel)
    
    # membytes: minimum total memory that will be used. We will the increase usage  when possible/allowed.
    membytes = 0
    for ivsdin,evsdin in enumerate(vsdin):
        membytes = membytes + alendfuncin[ivsdin] * vsdin[ivsdin]['itemsize']
    
    for ivsdout,evsdout in enumerate(vsdout):
        membytes = membytes + alendfuncout[ivsdout] * vsdout[ivsdout]['itemsize']
    
    
    if membytes > maxmembytes:
        print ('Warning, used memory ('+str(membytes)+') exceeds maximum memory ('+str(maxmembytes)+').')
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
                    # dselmarker
                    if vsdin[ivsdin]['dsel'][arefsin[ivsdin][idnam]] != False:
                        alendnoiterin_tmp[ivsdin] = alendnoiterin_tmp[ivsdin] *len(vsdin[ivsdin]['dsel'][arefsin[ivsdin][idnam]])
                    else:
                        alendnoiterin_tmp[ivsdin] = alendnoiterin_tmp[ivsdin] *vsdin[ivsdin]['dims'][arefsin[ivsdin][idnam]]
                for ivsdout,evsdout in enumerate(vsdout):
                    alendnoiterout_tmp[ivsdout] = alendnoiterout_tmp[ivsdout] *vsdout[ivsdout]['dims'][arefsout[ivsdout][idnam]]
    
                # recalculate the amount of bytes
                tmpmembytes = 0
                for ivsdin,evsdin in enumerate(vsdin):
                    tmpmembytes = tmpmembytes + alendnoiterin_tmp[ivsdin] * vsdin[ivsdin]['itemsize']
                
                for ivsdout,evsdout in enumerate(vsdout):
                    tmpmembytes = tmpmembytes + alendnoiterout_tmp[ivsdout] * vsdout[ivsdout]['itemsize']
    
                # if used memory still below threshold, we add it to the current dimension to the icecubes
                if tmpmembytes <= maxmembytes:
                    refdnoiterstd.insert(0,idnam)
                    for ivsdin,evsdin in enumerate(vsdin):
                        arefdnoiterin[ivsdin].insert(0, arefsin[ivsdin][idnam])
                        if vsdin[ivsdin]['dsel'][arefsin[ivsdin][idnam]] != False:
                            adimnoiterin[ivsdin].insert(0,vsdin[ivsdin]['dsel'][arefsin[ivsdin][idnam]])
                            alendnoiterin[ivsdin] = alendnoiterin[ivsdin] *len(vsdin[ivsdin]['dsel'][arefsin[ivsdin][idnam]])
                        else:
                            adimnoiterin[ivsdin].insert(0,range(vsdin[ivsdin]['dims'][arefsin[ivsdin][idnam]]))
                            alendnoiterin[ivsdin] = alendnoiterin[ivsdin] *vsdin[ivsdin]['dims'][arefsin[ivsdin][idnam]]
                    for ivsdout,evsdout in enumerate(vsdout):
                        arefdnoiterout[ivsdout].insert(0, arefsout[ivsdout][idnam])
                        adimnoiterout[ivsdout].insert(0,range(vsdout[ivsdout]['dims'][arefsout[ivsdout][idnam]]))
                        alendnoiterout[ivsdout] = alendnoiterout[ivsdout] *vsdout[ivsdout]['dims'][arefsout[ivsdout][idnam]]
                    #dnamselnoiter.insert(0,dnamsstd[idnam])
    
                    # recalculate the amount of bytes
                    membytes = 0
                    for ivsdin,evsdin in enumerate(vsdin):
                        membytes = membytes + alendnoiterin[ivsdin] * vsdin[ivsdin]['itemsize']
                    
                    for ivsdout,evsdout in enumerate(vsdout):
                        membytes = membytes + alendnoiterout[ivsdout] * vsdout[ivsdout]['itemsize']
    
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

    print 'dims in:',vsdin[ivsdin]['dims']
    print 'dnams in:',vsdin[ivsdin]['dnams']
    print 'dims out:',vsdout[ivsdin]['dims']
    print 'dnams out:',vsdout[ivsdin]['dnams']
    
    # guess from residual dimensions that are not in refnoiterin
    refditerstd = []
    dimiterstd = []
    for idim,edim in enumerate(dimsstd):
        if idim not in refdnoiterstd:
            refditerstd.append(idim)
            dimiterstd.append(edim) 
    if refditerstd == []:
        refditerstd = [-1]
        dimiterstd = [1]
    
    # guess from residual dimensions that are not in refnoiterin
    arefditerin = []
    adimiterin = []
    for ivsdin,evsdin in enumerate(vsdin):
        arefditerin.append([]) 
        adimiterin.append([]) 
        for idim,edim in enumerate(vsdin[ivsdin]['dims']):
            if idim not in arefdnoiterin[ivsdin]:
                arefditerin[ivsdin].append(idim)
                if vsdin[ivsdin]['dsel'][idim] != False:
                    adimiterin[ivsdin].append(vsdin[ivsdin]['dsel'][idim]) 
                else:
                    adimiterin[ivsdin].append(range(edim)) 
        if arefditerin[ivsdin] == []:
            arefditerin[ivsdin] = [-1]
            adimiterin[ivsdin] = [range(1)]
    
    # guess from residual dimensions that are not in refnoiterin
    arefditerout = []
    adimiterout = []
    for ivsdout,evsdout in enumerate(vsdout):
        arefditerout.append([]) 
        adimiterout.append([]) 
        for idim,edim in enumerate(vsdout[ivsdout]['dims']):
            if idim not in arefdnoiterout[ivsdout]:
                arefditerout[ivsdout].append(idim)
                if vsdout[ivsdout]['dsel'][idim] != False:
                    adimiterout[ivsdout].append(vsdout[ivsdout]['dsel'][idim]) 
                else:
                    adimiterout[ivsdout].append(range(edim)) 
        if arefditerout[ivsdout] == []:
            arefditerout[ivsdout] = [-1]
            adimiterout[ivsdout] = [range(1)]
    
    dimitermax = []
    for iref,eref in enumerate(refditerstd):
        dimitermax.append(1)
        for ivsdin,evsdin in enumerate(vsdin):
            dimitermax[iref] = max(dimitermax[iref],len(adimiterin[ivsdin][iref]))
        for ivsdout,evsdout in enumerate(vsdout):
            dimitermax[iref] = max(dimitermax[iref],len(adimiterout[ivsdout][iref]))
    
    rwchunksizein = [1]*len(vsdin)
    for ivsdin,evsdin in enumerate(vsdin):
        idim = len(vsdin[ivsdin]['dims'])-1
        while ((idim in arefdnoiterin[ivsdin]) & (idim >= 0) & (vsdin[ivsdin]['dsel'][idim] == False) & ((str(type(datin[ivsdin]['file']))[7:11]  != 'list') | (idim != 0))):
            # The inner dimensions just have to be referenced so not in correct order.  We know that they will be read in the correct order in the end
            rwchunksizein[ivsdin] = rwchunksizein[ivsdin]*vsdin[ivsdin]['dims'][idim]
            idim = idim - 1
    print "rwchunksizeout", rwchunksizein
    

    rwchunksizeout = [1]*len(vsdout)
    for ivsdout,evsdout in enumerate(vsdout):
        idim = len(vsdout[ivsdout]['dims']) -1
        while ((idim in arefdnoiterout[ivsdout]) & (idim >= 0)):
            # The inner dimensions just have to be referenced so not in correct order.  We know that they will be read in the correct order in the end
            rwchunksizeout[ivsdout] = rwchunksizeout[ivsdout]*vsdout[ivsdout]['dims'][idim]
            idim = idim - 1
    print "rwchunksizein",rwchunksizeout
    
    adimnoapplyout = []
    for ivsdout,evsdout in enumerate(vsdout):
        adimnoapplyout.append([])
        for irefdnoiterout in range(len(arefdnoiterout[ivsdout])-len(arefdfuncout[ivsdout])):
            adimnoapplyout[ivsdout].append(adimnoiterout[ivsdout][irefdnoiterout])
    
        if adimnoapplyout[ivsdout] == []:
            adimnoapplyout[ivsdout] = [range(1)]
    
    adimnoapplyin = []
    for ivsdin,evsdin in enumerate(vsdin):
        adimnoapplyin.append([])
        for irefdnoiterin in range(len(arefdnoiterin[ivsdin])-len(arefdfuncin[ivsdin])):
            adimnoapplyin[ivsdin].append(adimnoiterin[ivsdin][irefdnoiterin])
    
        if adimnoapplyin[ivsdin] == []:
            adimnoapplyin[ivsdin] = [range(1)] 

    dimnoapplymax = []
    for iref in range(len(arefdnoiterout[ivsdout])-len(arefdfuncout[ivsdout])):
        dimnoapplymax.append(1)
        for ivsdin,evsdin in enumerate(vsdin):
            dimnoapplymax[iref] = max(dimnoapplymax[iref],len(adimnoapplyin[ivsdin][iref]))
        for ivsdout,evsdout in enumerate(vsdout):
            dimnoapplymax[iref] = max(dimnoapplymax[iref],len(adimnoapplyout[ivsdout][iref]))
    if dimnoapplymax == []:
        dimnoapplymax = [1]
    
    lennoapplymax = reduce(mul,dimnoapplymax)
    
    lenitermax = reduce(mul,dimitermax)
    dimiterpos = [0]*len(dimitermax)
    sys.stdout.write(str(0)+'/'+str(lenitermax))
    for j in range(lenitermax):
        # reading icecube, rearranged in the order of dimensions specified by arefnoiterin
        dataicecubein = []
        for ivsdin,evsdin in enumerate(vsdin):
            # dataicecubein.append(np.zeros((elendnoiterin,),dtype=vsdin[ilendnoiterin]['dtype'])) 
            # if ((refiter == 0) & (type(vsdin[ivsdin]['fp']) == 'list'):
            #         dataicecubein.append(np.zeros(
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
        k = 0
        sys.stdout.write('    '+'('+str(0)+'/'+str(lennoapplymax)+')')
        for k in range(lennoapplymax):
            # actually, this is just the end of the file output already written
            ahunkin = []
            for ivsdin, evsdin in enumerate(vsdin):
                pos = 0
                # e.g. pos = (9)+ 20*(10) + 50*50*20*(5)
                for idimpos,edimpos in enumerate(dimnoapplypos):
                    curadd = np.mod(edimpos,len(adimnoapplyin[ivsdin][idimpos]))
                    #e.g. if edimpos == (5): curadd = 50*50*20*(5)
                    if ((idimpos + 1) < len(arefdnoiterin[ivsdin])):
                        for i in range(idimpos + 1,len(arefdnoiterin[ivsdin])) :
                            # here, we assume that the dimensions of the chunk are already in the order considered by adimsnoiter(out) etc. (cfr. preceeded transposition in readicecubeps)
                            curadd    = curadd    * len(adimnoiterin[ivsdin][i])
                            # curaddout = curaddout * dimnoiteroutref[i]
                    pos    = pos    + curadd
                ahunkin.append(dataicecubein[ivsdin][pos:(pos+alendfuncin[ivsdin])])
                ahunkin[ivsdin].shape = [len(e) for e in adimfuncin[ivsdin]]
    
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
                    curadd    = np.mod(edimpos,len(adimnoapplyout[ihunkout][idimpos]))
                    #e.g. if edimpos == (5): curadd = 50*50*20*(5)
                    if ((idimpos + 1) < len(arefdnoiterout[ihunkout])):
                        for i in range(idimpos + 1,len(arefdnoiterout[ihunkout])) :
                                                    # here, we assume that the idims are in the intended order (cfr. subsequent transposition in writeicecubeps)
                            curadd    = curadd    * len(adimnoiterout[ihunkout][i])
                            # curaddout = curaddout * dimnoiteroutref[i]
                    posout    = posout    + curadd
    
                dataicecubeout[ihunkout][posout:(posout+alendfuncout[ihunkout])] = np.array(ahunkout[ihunkout].ravel(),dtype=vsdout[ihunkout]['dtype'][1])
    
            # go to next data slice  
            dimnoapplypos[-1] = dimnoapplypos[-1] + 1
            for idimidx,edimidx in enumerate(reversed(dimnoapplypos)):
                # # alternative (makes 'dimiter' redundant)
                # if dimiterpos[idimidx] == shp[refiter[idimidx]]:
                if idimidx > 0:
                    if dimnoapplypos[idimidx] == dimnoapplymax[idimidx]:
                       dimnoapplypos[idimidx-1] = dimnoapplypos[idimidx-1] + 1
                       dimnoapplypos[idimidx] = 0
            sys.stdout.write ('\b'*(len('('+str(k)+'/'+str(lennoapplymax)+')')))
            sys.stdout.write ('('+str(k+1)+'/'+str(lennoapplymax)+')')
        # if lennoapplymax == 1:
        #     sys.stdout.write ('\b'*(len('('+str(k)+'/'+str(lennoapplymax)+')')))
        #     sys.stdout.write ('('+str(k+1)+'/'+str(lennoapplymax)+')')
        
        for idimsout in range(len(dataicecubeout)):
            dataicecubeout[idimsout].shape = [len(e) for e in adimnoiterout[idimsout]]
    
        for ivsdout in range(len(vsdout)):
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
    
        sys.stdout.write ('\b \b'*(len('('+str(k+1)+'/'+str(lennoapplymax)+')')))
        sys.stdout.write ('\b \b'*4)
        sys.stdout.write ('\b \b'*len(str(j)+'/'+str(lenitermax)))
        sys.stdout.write (str(j+1)+'/'+str(lenitermax))
    
    for ivsdin,evsdin in enumerate(vsdin):
        if str(type(vsdin[ivsdin]['fp']))[7:11] == 'file':
            vsdin[ivsdin]['fp'].close()
    for ivsdout,evsdout in enumerate(vsdout):
        vsdout[ivsdout]['fp'].close()
    print(' ')

import pylab as pl

