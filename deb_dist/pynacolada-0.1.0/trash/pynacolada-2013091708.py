import numpy as np
import pickle
from numpy import fromfile
import pylab as pl
from operator import itemgetter

def shake(fin,shp,dimapplyref,fout,dimiterref=None,maxicecubesize=10000):

    """
    purpose 
    -------

    swap specified dimensions to the back efficiently in a specified order

    input parameters
    ----------------

    fin: binary file input stream
    fout: binary file output stream
    shp: shape of the filestream
    dimapplyref: dimensions over which the function is applied
    dimnoiterref: reference to dimensions that are swapped to the back. Data in these dimensions are treated as icecubes. The order of those indices are taken into account
    dimiterref (optional): reference to dimensions that are swapped to the front. The order of those indices are
    taken into account.  Of not specified, it is guessed from the residual dimensions (defined in shp) that are not in dimnoiterref
    """

    lenapplyref = 1

    lennoiter = 1
    for edimapplyref in dimapplyref:
        dimapplyref.append(edimapplyref)
        dimapply.append(shp[edimapplyref])
        lenapply = lenapply*shp[edimapplyref]

        dimnoiterref.append(edimapplyref)
        dimnoiter.append(shp[edimapplyref])
        lennoiter = lennoiter*shp[edimapplyref]

    if lenapply > maxicecubesize:
        print 'Warning, the function data input length of'+lenapply+' (dimensions: ',dimapply,') exceeds the maximum icecubesize of'+str(maxicecubesize)+'.' 
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

    dimiterpos = [0]*len(dimiter)


    shpout = []
    for edimiterref in dimiterref:
        shpout.append(shp[edimiterref])

    for edimnoiterref in dimnoiterref:
        shpout.append(shp[edimnoiterref])
    # # or
    # for ishp,eshp in enumerate(shp):
    #     if ishp not in dimiterref: 
    #         shpout.append(shp[i])

    for j in range(leniter):

        # actually, this is just the end of the file output already written
        fpos = 0
        # e.g. fpos = (9)+ 20*(10) + 50*50*20*(5)
        for idimpos,edimpos in enumerate(dimiterpos):
            curadd = edimpos
            #e.g. if edimpos == (5): curadd = 50*50*20*(5)
            if ((idimpos + 1) < len(shpout)):
                for i in range(idimpos + 1,len(shpout)) :
                    curadd = curadd * shpout[i]
        
            fpos = fpos + curadd

        # drop data to file in reordered way
        fout.seek(4*fpos)
        
        # reading icecube, rearranged in the order of dimensions specified by dimnoiterref
        dataicecube = np.array(readicecubeps(fin,shp,dimiterref,dimiterpos,dimnoiterref),dtype='float32').ravel()
        # crush the ice

        # dimnoiterref = (6 ,7 ,8 ,4 ,5)
        # dimiter      = (30,20,15,20,15)
        # dimapplyref  =       (8 ,4 ,5)


        lenapply = long(1)
        dimapply = []
        for idimeapplyref in range(len(dimapplyref)):
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

        dimnoapplypos = [0]*len(dimnoapply)


        for k in range(lennoapply):

            # actually, this is just the end of the file output already written
            apos    = 0
            aposout = 0
            # e.g. apos = (9)+ 20*(10) + 50*50*20*(5)
            for idimpos,edimpos in enumerate(dimnoapplypos):
                curadd    = edimpos
                curaddout = edimpos
                #e.g. if edimpos == (5): curadd = 50*50*20*(5)
                if ((idimpos + 1) < len(dimnoiterref)):
                    for i in range(idimpos + 1,len(dimnoiterref)) :
                        curadd    = curadd    * dimnoiterref[i]
                        curaddout = curaddout * dimnoiteroutref[i]
            
                apos    = apos    + curadd
                aposout = aposout + curaddout

            hunk = dataicecube[apos:(apos+lenapply)]
            hunk.shape = dimapply

            icecubeout[aposout:(aposout+lenapplyout)] = function(hunk).ravel()

            # go to next data slice  
            dimapplypos[-1] = dimapplypos[-1] + 1
            for idimidx,edimidx in enumerate(reversed(dimapplypos)):
                # # alternative (makes 'dimiter' redundant)
                # if dimiterpos[idimidx] == shp[dimiterref[idimidx]]:
                if dimapplypos[idimidx] == dimapply[idimidx]:
                    if idimidx > 0:
                        dimapplypos[idimidx-1] = dimapplypos[idimidx-1] + 1
                        dimapplypos[idimidx] = 0

        
        # reading icecube, rearranged in the order of dimensions specified by dimnoiterref
        dataicecubeout = np.array(readicecubeps(fin,shp,dimiterref,dimiterpos,dimnoiterref,mode='write'),dtype='float32').ravel()
        # crush the ice

        # go to next data slice  
        dimiterpos[-1] = dimiterpos[-1] + 1
        for idimidx,edimidx in enumerate(reversed(dimiterpos)):
            # # alternative (makes 'dimiter' redundant)
            # if dimiterpos[idimidx] == shp[dimiterref[idimidx]]:
            if dimiterpos[idimidx] == dimiter[idimidx]:
                if idimidx > 0:
                    dimiterpos[idimidx-1] = dimiterpos[idimidx-1] + 1
                    dimiterpos[idimidx] = 0
    print leniter
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


# print readicecube(fin,(1,35,52),(5,),(3,))
def readicecube(filestream,shp,dimiterref,dimpos,dimnoiterref=None):
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
    # not really needed for application, but we implement it
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
        if ((dimiterref[idimpos] + 1) < len(shp)):
            for i in range(dimiterref[idimpos] + 1,len(shp)) :
                curadd = curadd * shp[i]
    
        fpos = fpos + curadd
    # print fpos,dimnoiterref,lennoiter
    
    
    # e.g. dimnoiterref = (0,2)
    #      dimnoiterpos = (5,20)
    #      j = based on (0,2) and (5,20)
    
    
    # create icecube array
    icecube = np.zeros((lennoiter,))*np.nan
    dimnoiterpos = [0]*len(dimnoiter)
    # print icecube,dimnoiterpos
    for j in range(lennoiter):
        fposicecube = fpos
        for idimpos,edimpos in enumerate(dimnoiterpos):
            curadd = edimpos
            # e.g. fposicecube = (1)*52
            # e.g. fposicecube = (9)+ 20*(10) + 50*50*20*(5)
            if ((dimnoiterref[idimpos] + 1) < len(shp)):
                for i in range(dimnoiterref[idimpos] + 1,len(shp)) :
                    curadd = curadd * shp[i]
    
            fposicecube = fposicecube + curadd
            # print j, idimpos,edimpos,fposicecube
    
        filestream.seek(4*fposicecube)
        icecube[j] = fromfile(filestream,dtype='float32',count=1)
        #print 'reading icecube with length / position: ', fposicecube,'/',1,icecube[j]
        # print j, dimnoiterpos,fposicecube,j == fposicecube,icecube[j]
    
        # go to next data strip 
        if dimnoiterpos != []:
            dimnoiterpos[-1] = dimnoiterpos[-1] + 1
            for idimidx,edimidx in enumerate(reversed(dimnoiterpos)):
                if dimnoiterpos[idimidx] == dimnoiter[idimidx]:
                    if idimidx > 0:
                        dimnoiterpos[idimidx-1] = dimnoiterpos[idimidx-1] + 1
                        dimnoiterpos[idimidx] = 0
    
    icecube.shape = dimnoiter
    
    return icecube

def readicecubeps(fin,shp,dimiterref,dimiterpos,dimnoiterref,mode='read'):
    """ 
    read an icecube and perform an in-memory Post Swap of dimensions (very fast)
    hereby, we acquire the order of the icecube dimensions
    """
    icecube =readicecube(fin,shp,dimiterref,dimiterpos) 
    # print 'shape',icecube,icecube.shape
    if mode=='read':
        # print icecube.shape,zip(*sorted(zip(dimnoiterref,range(len(dimnoiterref))),key=itemgetter(0,1)))
        if dimnoiterref == None:
            return icecube
        else:
            return np.transpose(icecube,zip(*sorted(zip(dimnoiterref,range(len(dimnoiterref))),key=itemgetter(0,1)))[1])

shp = (4,35,52)
dimnoiterref = (2,1)

fig = pl.figure()
fin = open('/home/hendrik/data/belgium_aq/rcm/aq09/stage1/aurorabc/hour16_beleuros.bin','r')
fout = open('/home/hendrik/data/belgium_aq/rcm/aq09/stage1/aurorabc/hour16_beleuros2.bin','wb')
# def readicecube(filestream,shp,dimiterref,dimpos,dimnoiterref=None):
testdat = readicecubeps(      fin,       shp,(1,),    (2,),dimnoiterref=(1,0))

pl.imshow(testdat)
fig.show()
fout.close()
fin.close()






