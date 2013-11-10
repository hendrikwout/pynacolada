import numpy as np
from numpy import fromfile
import pylab as pl
fin = open('/home/hendrik/data/belgium_aq/rcm/aq09/stage1/aurorabc/hour16_beleuros.bin','r')
fout = open('/home/hendrik/data/belgium_aq/rcm/aq09/stage1/aurorabc/hour16_beleurosT.bin','w')
# 
# 
# 
# # def binT (datain,dataout,dimin,perm)
# 
# 
# 
# instrmidx = 
# 
# dataout.seek()
# 
# 
# # tells on which (permuted) indices we want to iterate, and on which dimensions we want to operate
# applyon = (False,True,True,False,False)
# dim = (50,60,70)
# dimout = []
# 
# # amount of iteration steps, multiplication of dimensions over which we want to iterate
# leniter = 1
# lennoiter = 1
# # dimensions in new order over which we want to iterate
# dimiter = []
# dimnoiter = []
# dimiterref = []
# for idim,edim in enumerate(dims):
#     if applyon[irefdim] == True:
#         leniter = leniter*dim[idim]
#         dimiter.append(dim[idim])
#         dimiterref.append(idim)
#     else:
#         lennoiter = lennoiter*dim[idim]
#         dimnoiter.append(dim[idim])
#         dimnoiterref.append(idim)
# 
# 
# curdimiteridx = [0]*len(dimiter)
# for i in range(leniter):
#     fpos = 0
#     for idimidx,edimidx in enumerate(curdimiteridx):
#         curadd = edimidx
#         # get position
#         for i in range(1,dimiterref[idimidx]):
#             curadd = curadd * dim[i]
# 
#         fpos = fpos + curadd
# 
# 
#     outchunk = func(chunk)
#     outchunk.shape
# 
#     fposout = fpos
#     # write output chunk to file
#     dimnoiterref = []
#     curdimnoiteridx = [0]*len(dimnoiter)
#     for j in range(lennoiter):
#         curdimnoiter = [0]*len(dimnoiter)
#         for idim,edim in enumerate(dimnoiter):
#             curadd = edim
#             for i in range(1,dimnoiterref[idimidx]):
#                 curadd = curadd * dim[i]
# 
#             fposout = fposout + curadd
# 
#         fout.seek(fposout)
#         tofile(fout,dtype='float32',count=1, outchunk[j]  )
# 
#         curdimnoiteridx[-1] = curdimnoiteridx[-1] + 1
#         for idimidx,edimidx in reversed(enumerate(curdimnoiteridx)):
#             if curdimiteridx[idimidx] == dimnoiter[idimidx]:
#                 if idimidx > 0:
#                     curdimnoiteridx[idimidx-1] = curdimnoiteridx[idimidx-1] + 1
# 
#     curdimiteridx[-1] = curdimiteridx[-1] + 1
#     for idimidx,edimidx in reversed(enumerate(curdimiteridx)):
#         if curdimiteridx[idimidx] == dimiter[idimidx]:
#             if idimidx > 0:
#                 curdimiteridx[idimidx-1] = curdimiteridx[idimidx-1] + 1
# 
# 
# dimiter = []
# dimnoiter = []
# dimiterref = []
# for idim,edim in enumerate(dims):
#     if applyon[irefdim] == True:
#         leniter = leniter*dim[idim]
#         dimiter.append(dim[idim])
#         dimiterref.append(idim)
#     else:
#         lennoiter = lennoiter*dim[idim]
#         dimnoiter.append(dim[idim])
#         dimnoiterref.append(idim)
# 
# 
# applyon = (False,True,True,False,False)
dim = (50,60,70)

lennoiter = 1
# amount of iteration steps, multiplication of dimensions over which we want to iterate
leniter = 1



filestream = fin
shp = (1,35,52)
dimiterref = (0,)
dimpos = (0,)
# print binchunk(fin,(1,35,52),(5,),(3,))
# def binchunk(filestream,shp,dimiterref,dimpos):
"""
read data chunk from binary data and put it in an array
filestream: binary file reference
shp: shape of the filestream
dimiterref: reference to dimensions over which no slice is performed
pos: position of the non-sliced dimensions
"""

# e.g. shp = (200,100,50,50,20)
#      dimiterref = (1,3,4)
#      dimpos = (5,10,9)

# extend so that structured arrays are read at once


dimiter = []
dimnoiter = []
lennoiter = long(1)
dimnoiterref = []
for i in range(len(shp)):
    if i in dimiterref:
        dimiter.append(shp[i])
    else:
        dimnoiterref.append(i)
        dimnoiter.append(shp[i])
        lennoiter = lennoiter*shp[i]




fpos = 0
# e.g. fpos = (9)+ 20*(10) + 50*50*20*(5)
for idimpos,edimpos in enumerate(dimpos):
    curadd = edimpos
    #e.g. if edimpos == (5): curadd = 50*50*20*(5)
    if ((dimiterref[idimpos] + 1) < len(shp)):
        for i in range(dimiterref[idimpos] + 1,len(shp)) :
            curadd = curadd * shp[i]

    fpos = fpos + curadd
print fpos


# e.g. dimnoiterref = (0,2)
#      dimnoiterpos = (5,20)
#      j = based on (0,2) and (5,20)


# create chunk array
chunk = np.zeros((lennoiter,))*np.nan
dimnoiterpos = [0]*len(dimnoiter)
for j in range(lennoiter):
    fposchunk = fpos
    for idimpos,edimpos in enumerate(dimnoiterpos):
        curadd = edimpos
        # e.g. fposchunk = (1)*52
        # e.g. fposchunk = (9)+ 20*(10) + 50*50*20*(5)
        if ((dimnoiterref[idimpos] + 1) < len(shp)):
            for i in range(dimnoiterref[idimpos] + 1,len(shp)) :
                curadd = curadd * shp[i]

        fposchunk = fposchunk + curadd
        # print j, idimpos,edimpos,fposchunk

    filestream.seek(4*fposchunk)
    chunk[j] = fromfile(filestream,dtype='float32',count=3)[0]
    print j, dimnoiterpos,fposchunk,j == fposchunk,chunk[j]

    # go to next data strip 
    dimnoiterpos[-1] = dimnoiterpos[-1] + 1
    for idimidx,edimidx in enumerate(reversed(dimnoiterpos)):
        if dimnoiterpos[idimidx] == dimnoiter[idimidx]:
            if idimidx > 0:
                dimnoiterpos[idimidx-1] = dimnoiterpos[idimidx-1] + 1
                dimnoiterpos[idimidx] = 0

chunk.shape = dimnoiter


# return chunk



