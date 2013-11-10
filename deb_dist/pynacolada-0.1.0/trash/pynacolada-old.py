import numpy as np



datain = open('/media/hendrik/developrma/projects/N78E7/data/antwerp_project/rcm/aq09/stage1/aurorabc/hour16_beleuros.bin','r')
dataout = open('/media/hendrik/developrma/projects/N78E7/data/antwerp_project/rcm/aq09/stage1/aurorabc/hour16_beleurosT.bin','r')



# def binT (datain,dataout,dimin,perm)



instrmidx = 

dataout.seek()

# permutation 
permlst = (3,1,2,4,5)

# tells on which (permuted) indices we want to iterate, and on which dimensions we want to operate
applyon = (False,True,True,False,False)



# amount of execution steps, multiplication of dimensions over which we want to iterate
lenexec = 1
# dimensions in new order over which we want to iterate
dimexec = []
for irefdim,refdim in enumerate(permlist):
    if applyon[irefdim] == True:
        permlst = permlst*dim[refdim]
        dimexec.append(dim[refdim])

curdimidx = [0]*len(dimexec)

for i in range(permlst):

    seeek .... blablbla


    curdimidx[-1] = curidx[-1] + 1
    for idimidx,edimidix in reversed(enumerate(cudimidx)):
        if curdimidx[idimidx] == dimexe

















