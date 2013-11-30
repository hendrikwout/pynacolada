import numpy as np
import math as mt
import pylab as pl
import pynacolada as pcd
from sciproc import steinalp, dtrange
from Scientific.IO import NetCDF
import datetime as dt
import numpy as np

# purpose: 'advanced' example of making a Stein-Alpert decomposition from an monthly-mean averaged cycle (only on the 2 lowest model layers)

def avgcycle(x,n):
   xout = np.zeros([n+1]+list(np.shape(x)[1:]))
   for i in range(n):
      xout[i] = np.mean(x[i::n],axis=0)
   xout[n] = xout[0]
   return xout

dts = dtrange(dt.datetime.strptime('2009040100','%Y%m%d%H'),dt.datetime.strptime('2009050100','%Y%m%d%H'),dt.timedelta(hours=1))

for RUN in ['aq09ec_td','aq09_td','aq09ec','aq09']:
    path = '/media/URB_AQ_1/data/belgium_aq/rcm/'+RUN+'/stage1/aurora/au3d/'
    fins = [ path+'/au3d'+time.strftime('%Y%m%d%H')+'.nc' for time in dts]
    fout = NetCDF.NetCDFFile('/media/URB_AQ_1/data/belgium_aq/rcm/'+RUN+'_au3d200904_diucycle.nc','w')
    
    pcd.pcd(lambda x: avgcycle(x,24),('time',),\
            [{'file': fins, 'varname': 'O3', 'predim': 'time'}],\
            [{'file': fout, 'varname': 'O3'}],appenddim=True)
    
    fout.close(); print('data written to: ',fout)


    fin = NetCDF.NetCDFFile('/media/URB_AQ_1/data/belgium_aq/rcm/'+RUN+'_au3d200904_diucycle.nc','r')
    fout = NetCDF.NetCDFFile('/media/URB_AQ_1/data/belgium_aq/rcm/'+RUN+'_au3d200904_diucycle_spatvar.nc','w')
    pcd.pcd(lambda x: x - np.mean(x),('jy','kx'),\
            [{'file': fin, 'varname': 'O3'}],\
            [{'file': fout, 'varname': 'O3'}])
    fout.close(); print('data written to: ',fout)


fina = NetCDF.NetCDFFile('/media/URB_AQ_1/data/belgium_aq/rcm/aq09ec_td_au3d200904_diucycle_spatvar.nc','r')
finb = NetCDF.NetCDFFile('/media/URB_AQ_1/data/belgium_aq/rcm/aq09_td_au3d200904_diucycle_spatvar.nc','r')
finc = NetCDF.NetCDFFile('/media/URB_AQ_1/data/belgium_aq/rcm/aq09ec_au3d200904_diucycle_spatvar.nc','r')
find = NetCDF.NetCDFFile('/media/URB_AQ_1/data/belgium_aq/rcm/aq09_au3d200904_diucycle_spatvar.nc','r')

fouta = NetCDF.NetCDFFile('/media/URB_AQ_1/data/belgium_aq/rcm/aq09/au3d200904_a.nc','w')
foutb = NetCDF.NetCDFFile('/media/URB_AQ_1/data/belgium_aq/rcm/aq09/au3d200904_b.nc','w')
foutc = NetCDF.NetCDFFile('/media/URB_AQ_1/data/belgium_aq/rcm/aq09/au3d200904_c.nc','w')
foutd = NetCDF.NetCDFFile('/media/URB_AQ_1/data/belgium_aq/rcm/aq09/au3d200904_d.nc','w')
 
for evar in fina.variables :
    #print evar
    if evar not in fina.dimensions:
        pcd.pcd(lambda a,b,c,d: steinalp([a,b,c,d],2),\
            [],\
         [  {'file':fina , 'varname':evar, 'dsel':{'iz' : [0,1]}},\
            {'file':finb , 'varname':evar, 'dsel':{'iz' : [0,1]}},\
            {'file':finc , 'varname':evar, 'dsel':{'iz' : [0,1]}},\
            {'file':find , 'varname':evar, 'dsel':{'iz' : [0,1]}},\
         ] ,\
         [  {'file':fouta , 'varname':evar},\
            {'file':foutb , 'varname':evar},\
            {'file':foutc , 'varname':evar},\
            {'file':foutd , 'varname':evar},\
         ]\
            ,appenddim = True)
fouta.close(); print fouta
foutb.close(); print foutb
foutc.close(); print foutc
foutd.close(); print foutd
# 
# # f = np.array([[[1.,2.],[4. ,3.]],
# #              [[1.,4.],[5. ,7.]]])
# 
# 
# 
# # from Scientific.IO import NetCDF
# # import numpy as np
# # import sciproc as sp
# # import pynacolada as pcd
# # import datetime as dt
# # 
# # dts = sp.dtrange(dt.datetime(2009,5,1),dt.datetime(2009,7,1),dt.timedelta(hours=1))
# # path = '/media/URB_AQ_1/data/belgium_aq/rcm/aq09/stage1/aurometin/'
# # dtfiles = [path+'auro'+dt.datetime.strftime(e,"%Y%m%d%H")+'.nc' for e in dts]
# # fout = NetCDF.NetCDFFile('/media/URB_AQ_1/data/belgium_aq/rcm/aq09/stage1/aurometin/ana.nc','w')
# # for evar in ['T','BLH']:
# #     datin =  [{'file': dtfiles, \
# #                'varname': evar, \
# #                'dsel': {'nz' : range(0,3,1)}, \
# #               },\
# #              ]
# #     datout =  [{'file': fout, \
# #                'varname': evar, \
# #               },\
# #              ]
# #     dnamsel = ['nz',]
# #     def func(x):
# #     	return np.array([np.mean(x,axis=0)])
# #     
# #     pcd.pcd(func,dnamsel,datin,datout,appenddim=True)
# # 
# # pcd.ncwritedatetime(fout,dts,tunits='hours', refdat=dt.datetime(2009,1,1))
# # fout.close()




