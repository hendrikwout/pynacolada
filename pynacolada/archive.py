import glob
import os
import pandas as pd
import xarray as xr

def empty_multiindex(names):
    """
    Creates empty MultiIndex from a list of level names.
    """
    return pd.MultiIndex.from_tuples(tuples=[(None,) * len(names)], names=names)

class archive (object):
    def __init__(self,path = None):
        self.lib_dataarrays = pd.DataFrame(index=empty_multiindex(['variable','source','time','space'])).iloc[1:]
        self.dataarrays = {}
        if path is not None:
            self.load(path)


    def copy(self):
        return self.archive.apply(lambda x: x.copy())

    def sel(self,sel):
        return self.archive.apply(lambda x: x.sel(sel))

    def sel_lib(self,sel):
        lib_dataarrays_out = self.lib_dataarrays_out[sel]
        archive_out = archive()
        for index,lib_dataarray in lib_dataarays_out.iterrows():
            archive_out.add_dataarray(self.dataarrays[index])

    def add_dataarray(self,DataArray_input,path=None):

        for key in self.lib_dataarrays.index.names:
            if key not in (list(DataArray_input.attrs.keys())+['variable']):
                raise ValueError(key+' needs to be in DataArray_input.attrs')
        dict_index = {}
        dict_columns = {}
        dict_index['variable'] = DataArray_input.name
        for key,value in DataArray_input.attrs.items():
           if key in self.lib_dataarrays.index.names:
               dict_index[key] = value
           else:
               dict_columns[key] = value
        index = tuple(dict_index[key] for key in self.lib_dataarrays.index.names) 
        for key,value in dict_columns.items():
            if key not in self.lib_dataarrays.columns:
                self.lib_dataarrays[key] = ''
            if index not in self.lib_dataarrays.index:
                self.lib_dataarrays.loc[index] = ''
            self.lib_dataarrays.loc[index][key] = value


        # if attrs is not None:
        #     for key,value in attrs.items():
        #         DataArray_input


        if path is not None:
            key = 'path'
            if key not in self.lib_dataarrays.columns:
                self.lib_dataarrays[key] = ''
            self.lib_dataarrays.loc[index]['path'] = path

        self.dataarrays[index] = DataArray_input
        self.lib_dataarrays.sort_index(inplace=True)

    #    def intersect_times(self,DataArray_input):
    def dataarrays_apply(self,function,query=None,inplace=False, attrs=None):
        # if query is not None:
        #     lib_dataarrays_out = self.lib_dataarrays.query(query,engine='python').copy()
        # else:
        #     lib_dataarrays_out = self.lib_dataarrays.copy()

        archive_out = archive()
        for index,columns in self.lib_dataarrays.iterrows():
            dataarray_out_temp = function(self.dataarrays[index])
            for key,value in self.dataarrays[index].attrs.items():
                dataarray_out_temp.attrs[key] = value
            if attrs is not None:
                for key,value in attrs.items():
                    dataarray_out_temp.attrs[key] = value
            archive_out.add_dataarray(dataarray_out_temp)
            
        return archive_out


    def dump(self,path ,query=None,force_overwrite = False):
        os.system('mkdir -p '+path)
        for index,dataarray in self.dataarrays.items():
            fnout = path+'/'+dataarray.name+'_'+dataarray.attrs['source']+'_'+dataarray.attrs['time']+'_'+dataarray.attrs['space']+'.nc'
            if (not force_overwrite) and (os.path.isfile(fnout)):
                raise OSError(fnout+' exists. Use force_overwrite to overwrite file')
            dataarray.to_netcdf(fnout);print('file written to: '+fnout)

            key = 'path'
            if key not in self.lib_dataarrays.columns:
                self.lib_dataarrays[key] = ''
            self.lib_dataarrays.loc[index]['path'] = path


    def load(self,path):
        os.system('mkdir -p '+path)
        filenames = glob.glob(path+'/*_*_*_*.nc')
        for filename in filenames:
            print('Opening file : '+filename)
            self.add_dataarray(xr.open_dataarray(filename),path=filename)


    # def dataarrays_merge_apply(self,attrs,function):
    #     if len(attrs.keys()) > 1:
    #         print('not tested, please check')
    #         import pdb; pdb.set_trace()
    #     
    #     index_keys_groupby = []
    #     index_keys_nongroupby = []
    #     for key in self.lib_dataarrays.index.names:
    #         if key in attrs.keys():
    #             index_keys_nongroupby.append(key)
    #         else:
    #             index_keys_groupby.append(key)
    #     lfirst = True

    #     archive_out = archive()
    #     for index_groupby,group_lib_dataarrays in  self.lib_dataarrays.groupby(index_keys_groupby):
    #         group_dataarrays = [self.dataarrays[key] for key in group_lib_dataarrays.index]
    #         dataarray_out = function(group_dataarrays)
    #         for column in group_lib_dataarrays.columns:
    #             # if lib_dataarrays[column].unique() > 0:
    #             # aparently uniform attributes are taken over, so we can detect from the destination attributies
    #             if (column not in dataarray_out.attrs.keys()) and (column != 'path'):
    #                 dataarray_out.coords[column] = (attrs.keys(), group_lib_dataarrays[column].values)
    #         for key,value in attrs.items():
    #             dataarray_out.attrs[key] = value
    #         archive_out.add_dataarray(dataarray_out)
    #     return archive_out




    def dataarrays_merge(self,attrs):
        if len(attrs.keys()) > 1:
            print('not tested, please check')
            import pdb; pdb.set_trace()
        
        index_keys_groupby = []
        index_keys_nongroupby = []
        for key in self.lib_dataarrays.index.names:
            if key in attrs.keys():
                index_keys_nongroupby.append(key)
            else:
                index_keys_groupby.append(key)
        lfirst = True

        archive_out = archive()
        for index_groupby,group_lib_dataarrays in  self.lib_dataarrays.groupby(index_keys_groupby):
            group_dataarrays = [self.dataarrays[key] for key in group_lib_dataarrays.index]
            dataarray_out = xr.concat(group_dataarrays,dim=group_lib_dataarrays.loc[index_groupby].index) 
            for column in group_lib_dataarrays.columns:
                # if lib_dataarrays[column].unique() > 0:
                # aparently uniform attributes are taken over, so we can detect from the destination attributies
                if (column not in dataarray_out.attrs.keys()) and (column != 'path'):
                    dataarray_out.coords[column] = (attrs.keys(), group_lib_dataarrays[column].values)
            for key,value in attrs.items():
                dataarray_out.attrs[key] = value
            archive_out.add_dataarray(dataarray_out)
        return archive_out
            # group_dataarrays = self.lib_dataarrays.loc[index_groupby]
            # xr.concat(group_dataarrays.values(),dim=group_dataarrays.index)

            # if lfirst:
            #     dataarray_out = self.dataarrays



    #    for key,value in merges.items():

    #        self.lib_dataarrays.groupby:


    # def merge_time_in_space(self,time,datetimes,space):
    #     archive_temp = self.dataarrays_apply(lambda x: x.reindex({'time':datetimes}),attrs={'time':time})

    #         archive_out = archive()
    #         import pdb; pdb.set_trace()


        


        # for key in 
        # self.lib_dataarrays

        # self.timeframes = pd.DataFrame(index=empty_multiindex(['timeframe','grid','source','variable']),columns=['model','experiment','ensemble_member','start','end','start_clip','end_clip']).iloc[1:]
        # self.timeframes_pages = pd.DataFrame(index = empty_multiindex(['timeframe','grid','source','variable','start','end']),columns = list(self.pages.columns)+['start_clip','end_clip']).iloc[1:]


    # def add_source(self,source = 'gsod_historical',pathfile = ''):
    #     if source == 'ghcn_historical':    
    #         lib_temp = pd.read_fwf(pathfile,names=['locid','lat','lon','variable','yearstart','yearend'])
    #         self.lookupvar[source] = {'pr_dailysum':'PRCP'}
    #         for var,sourcevar in self.lookupvar[source].items():
    #             
    #             lib_temp['variable'].loc[lib_temp['variable'] == sourcevar] = var    # = lib_temp.var.str.rename(columns={0:self.lookupvar[source][seriesvar]})
    #     else:
    #         raise ValueError ('Source '+source+' not supported.')
    #         
    #     lib_temp['source'] = source
    #     lib_temp = lib_temp.set_index(['locid','source','variable'])
    #     
    #     self.lib = pd.concat([self.lib,lib_temp])    


