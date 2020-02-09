import glob
import os
import pandas as pd
import xarray as xr
import numpy as np
import datetime as dt

def empty_multiindex(names):
    """
    Creates empty MultiIndex from a list of level names.
    """
    return pd.MultiIndex.from_tuples(tuples=[(None,) * len(names)], names=names)

class archive (object):
    def __init__(self,**kwargs):
        self.lib_dataarrays = pd.DataFrame(index=empty_multiindex(['variable','source','time','space'])).iloc[1:]
        self.dataarrays = {}
        self.coordinates = {}
        self.load(**kwargs)


    def copy(self):
        return self.archive.apply(lambda x: x.copy())

    def sel(self,sel):
        return self.archive.apply(lambda x: x.sel(sel))

    def sel_lib(self,sel):
        lib_dataarrays_out = self.lib_dataarrays_out[sel]
        archive_out = archive()
        for index,lib_dataarray in lib_dataarays_out.iterrows():
            archive_out.add_dataarray(self.dataarrays[index])

    def add_dataarray(self,DataArray_input,**kwargs):

        # for key in self.lib_dataarrays.index.names:
        #     if key not in (list(DataArray_input.attrs.keys())+['variable']):
        #         raise ValueError(key+' needs to be in DataArray_input.attrs')
        dict_index = {}
        dict_columns = {}
        dict_index['variable'] = DataArray_input.name
        for key,value in DataArray_input.attrs.items():
           if key in self.lib_dataarrays.index.names:
               dict_index[key] = value
           else:
               dict_columns[key] = value

        # for key in self.lib_dataarrays.index.names:
        #     if key in kwargs.keys():
        #         dict_index[key] = kwargs[key]
        for key,value in kwargs.items():
            if key in self.lib_dataarrays.index.names:
                dict_index[key] = kwargs[key]
            else:
                dict_columns[key] = kwargs[key]

        if 'time' not in dict_index.keys():
            print('Guessing time coordinate from DataArray')
            

            if not np.any(DataArray_input.time[1:].values - DataArray_input.time[:-1].values != np.array(86400000000000, dtype='timedelta64[ns]')):
                #daily
                dict_index['time'] = 'daily_'+np.datetime_as_string(DataArray_input.time[0].values,unit='D')+'_'+np.datetime_as_string(DataArray_input.time[-1].values,unit='D')
            elif not np.any(DataArray_input.time[1:].values - DataArray_input.time[:-1].values != dt.timedelta(days=1)):
                dict_index['time'] = \
                        'daily_'+str(DataArray_input.time[0].values)[:10]+'_'+str(DataArray_input.time[-1].values)[:10]
            else:
                import pdb;pdb.set_trace()
                raise ValueError('time dimension not implemented')

        renamings = {'lat':'latitude','lon':'longitude'}
        for key,value in renamings.items():
            if key in DataArray_input.dims:
                DataArray_input = DataArray_input.rename({key:value})

        # filter coordinates that are listed in the library index (these are not treated under space but separately, eg., 'time').
        space_coordinates = list(DataArray_input.dims)
        for key in self.lib_dataarrays.index.names:
            if key in space_coordinates:
                space_coordinates.remove(key)

        if 'space' not in dict_index.keys():
            
            
            spacing = {}
            for coordinate in space_coordinates:
                spacing_temp = (DataArray_input[coordinate].values[1] - DataArray_input[coordinate].values[0])
                if not np.any(DataArray_input[coordinate][1:].values != (DataArray_input[coordinate].values[:-1] + spacing_temp)):
                    spacing[coordinate] = spacing_temp
                else:
                    spacing[coordinate] = 'irregular'
            dict_index_space = [key+'-'+str(value) for key,value in spacing.items()]
            dict_index_space ='_'.join(dict_index_space) 
            dict_index['space'] = dict_index_space

        for key,index in dict_index.items():
            if key not in self.coordinates:
                if key not in ['variable','source','space',]:
                        self.coordinates[key] = DataArray_input[key]
                if key == 'space':
                    self.coordinates[key] = []
                    for coordinate in space_coordinates:
                        self.coordinates[key].append(DataArray_input[coordinate])

        for key in self.lib_dataarrays.index.names:
            if key not in dict_index.keys():
                raise ValueError ('Could not track key "'+key+'" that is required for the archive index.')

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

        # for key,value in kwargs.items():
        #     if key not in self.lib_dataarrays.columns:
        #         self.lib_dataarrays[key] = ''
        #     self.lib_dataarrays.loc[index][key] = value

        self.dataarrays[index] = DataArray_input
        self.lib_dataarrays.sort_index(inplace=True)


    #    def intersect_times(self,DataArray_input):
    def dataarrays_apply(self,function,query=None,inplace=False, attrs=None):
        # if query is not None:
        #     lib_dataarrays_out = self.lib_dataarrays.query(query,engine='python').copy()
        # elsepath
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

    def load(self,path=None,query=None,**kwargs):
        print(path)
        if type(path).__name__ == 'list':
            print('Guessing files from file list...')
            #allkwargs = {**dict(zip(lib_dataarrays_temp.index.names, index)),**dict(dataarray),**kwargs}
            filenames = path
            for filename in filenames:
                print('Opening file : '+filename)
                self.add_dataarray(xr.open_dataarray(filename),path=filename,**kwargs)

        elif os.path.isfile(path):
            print('pkl file '+path+' detected. Listing files from there.' )
            lib_dataarrays_temp = pd.read_pickle(path)
            if query is not None:
                print('performing query subselection: '+query)
                lib_dataarrays_temp = lib_dataarrays_temp.query(query,engine='python')
            for index,dataarray in lib_dataarrays_temp.iterrows():
                allkwargs = {**dict(zip(lib_dataarrays_temp.index.names, index)),**dict(dataarray),**kwargs}
                if dataarray.path[0] == '/':
                    absolute_file_path = dataarray.path
                else:
                    absolute_file_path = os.path.dirname(path)+'/'+dataarray.path
                allkwargs['absolute_path'] = absolute_file_path
                if os.path.isfile(absolute_file_path):
                    print('adding '+absolute_file_path+' with additional attributes: ',allkwargs)
                    try: 
                        xrin = xr.open_dataarray(absolute_file_path)
                    except ValueError:
                        # try to open the variable out of dataset
                        variable = index[0]
                        xrin = xr.open_dataset(absolute_file_path)[variable]
                    self.add_dataarray(xrin,**allkwargs)

                else:
                    print('Warning. Could not add file '+absolute_file_path+' because it does not exist.')

        elif os.path.isdir(path):
            print('Guessing files from directory...')
            filenames = glob.glob(path+'/*_*_*_*.nc')
            for filename in filenames:
                print('Opening file : '+filename)
                self.add_dataarray(xr.open_dataarray(filename),path=filename)
        #elif not os.path.isdir(path):
        else:
            raise IOError ('path '+path+ ' does not exist.')
#         os.system('mkdir -p '+path)



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


