import glob
import os
import pandas as pd
import xarray as xr
import numpy as np
import datetime as dt
import itertools
import yaml
import logging
import sys
from tqdm import tqdm
import tempfile
from . import apply_func, nc_reduce_fn
import netCDF4 as nc4

def parse_grid_mapping(ds,DataArray):
    if ('grid_mapping' in DataArray.attrs) and (DataArray.attrs['grid_mapping'] not in [None, np.nan]) and not ((type(DataArray.attrs['grid_mapping']) in (float,np.float64)) and np.isnan(DataArray.attrs['grid_mapping'])):
        grid_mapping_type = DataArray.attrs['grid_mapping']
        logging.debug('grid_mapping ('+str(grid_mapping_type)+') detected. Reading grid_mapping type from separate character variable, which appears the standard according to qgis')
        for crsattr in ds[grid_mapping_type].attrs:
            logging.debug('reading '+crsattr+' ('+str(ds[grid_mapping_type].attrs[crsattr])+') and add it as '+grid_mapping_type+'_'+crsattr+' to the regular xarray attributes')
            DataArray.attrs[grid_mapping_type+'_'+crsattr] =  ds[grid_mapping_type].attrs[crsattr]

def parse_to_dataframe(list_or_dict_or_dataframe):
    if type(list_or_dict_or_dataframe) == pd.core.frame.DataFrame:
        return list_or_dict_or_dataframe
    elif type(list_or_dict_or_dataframe) == list:
        return pd.DataFrame(list_or_dict_or_dataframe)
    elif type(list_or_dict_or_dataframe) == dict:
        return pd.DataFrame.from_dict([dict(zip(list_or_dict_or_dataframe.keys(),v)) for v in itertools.product(*list_or_dict_or_dataframe.values())])


# test ;lkjdfg

def empty_multiindex(names):
    """
    Creates empty MultiIndex from a list of level names.
    """
    return pd.MultiIndex.from_tuples(tuples=[(None,) * len(names)], names=names)

def apply_func_wrapper(
    func,
    lib_dataarrays,
    dataarrays,
    archive_out,
    xarray_function_wrapper=apply_func,
    dataarrays_wrapper = lambda *x: (*x,),
    groupby=None,
    apply_groups_in=[],
    apply_groups_out=[],
    groups_for_index_level = [],
    divide_into_groups_extra = [],
    mode = 'numpy_output_to_disk_in_chunks',
    inherit_attributes = False,
    query=None,
    extra_attributes={},
    post_apply=None,
    force_recalculate=False,
    engine=None,
    update_pickle=True,
    delay = 10,
    nc_reduce=False,
    input_cache_to_ram=True,
    drop_duplicates_in_group = True,
    #lib_dataarrays = self.lib_dataarrays

    **kwargs,
):

    """
    purpose: this wrapper routine allows to apply a function in automatic groups of
    dataarrays in an archive or collection of archive, and dumps the output to
    a specified output archive.

    input:

    output:
    """
    apply_groups_in_df_temp = parse_to_dataframe(apply_groups_in)
    if drop_duplicates_in_group:
        apply_groups_in_df = apply_groups_in_df_temp.drop_duplicates()
        if len(apply_groups_in_df_temp) != len(apply_groups_in_df):
            logging.warning('duplicate source names found. Duplicates were removed')
    else:
        apply_groups_in_df = apply_groups_in_df_temp
    apply_groups_out_df = parse_to_dataframe(apply_groups_out)

    divide_into_groups = []
    for name in lib_dataarrays.index.names:
        if (name not in apply_groups_in_df.columns) and (name not in groups_for_index_level):
            divide_into_groups.append(name)
    for name in divide_into_groups_extra:
        if name not in divide_into_groups:
            divide_into_groups.append(name)

    if query is not None:
        if type(query) == str:

            read_lib_dataarrays = lib_dataarrays.query(query).copy()
        elif type(query) == pd.DataFrame:
            read_lib_dataarrays = query
        else:
            raise ValueError('type of input query ' + query + 'not implemented')
    else:
        read_lib_dataarrays = lib_dataarrays.copy()


    if len(divide_into_groups) == 0:
        print('creating dummy group that encompasses the whole library')
        divide_into_groups = ['dummy_group']
        read_lib_dataarrays['dummy_group'] = ""
    groups_in_loop = read_lib_dataarrays.reset_index().groupby(divide_into_groups)
    logging.info('Looping over data array input groups: '+ str([groupkeyvalues[0] for groupkeyvalues in groups_in_loop]) )
    for idx, group in tqdm(groups_in_loop,position=0):

        def always_tuple(idx):
            if type(idx) is not tuple:
                return (idx,)
            else:
                return idx


        def always_multi_index(index):

            if type(index) == pd.core.indexes.base.Index:
                return pd.MultiIndex.from_tuples([x.split()[::-1] for x in index], names=(index.name,))
            else:
                return index


        apply_this_group_in_df = apply_groups_in_df.copy()

        #for idx_group_in, row in apply_this_group_in_df.iterrows():
        for idx_group_in, row in enumerate(apply_this_group_in_df.to_dict('records')):
            for column in apply_this_group_in_df.columns:
                if row[column] is None:
                    # import pdb;pdb.set_trace()
                    apply_this_group_in_df[column].loc[idx_group_in] = group.loc[:, column].unique()

        for column in apply_this_group_in_df.columns:
            apply_this_group_in_df = apply_this_group_in_df.explode(column)

        multi_idx = always_tuple(idx)

        check_group_columns = []
        for column in apply_this_group_in_df.columns:
            if (~apply_this_group_in_df[column].isnull()).any():
                check_group_columns.append(column)

        group_reduced = group[check_group_columns].drop_duplicates().reset_index().drop(columns='index')
        apply_this_group_in_df_reduced = apply_this_group_in_df[check_group_columns].drop_duplicates().reset_index().drop(
            columns='index')

        if len(apply_groups_in_df) == 0:
            all_input_available_in_this_group = True
        else:
            apply_this_group_in_reduced_index = pd.MultiIndex.from_frame(apply_this_group_in_df_reduced)
            group_reduced_indexed_for_apply_groups_in = group.set_index(list(apply_this_group_in_df_reduced.columns))
            group_reduced_indexed_for_apply_groups_in.index = always_multi_index(
                group_reduced_indexed_for_apply_groups_in.index)

            all_input_available_in_this_group = not (len(apply_this_group_in_reduced_index) != len(
                apply_this_group_in_reduced_index.intersection(group_reduced_indexed_for_apply_groups_in.index)))

        # group_indexed_for_apply_groups_in = group.set_index(list(apply_this_group_in_df.columns))

        #
        #
        #
        # apply_this_group_in_index = pd.MultiIndex.from_frame(apply_this_group_in_df)
        # group_indexed_for_apply_groups_in.index = always_multi_index(apply_this_group_in_df.index)
        #
        # stop
        #
        # #group_selection_index = pd.MultiIndex.from_frame(frame_for_selecting_xarrays_from_current_group)
        #
        # all_input_available_in_this_group = not (group_indexed_for_apply_groups_in.index != apply_groups_in_index).any()
        #

        if all_input_available_in_this_group:

            if len(apply_this_group_in_df) == 0:
                table_this_group_in = group
            else:
                apply_this_group_in_index = pd.MultiIndex.from_frame(apply_this_group_in_df)
                group_indexed_for_apply_groups_in = group.set_index(list(apply_this_group_in_df.columns))
                group_indexed_for_apply_groups_in.index = always_multi_index(group_indexed_for_apply_groups_in.index)

                # apply_this_group_in_df = apply_groups_in_df.copy()
                # for idx_group_in,row in apply_this_group_in_df.iterrows():
                #     for column in apply_this_group_in_df.columns:
                #         if row[column] is None:
                #             apply_this_group_in_df[column].loc[idx_group_in] = group.loc[:,column].unique()

                # import pdb; pdb.set_trace()
                # apply_this_group_in_index = apply_this_group_in_index.intersection(group_indexed_for_apply_groups_in.index,sort=None)
                # import pdb; pdb.set_trace()
                table_this_group_in = group_indexed_for_apply_groups_in.loc[apply_this_group_in_index]

            # apply_groups_out_index = pd.MultiIndex.from_frame(apply_groups_out_df)
            # group_indexed_for_apply_groups_out = group.set_index(list(apply_groups_out_df.columns))
            # group_indexed_for_apply_groups_out.index = always_multi_index(group_indexed_for_apply_groups_out.index)

            apply_groups_out_df_this_group = apply_groups_out_df.copy()

            logging.info('converting label functions where necessary')
            table_this_group_out = apply_groups_out_df_this_group.copy()
            for idx_group_out, row in enumerate(apply_groups_out_df.to_dict('records')):
                for key, value in row.items():
                    if type(apply_groups_out_df_this_group.loc[idx_group_out, key]).__name__ == 'function':
                        table_this_group_out.loc[idx_group_out, key] = apply_groups_out_df_this_group.loc[
                            idx_group_out, key](*tuple(table_this_group_in.reset_index()[[key]].values[:, 0]))
                    # elif table_this_group_out.loc[idx_group_out,key] is None:
                    #     raise ValueError('Not supported yet')
                    else:
                        table_this_group_out.loc[idx_group_out, key] = apply_groups_out_df.loc[idx_group_out, key]


            # ??????
            # for dataarray in dataarrays_group_in:
            #     dataarray.close()
            #     del dataarray



            # building attributes of output dataarrays
            ifile = 0
            attributes_dataarrays_out = []
            attributes_dataarrays_in_for_out = []
            filenames_out_pattern = []
            dataarrays_out_already_available = []

            
            # for idx_group_in, row in enumerate(table_this_group_in.reset_index().to_dict('records')):
            #     attributes_dataarray_in = dict(row)
            #     attributes_dataarray_in = {key:attributes_dataarray_in[key] for key in attributes_dataarray_in.keys() if key not in [ 'path', 'available', 'linked', 'path_pickle']}

            #     attributes_dataarrays_in.append(attributes_dataarray_in)

            for idx_group_out, row in enumerate(table_this_group_out.to_dict('records')):
                attributes_dataarrays_in_for_out.append({})

                logging.info('start determining attributes of output files')

                if table_this_group_in.index.names[0] is None:  # trivial case where no group_in selection is made
                    attributes_in = \
                        dict(zip(table_this_group_in.columns,
                                 table_this_group_in.iloc[min(ifile, len(table_this_group_in) - 1)]))

                else:
                    attributes_in = \
                        {
                            **dict(zip(table_this_group_in.index.names,
                                       table_this_group_in.iloc[min(ifile, len(table_this_group_in) - 1)].name)),
                            **dict(zip(table_this_group_in.columns,
                                       table_this_group_in.iloc[min(ifile, len(table_this_group_in) - 1)]))
                        }

                for key, value in attributes_in.items():

                    if (key not in attributes_dataarrays_in_for_out[ifile]) and \
                            (not ((inherit_attributes == False) or (   (type(inherit_attributes) == list)    and  (    key not in inherit_attributes)))) and \
                            (key not in ['absolute_path_as_cache', 'absolute_path_for_reading', 'absolute_path',
                                         'path','available','path_pickle']):
                        attributes_dataarrays_in_for_out[ifile][key] = value

                attributes_dataarrays_out.append({})
                for key in row.keys():
                    attributes_dataarrays_out[ifile][key] = row[key]



                # !!
                for key in lib_dataarrays.index.names:
                    if key in table_this_group_out.columns:
                        attributes_dataarrays_out[ifile][key] = row[key]
                    elif key in table_this_group_in.index.names:
                        attributes_dataarrays_in_for_out[ifile][key] = \
                            table_this_group_in.iloc[min(ifile, len(table_this_group_in) - 1)].name[
                                table_this_group_in.index.names.index(key)]
                    else:
                        attributes_dataarrays_in_for_out[ifile][key] = \
                            table_this_group_in.iloc[min(ifile, len(table_this_group_in) - 1)][key]

                    # for key in self.lib_dataarrays.columns:
                    #     if key == 'provider':
                    #         import pdb; pdb.set_trace()

                    #     if (key not in attributes_dataarrays_out[ifile]) and (inherit_attributes or (key in inherit_attributes)) and (key not in ['absolute_path','path']):
                    #         attributes_dataarrays_out[ifile][key] = {**dict(zip(table_this_group_in.index.names,table_this_group_in.iloc[min(ifile,len(dataarrays_group_in)-1)])),**dict(zip(table_this_group_in.columns,table_this_group_in.iloc[min(ifile,len(dataarrays_group_in)-1)]))}[key]

                for key, value in extra_attributes.items():
                    attributes_dataarrays_out[ifile][key] = value


                # filter coordinates that are listed in the library index (these are not treated under space but separately, eg., 'time').
                if ('space' not in table_this_group_out.columns) and ('output_dims' in kwargs.keys()):
                    logging.info('update space attribute according to the new dimensions specified in "output_dims"')

                    attributes_space_dict_out = {}
                    if attributes_dataarrays_out[ifile]['space'] is not None:
                        for space_part in attributes_dataarrays_in_for_out[ifile]['space'].split('_'):
                            if ':' not in space_part:
                                space_key,space_value = space_part, None
                            else:
                                space_key,space_value = space_part.split(':')
                            attributes_space_dict_out[space_key] = space_value

                    output_dims = kwargs['output_dims']

                    space_coordinates = list(output_dims.keys())
                    for key in lib_dataarrays.index.names:
                        if key in space_coordinates:
                            space_coordinates.remove(key)

                    for coordinate in space_coordinates:
                        spacing_temp = (output_dims[coordinate][1] - output_dims[coordinate][0])
                        if not np.any(
                                output_dims[coordinate][1:] != (output_dims[coordinate][:-1] + spacing_temp)):
                            attributes_space_dict_out[coordinate] = str(output_dims[coordinate][0]) + ',' + str(
                                output_dims[coordinate][-1]) + ',' + str(spacing_temp)
                        else:
                            attributes_space_dict_out[coordinate] = 'irregular'

                    dict_index_space = []
                    #dict_index_space = [key + ':' + str(value) for key, value in spacing.items()]
                    for key,value in attributes_space_dict_out.items():
                        if value is None:
                            dict_index_space.append(key)
                        else:
                            dict_index_space.append(key+':'+str(value))

                    attributes_dataarrays_in_for_out[ifile]['space'] = '_'.join(dict_index_space)

                        # DataArray.attrs['space'] = dict_index_space

                attributes_dataarrays_all = []
                for iattrs,attrs_dataarray_in_for_out in enumerate(attributes_dataarrays_in_for_out):
                    attributes_dataarrays_all.append(attrs_dataarray_in_for_out.copy())
                    for key,value in attributes_dataarrays_out[iattrs].items():
                        attributes_dataarrays_all[iattrs][key] = value

                index_keys = ['variable','source','time','space']
                index_out = []
                for key in index_keys:
                    index_out.append(attributes_dataarrays_all[ifile][key])

                logging.info('end determining attributes of output files')

                logging.info('check current intended array is already available in the output')
                dataarrays_out_already_available.append(
                        (len(archive_out.lib_dataarrays) > 0) and
                        (tuple(index_out) in archive_out.lib_dataarrays.index)
                )

                logging.info('Start determining output filenames')
                if mode in ['numpy_output_to_disk_in_chunks', 'numpy_output_to_disk_no_chunks']:
                    if (archive_out.file_pattern is None):
                        raise ValueError("I don't know how to write the data file to disk. Please set to file_pattern")

                    # filename_out = os.path.dirname(archive_out.path_pickle) + '/' + ''.join(np.array(list(
                    #     zip(archive_out.file_pattern.split('"')[::2],
                    #         [attributes_dataarrays_out[ifile][key] for key in archive_out.file_pattern.split('"')[1::2]] + [
                    #             '']))).ravel())

                    filename_out_pattern = os.path.dirname(archive_out.path_pickle)+'/'+archive_out.file_pattern
                    filenames_out_pattern.append(filename_out_pattern)

                ifile += 1
                logging.info('End determining output filenames')
            logging.info('check whether any array is available in the output. If yes, then do not calculate.')
            all_dataarrays_out_already_available = np.prod(dataarrays_out_already_available)
            some_dataarrays_out_already_available = (np.sum(dataarrays_out_already_available) > 0)

            if all_dataarrays_out_already_available and not force_recalculate:
                logging.info('All output data is already available in the output archive and force_recalculate is switched False. Skipping group "'+str(idx)+'"')
            else:

                dataarrays_group_in = []
                dataarrays_group_in_cached = []

                for idx_group_in, row in table_this_group_in.iterrows():
                    if table_this_group_in.index.names[0] is None:  # trivial case where no group_in selection is made
                        index_dataarray = [dict(zip(table_this_group_in.columns, row))[key] for key in
                                           lib_dataarrays.index.names]
                    else:
                        index_dataarray = [{**dict(zip(table_this_group_in.index.names, idx_group_in)),
                                            **dict(zip(table_this_group_in.columns, row))}[key] for key in
                                           lib_dataarrays.index.names]

                    row_of_dataarray = lib_dataarrays.loc[tuple(index_dataarray)]


                    logging.debug('opening dataarray for: '+str(row_of_dataarray))
                    if tuple(index_dataarray) in dataarrays.keys():
                        dataarrays_group_in.append(dataarrays[tuple(index_dataarray)])
                    else:
                        filename = os.path.dirname(row_of_dataarray.path_pickle) + '/' + row_of_dataarray.path
                        # import pdb; pdb.set_trace()

                        # import netCDF4 as nc4
                        # ds = nc4.Dataset(filename)
                        # var = ds.variables['bar']
                        # print('complevel: %s', var.filters().get('complevel', False))


                        #if ("nc_compressed" in row_of_dataarray) and (row_of_dataarray["nc_compressed"] == 'True'):
                        if input_cache_to_ram == True:
                            filename_work = tempfile.mktemp(suffix='.nc',dir='/tmp/')
                            #os.system('ncks -L 0 '+filename+' '+filename_work)
                            os.system('cp '+filename+' '+filename_work)
                            #from nc_reduce import restore
                            #restore(filename, filename_work)
                            
                            cached = filename_work
                        else:
                            filename_work = filename
                            cached = False

                        try:
                            dataarrays_group_in.append( xr.open_dataarray(filename_work, engine=engine))
                        except:
                            try:
                                dataarrays_group_in.append( xr.open_dataarray(filename_work))
                            except:
                                try:
                                    dataarrays_group_in.append(xr.open_dataset(filename_work, engine=engine)[row_of_dataarray.ncvariable])
                                except:
                                    dataarrays_group_in.append(xr.open_dataset(filename_work)[row_of_dataarray.ncvariable])
                        dataarrays_group_in_cached.append(cached)

                    attributes_from_table = {**dict(zip(['variable','source','time','space'],index_dataarray,)),**dict(row)}

                    logging.debug('linked dataarray detected. Overriding xarray attributes with the those supplemented in the pandas table')
                    for key in attributes_from_table.keys():
                        if key not in ['path','available','linked','ncvariable','path_pickle','linked','nc_reduced','nc_compresed']:
                            dataarrays_group_in[-1].attrs[key] = attributes_from_table[key]


                if force_recalculate and some_dataarrays_out_already_available:
                    logging.info('some output dataarrays were available but force_recalulate is set True, so I force recaculation'
                                 'and overwrite output files')
                    kwargs['overwrite_output_filenames'] = True

                if mode == 'xarray':
                    print('making a temporary dataarray copy to prevent data hanging around into memory afterwards')
                    dataarrays_group_in_copy = [dataarray.copy(deep=False) for dataarray in dataarrays_group_in]
                    temp_dataarrays = func(*dataarrays_wrapper(*tuple(dataarrays_group_in_copy)))
                    # temp_dataarrays = func(*dataarrays_wrapper(*tuple(dataarrays_group_in)))

                    for idataarray, dataarray in enumerate(temp_dataarrays):
                        for key, value in attributes_dataarrays_out[idataarray].items():
                            if key == 'variable':
                                dataarray.name = value
                            else:
                                dataarray.attrs[key] = value
                        archive_out.add_dataarray_old(dataarray)  # attributes_dataarrays_out[idataarray])
                    for idataarray in range(len(dataarrays_group_in_copy)):
                        dataarrays_group_in_copy[idataarray].close()
                    for itemp_dataarray in range(len(temp_dataarrays)):
                        temp_dataarrays[itemp_dataarray].close()

                elif mode in ['numpy_output_to_disk_in_chunks', 'numpy_output_to_disk_no_chunks']:
                    logging.info('starting apply_func')
                    if mode == 'numpy_output_to_disk_in_chunks':

                        filenames_out = xarray_function_wrapper(func, dataarrays_wrapper(*tuple(dataarrays_group_in)),
                                xarrays_output_filenames=filenames_out_pattern,
                                attributes_dataarrays_in_for_out=attributes_dataarrays_in_for_out,  #inherit of attributes of the input files to the output files 
                                attributes_dataarrays_out=attributes_dataarrays_out, # attribute dataarrays that should override whatever happens by the apply_func
                                return_type='paths',
                                delay=delay,
                                nc_reduce=nc_reduce,**kwargs)
                    elif mode == 'numpy_output_to_disk_no_chunks':
                        temp_dataarrays = xarray_function_wrapper(func, dataarrays_wrapper(*tuple(dataarrays_group_in)),
                                                                  **kwargs)
                        if type(temp_dataarrays) != tuple:
                            print(
                                'this is a workaround in case we get a single dataarray instead of tuple of dataarrays from the wrapper function. This needs revision')
                            idataarray = 0
                            for key, value in attributes_dataarrays_out[idataarray].items():
                                if key not in archive_out.not_dataarray_attributes:
                                    if type(value) == bool:
                                        temp_dataarrays.attrs[key] = int(value)
                                    else:
                                        temp_dataarrays.attrs[key] = value
                                if key == 'variable':
                                    temp_dataarrays.name = value
                            # import pdb;pdb.set_trace()
                            os.system('rm ' + filenames_out[idataarray])
                            if post_apply is not None:
                                post_apply(temp_dataarrays)
                            os.system('mkdir -p ' + os.path.dirname(filenames_out[idataarray]))
                            temp_dataarrays.to_netcdf(filenames_out[idataarray])
                            temp_dataarrays.close()
                        else:
                            for idataarray in range(len(temp_dataarrays)):
                                for key, value in attributes_dataarrays_out[idataarray].items():
                                    if key not in ['variable', 'absolute_path_for_reading', 'absolute_path_as_cache',
                                                   'absolute_path', 'path','path_pickle']:
                                        temp_dataarrays[idataarray].attrs[key] = value
                                    if key == 'variable':
                                        temp_dataarrays[idataarray].name = value

                            for idataarray in range(len(temp_dataarrays)):
                                if post_apply is not None:
                                    post_apply(temp_dataarrays[idataarray])
                                os.system('rm ' + filenames_out[idataarray])
                                os.system('mkdir -p ' + os.path.dirname(filenames_out[idataarray]))
                                temp_dataarrays[idataarray].to_netcdf(filenames_out[idataarray])
                                temp_dataarrays[idataarray].close()

                    logging.info('ending apply_func')

                    for ixr_out, filename_out in enumerate(filenames_out):
                        logging.info('add_dataarray start')

                        archive_out.add_dataarray(filename_out,method='add')
                        logging.info('add_dataarray end')
                else:
                    ValueError('mode ' + mode + ' not implemented')
                for idataarray in reversed(range(len(dataarrays_group_in))):
                    dataarrays_group_in[idataarray].close()
                    del dataarrays_group_in[idataarray]
                for dataarray_cached in dataarrays_group_in_cached:
                    if dataarray_cached != False: 
                        os.system('rm '+dataarray_cached)

                # if update_pickle:
                #     logging.info('update archive_out start')
                #     archive_out.update(force_overwrite_pickle =True)
                #     logging.info('update archive_out end')


class collection (object):
    def __init__(self,archives,*args,**kwargs):
        self.archives =  archives

    def get_path(self,index):
        path = os.path.dirname(self.get_lib_dataarrays().loc[tuple(index)].path_pickle) + '/' + self.get_lib_dataarrays().loc[tuple(index)].path
        return path

    def get_dataarray(self,index,engine=None):
        path = self.get_path(index)
        try:
            xropen = xr.open_dataarray(path,engine=engine)
        except:
            xropen = xr.open_dataset(path,engine=engine)[self.get_lib_dataarrays().loc[tuple(index)].ncvariable]
        if ('linked' in self.get_lib_dataarrays().columns) and (self.get_lib_dataarrays().loc[tuple(index)].linked == True):
            logging.debug('linked dataarray detected. Overriding xarray attributes with the those supplemented in the pandas table')
            for key in self.get_lib_dataarrays().loc[tuple(index)].keys():
                if key not in ['path','available','linked','ncvariable','path_pickle','linked']:
                    xropen.attrs[key] = self.get_lib_dataarrays().loc[tuple(index)][key]


        return xropen

    def get_lib_dataarrays(self,with_full_paths = False):
        logging.info('Build common library from collection of archives')
        if len(self.archives) > 0:
            lib_dataarrays = pd.concat([archive.lib_dataarrays for archive in self.archives]).sort_index()
        else:
            lib_dataarrays = pd.DataFrame()


        if with_full_paths == True:
            lib_dataarrays['path_full'] = lib_dataarrays.apply(lambda row: ('/'.join(row['path_pickle'].split('/')[:-1])+'/'+row['path']),axis=1).values

        return lib_dataarrays

    def get_dataarrays(self):
        logging.info('Start build common dataarray pool from collection of archives')
        dataarrays = {}
        for archive in self.archives:
            dataarrays = {**dataarrays, **archive.dataarrays}
        logging.info('end build common dataarray pool from collection of archives')
        return dataarrays


    

    def apply_func(
            self,
            func,
            archive_out = None,
            add_archive_out_to_collection=False,
            update_pickle=True,
            file_pattern=None,
            nc_reduce=False,
            transpose_inner = True,
            **kwargs
    ):

        if type(archive_out) is str:
            if file_pattern != None:
                archive_out = archive(archive_out,file_pattern=file_pattern)
            else:
                archive_out = archive(archive_out)
            write_mode = 'create_new_archive'
        elif archive_out is not None: # type is considered an archive object
            write_mode = 'add_to_external_archive'
        else:
            write_mode = 'add_to_current_archive'

        lib_dataarrays = self.get_lib_dataarrays()
        dataarrays = self.get_dataarrays()
        apply_func_wrapper(
            func,
            lib_dataarrays = lib_dataarrays,
            dataarrays = dataarrays,
            archive_out=archive_out,
            update_pickle=update_pickle,
            nc_reduce=nc_reduce,
            transpose_inner=transpose_inner,
            **kwargs
        )

        if add_archive_out_to_collection and (archive_out not in self.archives):
            self.archives.append(archive_out)

        if write_mode == 'create_new_archive':
            return archive_out


class archive (object):
    def get_dataarray(self,index,engine=None):
        path = self.get_path(index)
        try:
            xropen = xr.open_dataarray(path,engine=engine)
            dataarrays_group_in.append( xr.open_dataarray(filename, engine=engine))
        except:
            xropen = xr.open_dataset(path,engine=engine)[self.lib_dataarrays.loc[tuple(index)].ncvariable]
        if ('linked' in self.lib_dataarrays.columns) and (self.lib_dataarrays.loc[tuple(index)].linked == True):
            logging.debug('linked dataarray detected. Overriding xarray attributes with the those supplemented in the pandas table')
            for key in self.lib_dataarrays.loc[tuple(index)].keys():
                if key not in ['path','available','linked','ncvariable','path_pickle','linked']:
                    xropen.attrs[key] = self.lib_dataarrays.loc[tuple(index)][key]


        return xropen

    def remove_orphans(self):
        read_lib_dataarrays = self.lib_dataarrays.copy()
        for idx,row in read_lib_dataarrays.iterrows():
            orphan = os.path.isfile(self.get_path(idx)) == False
            if orphan:
                logging.warning(str(idx)+' ("'+self.get_path(idx)+'") does not exist. Removing from library file')
                logging.warning('please test and check this procedure!')
        #self.remove(query = "available == False",dataarrays=False,reset_lib=True)
                self.lib_dataarrays = self.lib_dataarrays.drop(idx)
        self.update(force_overwrite_pickle =True)


    def get_path(self,index):
        path = os.path.dirname(self.lib_dataarrays.loc[tuple(index)].path_pickle) + '/' + self.lib_dataarrays.loc[tuple(index)].path
        return path

    def __init__(
            self,
            path_pickle=None,
            file_pattern='"variable"_"source"_"time"_"space".nc',
            reset = False,
            debug = False,
            query_dict = None,
            *args,
            **kwargs):

        self.settings_keys = ['file_pattern']
        print('Creating generic functions to set attributes')
        for key in self.settings_keys:
            print('creating function self.set_'+key)
            self.__dict__['set_'+key] = lambda value: self.__setattr__(key,value)
        print('Loading default settings')
        self.file_pattern = file_pattern
        self.not_dataarray_attributes = ['ncvariable', 'path','path_pickle','linked','available','linked' ]

        self.dataarrays = {}
        self.coordinates = {}


        if os.path.isfile(path_pickle):
            self.lib_dataarrays = pd.read_pickle(path_pickle)
            self.set_path_pickle(path_pickle)

            if query_dict is not None:

                def dict_query(df1,method='match',**filter_v):
                    index_names = df1.index.names
                    df1_reset = df1.reset_index()
                    filter_v_reduced = {key: value for key,value in filter_v.items() if key in df1_reset.columns}

                    if method == 'match':
                        return df1_reset.loc[(df1_reset[list(filter_v_reduced)] == pd.Series(filter_v_reduced)).all(axis=1)].set_index(index_names)
                    elif method == 'isin':

                        query = ' & '.join([key+'.isin('+str(value.split(','))+')' for key,value in filter_v_reduced.items()])
                        if query != '' :
                            return df1_reset.query(query,engine='python').set_index(index_names),filter_v_reduced
                        else: 
                            return df1_reset.set_index(index_names),filter_v_reduced

                self.lib_dataarrays = dict_query(self.lib_dataarrays,method='isin',**query_dict)[0]





        if reset == True:
            self.remove(reset_lib=True)
            os.system('rm '+path_pickle)

        if not os.path.isfile(path_pickle):
            self.lib_dataarrays = pd.DataFrame(
                index=empty_multiindex(
                    ['variable', 'source', 'time', 'space']
                ),
                columns=['path', 'available']).iloc[1:]
            self.set_path_pickle(path_pickle)
        if debug == True:
            import pdb; pdb.set_trace()
        # if path is not None:
        #     self.load(path,*args,**kwargs)
    def set_path_pickle(self,path_pickle):
        self.path_pickle = path_pickle
        self.lib_dataarrays['path_pickle'] = path_pickle


    def copy(self):
        return self.archive.apply(lambda x: x.copy())

    def sel(self,sel):
        return self.archive.apply(lambda x: x.sel(sel))

    def sel_lib(self,sel):
        lib_dataarrays_out = self.lib_dataarrays_out[sel]
        archive_out = archive()
        for index,lib_dataarray in enumerate(lib_dataarays_out.to_dict('records')):
            archive_out.add_dataarray_old(self.dataarrays[index])



    def nc_reduce(self,query=None,compress=True):

        if query is not None:
            read_lib_dataarrays = self.lib_dataarrays.query(query).copy()
        else:
            read_lib_dataarrays = self.lib_dataarrays.copy()
        for idx,row in read_lib_dataarrays.iterrows():
            if (('nc_reduced' not in read_lib_dataarrays) or row['nc_reduced'] != "True") or ('nc_compressed' not in read_lib_dataarrays) or row['nc_compressed'] != "True":
                filepath_as_cache = tempfile.mktemp(suffix='.nc',dir='/tmp/')
                CMD ='cp '+self.get_path(idx) +' '+filepath_as_cache
                logging.info('executing: '+CMD);os.system(CMD)
                dict_row = dict(row)
                if 'nc_reduce' in dict_row:
                    dict_row.pop('nc_reduce') #work around early implementation
                logging.info('trying to reduce: '+str(idx)+' ("'+self.get_path(idx)+'")')

                #compress = (nc_compress if nc_compress is not None else True)
                reduced_temporary_file = tempfile.mktemp(suffix='.nc',dir='/tmp/')
                nc_reduce_fn(os.path.realpath(filepath_as_cache),reduced_temporary_file,ncvariable = (dict_row['ncvariable'] if 'ncvariable' in row else None),overwrite=True,nc_reduce=(row['nc_reduced'] != "True"),nc_compress=compress)
                CMD ='rm '+filepath_as_cache
                logging.info('executing: '+CMD);os.system(CMD)

                dict_row['nc_compressed']=str(compress)
                dict_row['nc_reduced']='True'
                self.add_dataarray(reduced_temporary_file,destination_file=self.get_path(idx),method='copy_force_overwrite',variable=idx[0],source=idx[1],time=idx[2],space=idx[3],**dict_row)
                CMD ='rm '+reduced_temporary_file
                logging.info('executing: '+CMD);os.system(CMD)



    def remove(self,query=None,update_pickle = True,dataarrays=True,reset_lib=False):

        if (not dataarrays) and reset_lib:
            raise ValueError('the dataarrays on the disk is maintained by this archive.'
            'Not removing them while removing them from the database will lead to orphaned files. Aborting... ')

        if query is not None:
            read_lib_dataarrays = self.lib_dataarrays.query(query).copy()
        else:
            read_lib_dataarrays = self.lib_dataarrays.copy()
        for idx,row in read_lib_dataarrays.iterrows():
            if dataarrays == True:
                if ('linked' not in row) or (row['linked'] == False):
                    CMD ='rm '+os.path.dirname(os.path.realpath(row['path_pickle'])) + '/' + str(row['path'])
                    os.system(CMD)
                # if 'available' not in self.lib_dataarrays.columns:
                #     self.lib_dataarrays['available'] = ""
                #     self.lib_dataarrays['available'] = True

            if idx in self.lib_dataarrays.index:
                if reset_lib:
                    self.lib_dataarrays = self.lib_dataarrays.drop(idx)
                else:
                    self.lib_dataarrays.loc[idx]['available'] = False
        if update_pickle:
            self.update(force_overwrite_pickle =True)

    def remove_old(self,query=None,update_pickle = True,dataarrays=True,records=False):

        if (not dataarrays) and records:
            raise ValueError('the dataarrays on the disk is maintained by this archive.'
            'Not removing them while removing them from the database will lead to orphaned files. Aborting... ')

        if query is not None:
            read_lib_dataarrays = self.lib_dataarrays.query(query).copy()
        else:
            read_lib_dataarrays = self.lib_dataarrays.copy()
        for idx,row in read_lib_dataarrays.iterrows():
            if dataarrays:
                CMD ='rm '+row['absolute_path']
                os.system(CMD)
                # if 'available' not in self.lib_dataarrays.columns:
                #     self.lib_dataarrays['available'] = ""
                #     self.lib_dataarrays['available'] = True

            if self.lib_dataarrays.index.contains(idx):
                if records:
                    self.lib_dataarrays = self.lib_dataarrays.drop(idx)
                else:
                    self.lib_dataarrays.loc[idx]['available'] = False
        if update_pickle:
            self.update(force_overwrite_pickle =True)

    def remove_by_index(self,index,delete_on_disk=False,update_pickle=True):
        #self.dataarrays[index].close()
        #del self.dataarrays[index]
        if delete_on_disk:
            os.system('rm '+os.path.dirname(self.lib_dataarrays.loc[index].path_pickle)+'/'+self.lib_dataarrays.loc[index].path)
        if ('absolute_path' in self.lib_dataarrays.columns) and (self.lib_dataarrays.loc[index].absolute_path_as_cache is not None):
            print(self.lib_dataarrays.loc[index].absolute_path_as_cache)
            print(np.isnan(self.lib_dataarrays.loc[index].absolute_path_as_cache))
            print(np.isnan(self.lib_dataarrays.loc[index].absolute_path_as_cache) == True)
            if (np.isnan(self.lib_dataarrays.loc[index].absolute_path_as_cache == False)) :
                CMD = 'rm '+self.lib_dataarrays.loc[index].absolute_path_as_cache
                print('removing cached file:',CMD)

        self.lib_dataarrays.drop(index=index,inplace=True)


        if update_pickle:
            self.update(force_overwrite_pickle=True)


    def close(self,delete_archive=False):
        lib_dataarrays_temp = self.lib_dataarrays.copy()
        for index,columns in lib_dataarrays_temp.iterrows():
            self.remove_by_index(index=index,delete_on_disk=(delete_archive==True))

        del lib_dataarrays_temp
        if delete_archive:
            os.system('rm '+self.path_pickle)

    def add_from_dataset(self,Dataset_or_filepath,variables=None,**kwargs):
        if type(Dataset_or_filepath).__name__ == 'str':
            Dataset = xr.open_dataset(Dataset_or_filepath,engine=engine)
            kwargs['absolute_path'] = os.path.abspath(Dataset_or_filepath)
            kwargs['absolute_for_reading'] = kwargs['absolute_path']
        else:
            Dataset = Dataset_or_filepath

        if variables is None:
            variables= []#Dataset.variables
            for variable in Dataset.variables:
                print(Dataset.dims)
                if variable not in Dataset.dims:
                    variables.append(variable)
        for variable in variables:
            self.add_dataarray_old(Dataset[variable],**kwargs)

        Dataset.close()

    def add_dataarray(
            self,
            DataArray_or_filepath,
            destination_file = None,
            skip_unavailable= False,
            release_dataarray_pointer=False,
            method = 'link',
            cache_to_ram=False,
            reset_space=False,
            sort_lib = True,
            update_lib = True,
            engine = None,
            nc_reduce = False,
            nc_compress = None,
            **kwargs,
    ):
        #DataArray = None

        logging.info('Reading xarray')
        dict_index = {}
        dict_columns = {}

        if type(DataArray_or_filepath).__name__ == 'str':
            filepath = DataArray_or_filepath
            filepath_as_cache = None
            filepath_for_reading = filepath
            if cache_to_ram:
                CMD='cat '+filepath_for_reading+' > /dev/null'
                print('caching to ram:',CMD)
                os.system(CMD)

            # ncvariable: variable as seen on disk
            # variable (= DataArray.name): variable as considered in the library
            ncvariable = None # netcdf variable as seen on disk, not necessarily in the library
            if ('ncvariable' not in kwargs.keys()) or ((type(kwargs['ncvariable']).__name__ == 'float') and np.isnan(kwargs['ncvariable'])):
                logging.info('Opening file:'+filepath_for_reading+ '(original file: '+filepath+')')
                if 'variable' in kwargs.keys():
                        # print('reading',filepath,ncvariable)
                    # try:
                    #     print('trying with open_dataarray first because of obscure performance decreases with xr.open_dataset')
                    #     DataArray = xr.open_dataarray(filepath)
                    #     if DataArray.name != kwargs['ncvariable']:
                    #         DataArray.close()
                    #         del DataArray
                    #         print('first variable is not the correct ncvariable. Trying it with xr.open_dataset...')
                    #         raise ValueError('first variable is not the correct ncvariable')
                    # except:

                    try:
                        ds = xr.open_dataset(filepath_for_reading,engine=engine)
                        DataArray = ds[kwargs['variable']]
                        parse_grid_mapping(ds,DataArray)
                        ds.close()
                        del ds
                        #kwargs['ncvariable'] = ncvariable
                    except:
                        DataArray = xr.open_dataarray(filepath_for_reading,engine=engine)
                else:
                    ds = xr.open_dataset(filepath_for_reading,engine=engine)
                    variables = list(ds.variables)
                    for var in ['crs']+list(ds.coords):
                         if var in variables:
                             variables.pop(variables.index(var))
                    if len(variables) == 1:
                        DataArray = ds[variables[0]]
                        parse_grid_mapping(ds,DataArray)
                    else:
                        import pdb; pdb.set_trace()
                        raise ValueError('multiple dataset variables detected. It should have only one')
                    ds.close()
                    del ds
                    #kwargs['ncvariable'] = ncvariable


            else:
                ds = xr.open_dataset(filepath_for_reading,engine=engine)
                DataArray = ds[kwargs['ncvariable']]
                parse_grid_mapping(ds,DataArray)
                ds.close()
                del ds


            # kwargs['absolute_path'] = os.path.abspath(filepath)
            # kwargs['absolute_path_for_reading'] = os.path.abspath(filepath_for_reading)
            # kwargs['absolute_path_as_cache'] = (None if filepath_as_cache is None else os.path.abspath(filepath_as_cache))
            # kwargs['available'] = True
        elif type(DataArray_or_filepath) is xr.DataArray:
            DataArray = DataArray_or_filepath
            dict_columns['path'] = None
        else:
            raise IOError('Input type '+type(DataArray_or_filepath).__name__+' not supported')

        logging.info('Acquiring xarray attributes')
        for key,value in DataArray.attrs.items():
            if key in self.lib_dataarrays.index.names:
                dict_index[key] = value
            else:
                dict_columns[key] = value

        dict_columns['ncvariable'] = DataArray.name

        logging.info('Acquiring attributes from input arguments')
        for key,value in kwargs.items():
            if key in self.lib_dataarrays.index.names:
                dict_index[key] = kwargs[key]
            else:
                dict_columns[key] = kwargs[key]

        if 'variable' not in dict_index.keys():
            dict_index['variable'] = dict_columns['ncvariable']
        #TODO !!! merge this procedure with get_coordinates_attributes in apply_func.py
        logging.info('Creating attributes from dimension data.')
        if ('time' not in dict_index.keys()) or (dict_index['time'] is None) or (type(dict_index['time']).__name__ == 'float') and (  np.isnan(dict_index['time']).any()):
            print('Guessing time coordinate from DataArray')
            # is month type

            #monthly spacing
            if np.apply_along_axis(lambda y: np.sum((y[1:] - y[:-1] != 1),0),0,np.vectorize(lambda x: int(x[:4])*12+int(x[5:7]))(DataArray.time.values.astype('str'))).item() == 0:
                dict_index['time'] = \
                    'monthly_'+str(DataArray.time[0].values)[:7]+'_'+str(DataArray.time[-1].values)[:7]
            # also monthly
            elif (not np.any( ~(np.vectorize(lambda x: x[8:])(DataArray.time.values.astype('str')) == '01T00:00:00.000000000'))):
                dict_index['time'] = \
                    'monthly_'+str(DataArray.time[0].values)[:7]+'_'+str(DataArray.time[-1].values)[:7]
            elif not np.any((DataArray.time[2:-1].values - DataArray.time[1:-2].values) != np.array(86400000000000, dtype='timedelta64[ns]')):
                #daily
                dict_index['time'] = 'daily_'+np.datetime_as_string(DataArray.time[0].values,unit='D')+'_'+np.datetime_as_string(DataArray.time[-1].values,unit='D')
            elif not np.any((DataArray.time[2:-1].values - DataArray.time[1:-2].values) != dt.timedelta(days=1)):
                dict_index['time'] = \
                    'daily_'+str(DataArray.time[0].values)[:10]+'_'+str(DataArray.time[-1].values)[:10]
            else:
                dict_index['time'] = 'irregular'
                logging.warning('Warning. No time dimension found')

            #DataArray.attrs['time'] = dict_index['time']

        renamings = {'lat':'latitude','lon':'longitude'}
        for key,value in renamings.items():
            if key in DataArray.dims:
                DataArray = DataArray.rename({key:value})

        # filter coordinates that are listed in the library index (these are not treated under space but separately, eg., 'time').
        space_coordinates = list(DataArray.dims)
        for key in self.lib_dataarrays.index.names:
            if key in space_coordinates:
                space_coordinates.remove(key)


        if ('space' not in dict_index.keys()) or (dict_index['space'] is None) or reset_space:
            space_coordinates_only = ['latitude','longitude']
            spacing = {}
            for coordinate in space_coordinates:
                if coordinate in space_coordinates_only:
                    spacing_temp = (DataArray[coordinate].values[1] - DataArray[coordinate].values[0])
                    if not np.any(DataArray[coordinate][1:].values != (DataArray[coordinate].values[:-1] + spacing_temp)):
                        spacing[coordinate] = str(DataArray[coordinate][0].values)+','+str(DataArray[coordinate][-1].values)+','+str(spacing_temp)
                    else:
                        spacing[coordinate] = 'irregular'
                else:
                    logging.warning('unknown dimension found that we will not be tracked in lib_dataarrays: ' + str(coordinate))

            dict_index_space = [key+':'+str(value) for key,value in spacing.items()]
            dict_index_space ='_'.join(dict_index_space)
            dict_index['space'] = dict_index_space


            logging.debug('adding crs grid mapping definition according to grid')
            if not (('grid_mapping' in dict_index) and (dic_index['grid_mapping'] not in [None, np.nan]) and not ((type(dict_index['grid_mapping']) in (float,np.float64)) and np.isnan(dict_index['grid_mapping']))):
                if ('latitude' in spacing) and ('longitude' in spacing):
                      grid_mapping_attributes = {
                              'grid_mapping':'crs',
                              'crs_grid_mapping_name': 'latitude_longitude',
                              'crs_long_name': 'CRS definition',
                              'crs_longitude_of_prime_meridian': 0.0,
                              'crs_semi_major_axis': 6378137.0,
                              'crs_inverse_flattening': 298.257223563,
                              'crs_spatial_ref': 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]]',
                              'crs_GeoTransform': spacing['longitude'].split(',')[0]+ ' '+spacing['longitude'].split(',')[2] +' '+spacing['latitude'].split(',')[0]+ ' '+spacing['latitude'].split(',')[2]
                      }
                      dict_index.update(grid_mapping_attributes)
                  
            #DataArray.attrs['space'] = dict_index_space

        # # not sure why we need to track coordinates
        # for key,index in dict_index.items():
        #     if key not in self.coordinates:
        #         if key not in ['variable','source','space',]:
        #             self.coordinates[key] = DataArray[key]
        #         if key == 'space':
        #             self.coordinates[key] = []
        #             for coordinate in space_coordinates:
        #                 self.coordinates[key].append(DataArray[coordinate])
        index = tuple([dict_index[key] for key in self.lib_dataarrays.index.names])

        for key in self.lib_dataarrays.index.names:
            if (key not in dict_index.keys()) or (dict_index[key] is None):
                raise ValueError ('Could not track key "'+key+'" that is required for the archive index.')

        #self.dataarrays[index] = DataArray

        if method in ['copy','copy_force_overwrite']:
            logging.info('Copying xarray as netcdf file to the internal library.')
            if destination_file == None:
                destination_file = os.path.dirname(os.path.realpath(self.path_pickle)) + '/' + ''.join(np.array(list(
                zip(self.file_pattern.split('"')[::2],
                    [{**dict_index, **dict_columns}[key] for key in self.file_pattern.split('"')[1::2]] + [
                        '']))).ravel())
            # if index in self.lib_dataarrays.index:
            #     if (method == 'copy_force_overwrite'):
            #         # if (self.lib_dataarrays.loc[index].path is not None) and \
            #         # (os.path.realpath(destination_file) == os.path.realpath(self.lib_dataarrays.loc[index].path):
            #         if (self.lib_dataarrays.loc[index].path is not None):
            #             filename = os.path.realpath(os.path.dirname(self.path_pickle)) + '/' + self.lib_dataarrays.loc[ index].path
            #             logging.warning('overwriting existing index ' + str(index) + ': ' + filename)
            #             if os.path.isfile(filename):
            #                 os.remove(filename)
            #             else:
            #                 logging.warning("I could not track previous file in library. So I'm not deleting")
            #         self.lib_dataarrays.loc[index]  = None
            #     else:
            #         raise IOError('index '+index+' already exists in library. Aborting. Please use copy_force_overwrite to override.')

            if os.path.isfile(destination_file):
                if (method == 'copy_force_overwrite'):
                    logging.warning('overwriting existing destination file '+destination_file)
                else:
                    raise IOError('Intended destination filename '+destination_file+' already exists')
            if not os.path.isdir(os.path.dirname(destination_file)):
                os.makedirs(os.path.dirname(destination_file))
            if filepath is not None:
                original_file = filepath #os.path.dirname(os.path.realpath(self.path_pickle))+'/'+filedict_columns['path']
                if os.path.realpath(original_file) == os.path.realpath(destination_file):
                    raise IOError('Cannot make a copy of the dataarray file, because the original and destination path are the same ('+original_file+'). Please use method=link file instead.')

                if (nc_reduce == True) and (('nc_reduced' not in dict_columns) or (dict_columns['nc_reduced'] != "True")):
                    reduced_temporary_file = tempfile.mktemp(suffix='.nc',dir='/tmp/')
                    logging.info('nc_reducing file: to '+reduced_temporary_file)
                    #try:
                    nc_reduce_fn(os.path.realpath(original_file),reduced_temporary_file,ncvariable = (kwargs['ncvariable'] if 'ncvariable' in kwargs else None),overwrite=True,nc_compress=nc_compress)
                    CMD = 'mv '+reduced_temporary_file+ ' '+destination_file
                    logging.info('executing: '+CMD); os.system(CMD)
                    dict_columns['nc_reduced'] = "True"
                    dict_columns['nc_compressed'] = str(nc_compress)
                    # except:
                    #     logging.critical('nc_reduce failed. removing temporary file. So we just keep the original file without reducing.')
                    #     os.system('rm '+reduced_temporary_file)
                #         
                else:
                    CMD = 'cp '+original_file+ ' '+destination_file
                    logging.info('Executing: ' + CMD); os.system(CMD)
            else:
                DataArray.to_netcdf(destination_file)
            dict_columns['path'] = os.path.relpath(os.path.realpath(destination_file),os.path.dirname(os.path.realpath(self.path_pickle)))

            DataArray.close()

            logging.info('We are copying, so attributes are updated to the new netcdf file.')
            ncfile = nc4.Dataset(os.path.dirname(os.path.realpath(self.path_pickle))+'/'+dict_columns['path'],'a')
            if 'ncvariable' in dict_columns.keys():
                ncvar = ncfile[dict_columns['ncvariable']]
            else:
                ncvar = ncfile['__xarray_dataarray_variable__']

            if 'grid_mapping' in dict_columns:
                grid_mapping_type = dict_columns['grid_mapping']
                if grid_mapping_type not in ncfile.variables:
                    ncfile.createVariable(grid_mapping_type,'S1')
                    logging.info('creating '+grid_mapping_type+' dummy char variable for storing grid_mapping attributes, which seems to be the format behaviour by, eg., qgis')


            for key,value in {**dict_index, **dict_columns}.items():

                if key not in ['ncvariable','path','path_pickle','linked','nc_reduced','absolute_path_as_cache']:# we also exclude linked and nc_reduced, because they are booleans and they give errors

                    if ('grid_mapping' in dict_columns) and key.startswith(grid_mapping_type+'_'):
                        logging.info('write '+grid_mapping_type+' attribute '+key+': '+str(value)+' in separate dummy char variable, which seems to be the format behaviour by eg., qgis')
                        ncfile.variables[grid_mapping_type].setncattr(key[(len(grid_mapping_type)+1):],value)

                    else:

                        logging.info('setting attribute '+key+' to '+str(value))
                        try:
                            ncvar.setncattr(key, value)
                        except:
                            ncvar.setncattr(key, str(value))

            ncfile.close()
            if 'ncvariable' in dict_columns.keys():
                DataArray =xr.open_dataset(os.path.dirname(os.path.realpath(self.path_pickle))+'/'+dict_columns['path'])[dict_columns['ncvariable']]
            else:
                DataArray = xr.open_dataarray(os.path.dirname(os.path.realpath(self.path_pickle)) + '/' + dict_columns['path'])
            dict_columns['linked'] = False
        elif method == 'add':
            dict_columns['linked'] = False
            dict_columns['path'] = os.path.relpath(os.path.realpath(filepath),os.path.dirname(os.path.realpath(self.path_pickle)))
        elif method == 'link':
            dict_columns['linked'] = True
            dict_columns['path'] = os.path.relpath(os.path.realpath(filepath),os.path.dirname(os.path.realpath(self.path_pickle)))
        else:
            raise ValueError('Add_dataarray method "{method}" not implemented.')

        if 'path' in dict_columns.keys():
            DataArray.close()
        else:
            self.dataarrays[index] = DataArray


        for key,value in dict_columns.items():
            if key not in self.lib_dataarrays.columns:
                self.lib_dataarrays[key] = ''
            if index not in self.lib_dataarrays.index:
                #import pdb; pdb.set_trace()
                lib_dataarrays_index_names = self.lib_dataarrays.index.names
                self.lib_dataarrays.loc[index] = None
                logging.debug('workaround index_names are forgotten when assigning first row')
                self.lib_dataarrays.index.names = lib_dataarrays_index_names
            self.lib_dataarrays.at[index,key] = value

        if sort_lib == True:
            self.lib_dataarrays.sort_index(inplace=True)

        if update_lib == True:
            # we don't write pickle location since this we enforce relative file structure. the column is only to work
            # with collection of archives.
            self.lib_dataarrays.drop(columns='path_pickle').to_pickle(self.path_pickle)


        logging.info('writing  path_pickle on a row basis. This is required to be able to work with collections of archives')
        self.lib_dataarrays.loc[index,'path_pickle'] = self.path_pickle

        return self.lib_dataarrays.loc[index]

    def add_dataarray_old(
            self,
            DataArray_or_filepath,
            skip_unavailable= False,
            release_dataarray_pointer=False,
            cache_to_tempdir=False,
            cache_to_ram=False,
            reset_space=False,
            engine=None,
            **kwargs,
    ):
        #DataArray = None
        if type(DataArray_or_filepath).__name__ == 'str':
            filepath = DataArray_or_filepath
            if (cache_to_tempdir is None) or cache_to_tempdir:
                if type(cache_to_tempdir) is not str:
                    cache_to_tempdir = tempfile.gettempdir()
                #filepath_as_cache = cache_to_tempdir+'/'+os.path.basename(filepath)

                filepath_as_cache = tempfile.mktemp(prefix=os.path.basename(filepath)[:-3]+'_',suffix='.nc',dir=cache_to_tempdir)
                CMD='cp '+filepath+' '+filepath_as_cache

                logging.info('caching to temporary file: '+CMD)
                os.system(CMD)
                filepath_for_reading = filepath_as_cache
            else:
                filepath_as_cache = None
                filepath_for_reading = filepath
                if cache_to_ram:
                    CMD='cat '+filepath_for_reading+' > /dev/null'
                    logging.info('caching to ram: '+CMD)
                    os.system(CMD)

            # ncvariable: variable as seen on disk
            # variable (= DataArray.name): variable as considered in the library
            if ('ncvariable' not in kwargs.keys()) or ((type(kwargs['ncvariable']).__name__ == 'float') and np.isnan(kwargs['ncvariable'])):
                if 'variable' in kwargs.keys():
                    ncvariable = kwargs['variable']

                logging.info('Opening file: '+filepath_for_reading+ ' (original file: '+filepath+')')
                if ncvariable is not None:
                    # print('reading',filepath,ncvariable)
                    # try:
                    #     print('trying with open_dataarray first because of obscure performance decreases with xr.open_dataset')
                    #     DataArray = xr.open_dataarray(filepath)
                    #     if DataArray.name != kwargs['ncvariable']:
                    #         DataArray.close()
                    #         del DataArray
                    #         print('first variable is not the correct ncvariable. Trying it with xr.open_dataset...')
                    #         raise ValueError('first variable is not the correct ncvariable')
                    # except:
                    try:

                        ds = xr.open_dataset(filepath_for_reading,engine=engine)
                        DataArray = ds[ncvariable]
                        ds.close()
                        del ds
                        #kwargs['ncvariable'] = ncvariable
                    except:
                        DataArray = xr.open_dataarray(filepath_for_reading,engine=engine)
                else:
                    DataArray = xr.open_dataarray(filepath_for_reading,engine=engine)
                kwargs['ncvariable'] = DataArray.name
            else:
                ds = xr.open_dataset(filepath_for_reading,engine=engine)
                DataArray = ds[kwargs['ncvariable']]
                ds.close()
                del ds
            # except:
            #     print ( "Warning! Error while reading from disk: ", sys.exc_info()[0])
            #     if not skip_unavailable:
            #         raise ('...stopping' )
            #     else:
            #         print('...skipping' )

            kwargs['absolute_path'] = os.path.abspath(filepath)
            kwargs['absolute_path_for_reading'] = os.path.abspath(filepath_for_reading)
            kwargs['absolute_path_as_cache'] = (None if filepath_as_cache is None else os.path.abspath(filepath_as_cache))
            kwargs['available'] = True
        else:
            DataArray = DataArray_or_filepath

            #kwargs['absolute_path'] = None

        # for key in self.lib_dataarrays.index.names:
        #     if key not in (list(DataArray.attrs.keys())+['variable']):
        #         raise ValueError(key+' needs to be in DataArray.attrs')

        # if DataArray is None:
        #     print('Skipping ',DataArray_or_filepath,kwargs)
        # else:
        if not DataArray is None:

            dict_index = {}
            dict_columns = {}

            if DataArray.name is None:
                raise ValueError ('input dataarray should have a name')

            if 'variable' not in kwargs.keys():
                kwargs['variable'] = DataArray.name

            DataArray.name = kwargs['variable']

            dict_index['variable'] = DataArray.name

            for key,value in DataArray.attrs.items():
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
            if ('time' not in dict_index.keys()) or (dict_index['time'] is None) or (type(dict_index['time']).__name__ == 'float') and (  np.isnan(dict_index['time']).any()):
                logging.info('Guessing time coordinate from DataArray')
                # is month type

                #monthly spacing
                if np.apply_along_axis(lambda y: np.sum((y[1:] - y[:-1] != 1),0),0,np.vectorize(lambda x: int(x[:4])*12+int(x[5:7]))(DataArray.time.values.astype('str'))).item() == 0:
                    dict_index['time'] = \
                            'monthly_'+str(DataArray.time[0].values)[:7]+'_'+str(DataArray.time[-1].values)[:7]
                # also monthly
                elif (not np.any( ~(np.vectorize(lambda x: x[8:])(DataArray.time.values.astype('str')) == '01T00:00:00.000000000'))):
                    dict_index['time'] = \
                            'monthly_'+str(DataArray.time[0].values)[:7]+'_'+str(DataArray.time[-1].values)[:7]
                elif not np.any((DataArray.time[2:-1].values - DataArray.time[1:-2].values) != np.array(86400000000000, dtype='timedelta64[ns]')):
                    #daily
                    dict_index['time'] = 'daily_'+np.datetime_as_string(DataArray.time[0].values,unit='D')+'_'+np.datetime_as_string(DataArray.time[-1].values,unit='D')
                elif not np.any((DataArray.time[2:-1].values - DataArray.time[1:-2].values) != dt.timedelta(days=1)):
                    dict_index['time'] = \
                            'daily_'+str(DataArray.time[0].values)[:10]+'_'+str(DataArray.time[-1].values)[:10]
                else:
                    raise ValueError('time dimension not implemented')

                #DataArray.attrs['time'] = dict_index['time']

            renamings = {'lat':'latitude','lon':'longitude'}
            for key,value in renamings.items():
                if key in DataArray.dims:
                    DataArray = DataArray.rename({key:value})

            # filter coordinates that are listed in the library index (these are not treated under space but separately, eg., 'time').
            space_coordinates = list(DataArray.dims)
            for key in self.lib_dataarrays.index.names:
                if key in list(DataArray.dims):
                    space_coordinates.remove(key)
            space_coordinates_only = ['latitude','longitude']

            if ('space' not in dict_index.keys()) or (dict_index['space'] is None) or reset_space:
                spacing = {}
                for coordinate in space_coordinates:
                    if coordinate in space_coordinates_only:
                        spacing_temp = (DataArray[coordinate].values[1] - DataArray[coordinate].values[0])
                        if not np.any(DataArray[coordinate][1:].values != (DataArray[coordinate].values[:-1] + spacing_temp)):
                            spacing[coordinate] = str(DataArray[coordinate][0].values)+','+str(DataArray[coordinate][-1].values)+','+str(spacing_temp)
                        else:
                            spacing[coordinate] = 'irregular'
                    else:
                        logging.warning('unknown dimension found that we will not be tracked in lib_dataarrays: '+str(coordinate))
                dict_index_space = [key+':'+str(value) for key,value in spacing.items()]
                dict_index_space ='_'.join(dict_index_space)
                dict_index['space'] = dict_index_space
                #DataArray.attrs['space'] = dict_index_space


            for key,index in dict_index.items():
                if key not in self.coordinates:
                    if key not in ['variable','source','space',]:
                            self.coordinates[key] = DataArray[key]
                    if key == 'space':
                        self.coordinates[key] = []
                        for coordinate in space_coordinates:
                            self.coordinates[key].append(DataArray[coordinate])

            for key in self.lib_dataarrays.index.names:
                if (key not in dict_index.keys()) or (dict_index[key] is None):
                    raise ValueError ('Could not track key "'+key+'" that is required for the archive index.')
            index = tuple([dict_index[key] for key in self.lib_dataarrays.index.names])


            self.dataarrays[index] = DataArray
            DataArray.close()
            del DataArray

            if (self.mode == 'passive') and (not self.lib_dataarrays.loc[index]['absolute_path'].isnull().any() ):
               self.dataarrays[index].close()

            for key,value in dict_index.items():
                if key == 'variable':
                    self.dataarrays[index].name = value
                else:
                    self.dataarrays[index].attrs[key] = value

            if 'absolute_path' not in self.lib_dataarrays.columns:
                self.lib_dataarrays['absolute_path'] = None
            if 'absolute_path_for_reading' not in self.lib_dataarrays.columns:
                self.lib_dataarrays['absolute_path_for_reading'] = None
            if 'absolute_path_as_cache' not in self.lib_dataarrays.columns:
                self.lib_dataarrays['absolute_path_as_cache'] = None
            # if 'dataarray_pointer' not in self.lib_dataarrays.columns:
            #     self.lib_dataarrays['dataarray_pointer'] = None
            if 'path' not in self.lib_dataarrays.columns:
                self.lib_dataarrays['path'] = None
            if 'available' not in self.lib_dataarrays.columns:
                self.lib_dataarrays['available'] = None

            self.lib_dataarrays.loc[index] = None
            for key,value in dict_columns.items():
                if key not in self.lib_dataarrays.columns:
                    self.lib_dataarrays[key] = ''
                if index not in self.lib_dataarrays.index:
                    self.lib_dataarrays.loc[index] = ''
                self.lib_dataarrays[key].loc[index] = value
                if key not in [
                    #'dataarray_pointer',
                    'absolute_path_as_cache',
                    'absolute_path_for_reading',
                    'absolute_path',
                    'path',
                    'available']:
                    self.dataarrays[index].attrs[key] = value

            self.lib_dataarrays.sort_index(inplace=True)

            if release_dataarray_pointer:
                logging.info('closing '+str(index))
                self.dataarrays[index].close()
                if cache_to_tempdir:
                    CMD='rm '+filepath_as_cache
                    logging.info('Released pointer, so removing cached file: '+CMD)
                    os.system(CMD)
                # del self.dataarrays[index]
                #self.lib_dataarrays.loc[index,'dataarray_pointer'] = None
            # else:
            #     self.lib_dataarrays.loc[index,'dataarray_pointer'] = self.dataarrays[index]

    #         if self.path_pickle:# and (type(self.lib_dataarrays.loc[index].absolute_path) != str):
    #             self.dump()


            # if attrs is not None:
            #     for key,value in attrs.items():
            #         DataArray

            # for key,value in kwargs.items():
            #     if key not in self.lib_dataarrays.columns:
            #         self.lib_dataarrays[key] = ''
            #     self.lib_dataarrays.loc[index][key] = value

    #    def intersect_times(self,DataArray_input):
    def dataarrays_apply(self,function,query=None,inplace=False, attrs=None):
        # if query is not None:
        #     lib_dataarrays_out = self.lib_dataarrays.query(query,engine='python').copy()
        # elsepath
        #     lib_dataarrays_out = self.lib_dataarrays.copy()

        archive_out = archive()
        for index,columns in enumerate(self.lib_dataarrays.to_dict('records')):
            dataarray_out_temp = function(self.dataarrays[index])
            for key,value in self.dataarrays[index].attrs.items():
                dataarray_out_temp.attrs[key] = value
            if attrs is not None:
                for key,value in attrs.items():
                    dataarray_out_temp.attrs[key] = value
            archive_out.add_dataarray_old(dataarray_out_temp)

        return archive_out

    def apply_virtual(self,func,groupby=None,apply_merge=[],apply_merge_out=[],archive_out = None, inherit_attributes = True,extra_attributes={}, **kwargs):

        if (archive_out is None) and (self.path_pickle is None):
            raise ValueError('Please specify how the data should be written '
                    'out. In case you want to create a new archive returned by '
                    'this procedure. Specify path_archive_out="/path/to/dir". In case '
                    'you want to merge it to the current archive, you need to '
                    'dump the current archive so that the self.path_pickle is '
                    'set.')

        temp_df = self.lib_dataarrays.reset_index()


        if type(apply_merge) == pd.core.frame.DataFrame:
            apply_merge_df = apply_merge
        elif type(apply_merge) == list:
            apply_merge_df = pd.DataFrame(apply_merge)
        elif type(apply_merge) == dict:
            apply_merge_df = pd.DataFrame.from_dict([dict(zip(apply_merge.keys(),v)) for v in itertools.product(*apply_merge.values())])
        apply_merge_index = pd.MultiIndex.from_frame(apply_merge_df)

        if type(apply_merge_out) == pd.core.frame.DataFrame:
            apply_merge_out_df = apply_merge_out
        elif type(apply_merge_out) == list:
            apply_merge_out_df = pd.DataFrame(apply_merge_out)
        elif type(apply_merge_out) == dict:
            apply_merge_out_df = pd.DataFrame.from_dict([dict(zip(apply_merge_out.keys(),v)) for v in itertools.product(*apply_merge_out.values())])

        if len(apply_merge_out_df) == 0:
            logging.info('creating automatic single output table')
            apply_merge_out_dict = {}
            for column in apply_merge_df.columns:
                apply_merge_out_dict[column] = ['from__'+'__'.join(apply_merge_df[column].unique())]
            apply_merge_out_df = pd.DataFrame.from_dict([dict(zip(apply_merge_out_dict.keys(),v)) for v in itertools.product(*apply_merge_out_dict.values())])


        apply_merge_out_index = pd.MultiIndex.from_frame(apply_merge_out_df)

        #apply_merge_pandas = pd.DataFrame([{'variable':'msshf_0001'},{'variable':'mslhf_0001','source':'cds_era5'}])

        #self.lib_dataarrays.index.names
        if groupby is None:
            groupby = list(self.lib_dataarrays.index.names)
            for key in apply_merge_df.columns:
                groupby = list(filter((key).__ne__, groupby))

        if archive_out is not None:
            write_mode = 'add_to_external_archive'
        else:
            write_mode = 'add_to_current_archive'


        if write_mode == 'create_new_archive':
            archive_out = archive()
            archive_out.dump(path_archive_out)
        elif write_mode == 'add_to_current_archive':
            archive_out = self

        xr_outputs = {}
        for index,group in temp_df.groupby(groupby):
            if type(index) is not tuple:
                index_multi = (index,)
            else:
                index_multi = index

            group_columns = group.set_index(list(apply_merge_df.columns))
            if type(group_columns.index) == pd.core.indexes.base.Index:
                MultiIndex_from_Single_Index = lambda index: pd.MultiIndex.from_tuples([x.split()[::-1] for x in index])
                group_columns.index = MultiIndex_from_Single_Index(group_columns.index)

            all_variables_available_in_this_group = True
            if len(apply_merge_index.intersection(group_columns.index)) != len(apply_merge_index):
             all_variables_available_in_this_group = False
            # if not group_columns.loc[apply_merge_index][groupby].isnull().any().any():
            #  all_variables_available_in_this_group = False

            if all_variables_available_in_this_group:
                  dataarrays_for_func = []

                  logging.info('start iterrows')
                  for index_group,columns in group_columns.loc[apply_merge_index].iterrows():
                      index_array_dict =   {**dict(columns),**dict(zip(apply_merge_index.names,index_group))}#**dict(zip(apply_merge_index.names,index_group))} **dict(zip(groupby,index)),
                      index_array_tuple_ordered =  tuple([index_array_dict[key] for key in self.lib_dataarrays.index.names])

                      if (self.mode == 'passive') and (not self.lib_dataarrays.loc[index]['absolute_path'].isnull().any() ):
                          logging.critical('to be implemented')
                          import pdb; pdb.set_trace()
                      else:
                          dataarrays_for_func.append(self.dataarrays[index_array_tuple_ordered])
                  logging.info('end iterrows')

                  filenames_out = []

                  attributes = []
                  ifile = 0
                  logging.info('start iterrows')
                  for index_group,group_columns in apply_merge_out_df.iterrows():
                      index_array_out_dict = {**dict(zip(groupby,index_multi)),**dict(zip(apply_merge_out_df.columns,group_columns))}
                      attributes.append(index_array_out_dict)
                      index_array_out_tuple_ordered =  tuple([index_array_out_dict[key] for key in self.lib_dataarrays.index.names])

                      if inherit_attributes:
                          for key,value in dataarrays_for_func[min(len(dataarrays_for_func),ifile)].attrs.items():
                              if (key not in self.lib_dataarrays.index.names)  and \
                                 ( inherit_attributes or ((type(inherit_attributes) is list) and (key in inherit_attributes))) and \
                                 (key not in attributes[-1].keys()):
                                  attributes[-1][key] = value

                      for key,value in extra_attributes.items():
                          attributes[-1][key] = value

                      # if (archive_out.file_pattern is None):
                      #     raise ValueError("I don't know how to write the data file to disk. Please set to file_pattern")
                      #filenames_out.append(os.path.dirname(archive_out.path_pickle)+'/'+''.join(np.array(list(zip(archive_out.file_pattern.split('"')[::2],[attributes[-1][key] for key in archive_out.file_pattern.split('"')[1::2]]+['']))).ravel()))
                      ifile +=1
                  logging.info('end iterrows')
                  #  ifile = 0
                  #  for index_group,group_columns in apply_merge_out_df.iterrows():
                  #      index_array_out_tuple_ordered =  tuple([attributes[ifile][key] for key in archive_out.lib_dataarrays.index.names])
                  #      if index_array_out_tuple_ordered in archive_out.dataarrays.keys():
                  #          print('forcing to overwrite data for ',index_array_out_tuple_ordered,)
                  #          self.remove_by_index(index_array_out_tuple_ordered,delete_on_disk=True)
                  #      ifile +=1


                  # for filename_out in filenames_out:
                  #     os.system('mkdir -p '+os.path.dirname(filename_out))

                  temp_dataarrays = func(tuple(dataarrays_for_func))

                    # if type(temp_dataarrays) != tuple:
                    #     print('this is a workaround in case we get a single dataarray instead of tuple of dataarrays from the wrapper function. This needs revision')
                    #     idataarray = 0
                    #     for key,value in attributes[idataarray].items():
                    #         if key not in ['variable','absolute_path','path']:
                    #             temp_dataarrays.attrs[key] = value
                    #         if key == 'variable':
                    #             temp_dataarrays.name = value
                    #     #import pdb;pdb.set_trace()
                    #     temp_dataarrays.to_netcdf(filenames_out[idataarray])
                    #     temp_dataarrays.close()
                    # else:
                    #     for idataarray in range(len(temp_dataarrays)):
                    #         for key,value in attributes[idataarray].items():
                    #             if key not in ['variable','absolute_path','path']:
                    #                 temp_dataarrays[idataarray].attrs[key] = value
                    #             if key == 'variable':
                    #                 temp_dataarrays[idataarray].name = value

                    #     for idataarray in range(len(temp_dataarrays)):
                    #         temp_dataarrays[idataarray].to_netcdf(filenames_out[idataarray])
                    #         temp_dataarrays[idataarray].close()

                  for idataarray,dataarray in enumerate(temp_dataarrays):
                      for key,value in attributes[idataarray].items():
                          dataarray.attrs[key] = value
                      archive_out.add_dataarray_old(dataarray)
        if write_mode == 'create_new_archive':
            return archive_out



#             if type(group_columns_out.index) == pd.core.indexes.base.Index:
#                 MultiIndex_from_Single_Index = lambda index: pd.MultiIndex.from_tuples([x.split()[::-1] for x in index])
#                 group_columns_out.index = MultiIndex_from_Single_Index(group_columns_out.index)
#
#             # not_all_arrays_available_in_this_group = group_columns.loc[apply_merge_out_index][groupby].isnull().any().any()
#             #       dataarrays_for_func = []
#
#             for index_group,columns in group_columns_out.loc[apply_merge_out_index].iterrows():
#                 index_array_dict =   {**dict(columns),**dict(zip(apply_merge_out_index.names,index_group))}#**dict(zip(apply_merge_index.names,index_group))} **dict(zip(groupby,index)),
#                 index_array_tuple_ordered =  tuple([index_array_dict[key] for key in self.lib_dataarrays.index.names])
#                 # dataarrays_for_func.append(self.dataarrays[index_array_tuple_ordered])
#

                #group_columns.loc[apply_merge_index]


        # temp_df.groupby(self.lib_dataarrays.index.names
        # if kwargs.keys() != 1:
        #     raise ValueError('length different from 1 is not allowed')

    def apply_virtual2(self,func,divide_into_groups_extra=None, apply_groups_in=[],apply_groups_out=[], archive_out=None,inherit_attributes=True,extra_attributes={},**kwargs ):


        divide_into_groups = []
        for name in self.lib_dataarrays.index.names:
            if ((name not in apply_groups.keys()) and (name not in apply_groups.keys())):
                divide_into_groups.append(name)
        for name in divide_into_groups_extra:
            if name not in divide_into_groups:
                divide_into_groups.append(name)

        for index,group in self.lib_dataarrays.reset_index().groupby(divide_into_groups):

            def allways_multi_index (idx):
                if type(idx) is not tuple:
                    return (idx,)
                else:
                    return idx

            multi_index = allways_multi_index(index)

            apply_groups_columns = []
            for key in apply_groups_in.keys():
                apply_groups_columns.append(key)
            for key in apply_groups_out.keys():
                if key not in apply_groups_columns:
                    apply_groups_columns.append(key)

            apply_this_group_in = pd.dataframe(columns=apply_groups_columns)







        def parse_to_dataframe(list_or_dict_or_dataframe):
            if type(list_or_dict_or_dataframe) == pd.core.frame.DataFrame:
                return list_or_dict_or_dataframe
            elif type(list_or_dict_or_dataframe) == list:
                return pd.DataFrame(list_or_dict_or_dataframe)
            elif type(list_or_dict_or_dataframe) == dict:
                return pd.DataFrame.from_dict([dict(zip(list_or_dict_or_dataframe.keys(),v)) for v in itertools.product(*list_or_dict_or_dataframe.values())])

        apply_groups_in_df = parse_to_dataframe(apply_groups_in)
        apply_groups_out_df = parse_to_dataframe(apply_groups_out)

        # import pdb;pdb.set_trace()


        # apply_merge_index = pd.MultiIndex.from_frame(apply_merge_df)

        # if type(apply_groups_out) == pd.core.frame.DataFrame:
        #     apply_groups_out_df = apply_groups_out
        # elif type(apply_groups_out) == list:
        #     apply_groups_out_df = pd.DataFrame(apply_groups_out)
        # elif type(apply_groups_out) == dict:





        # apply_groups_in_out_pre =

        # if archive_out is not None:
        #     write_mode = 'add_to_external_archive'
        # else:
        #     write_mode = 'add_to_current_archive'



        #     apply_groups_index




    #def cdo(self,cdostring,):
    def apply_func(self,
            func,
            #lib_dataarrays = self.lib_dataarrays
            archive_out = None,
            update_pickle = True,
            force_recalculate=False,
            nc_reduce=False,
            transpose_inner=True,
            *args,
            **kwargs):
        #apply_groups_in = {'variable':['aridity'],'source':[None]}
        #apply_groups_out={'variable':['aridity'],'source':[lambda labels: labels[0].replace('historical','rcp45'),lambda labels: labels[0].replace('historical','rcp85')]}
        #archive_out = pcd.archive()
        #mode='xarray'

        if (archive_out is None) and (self.path_pickle is None) and (mode != 'xarray'):
            raise ValueError('Please specify how the data should be written '
                    'out. In case you want to create a new archive returned by '
                    'this procedure. Specify path_archive_out="/path/to/dir". In case '
                    'you want to merge it to the current archive, you need to '
                    'dump the current archive so that the self.path_pickle is '
                    'set.')

        if archive_out is not None:
            write_mode = 'add_to_external_archive'
        else:
            write_mode = 'add_to_current_archive'


        # if write_mode == 'create_new_archive':
        #     archive_out = archive()
        #     archive_out.dump(path_archive_out)
        #el
        if write_mode == 'add_to_current_archive':
            archive_out = self

        apply_func_wrapper(
            func,
            lib_dataarrays = self.lib_dataarrays,
            dataarrays = self.dataarrays,
            archive_out = archive_out,
            update_pickle=update_pickle,
            force_recalculate=force_recalculate,
            nc_reduce = nc_reduce,
            **kwargs,
        )

        if write_mode == 'create_new_archive':
            return archive_out

    def apply_func_old(self,func, xarray_function_wrapper=apply_func,dataarrays_wrapper = lambda *x: (*x,),groupby=None,apply_merge=[],apply_merge_out=[],archive_out = None,keep_in_memory_during_processing = False, inherit_attributes = False,extra_attributes={}, **kwargs):

        if (archive_out is None) and (self.path_pickle is None):
            raise ValueError('Please specify how the data should be written '
                    'out. In case you want to create a new archive returned by '
                    'this procedure. Specify path_archive_out="/path/to/dir". In case '
                    'you want to merge it to the current archive, you need to '
                    'dump the current archive so that the self.path_pickle is '
                    'set.')

        temp_df = self.lib_dataarrays.reset_index()


        if type(apply_merge) == pd.core.frame.DataFrame:
            apply_merge_df = apply_merge
        elif type(apply_merge) == list:
            apply_merge_df = pd.DataFrame(apply_merge)
        elif type(apply_merge) == dict:
            apply_merge_df = pd.DataFrame.from_dict([dict(zip(apply_merge.keys(),v)) for v in itertools.product(*apply_merge.values())])
        apply_merge_index = pd.MultiIndex.from_frame(apply_merge_df)

        if type(apply_merge_out) == pd.core.frame.DataFrame:
            apply_merge_out_df = apply_merge_out
        elif type(apply_merge_out) == list:
            apply_merge_out_df = pd.DataFrame(apply_merge_out)
        elif type(apply_merge_out) == dict:
            apply_merge_out_df = pd.DataFrame.from_dict([dict(zip(apply_merge_out.keys(),v)) for v in itertools.product(*apply_merge_out.values())])

        if len(apply_merge_out_df) == 0:
            logging.info('creating automatic single output table')
            apply_merge_out_dict = {}
            for column in apply_merge_df.columns:
                apply_merge_out_dict[column] = ['from__'+'__'.join(apply_merge_df[column].unique())]
            apply_merge_out_df = pd.DataFrame.from_dict([dict(zip(apply_merge_out_dict.keys(),v)) for v in itertools.product(*apply_merge_out_dict.values())])


        apply_merge_out_index = pd.MultiIndex.from_frame(apply_merge_out_df)


        #apply_merge_pandas = pd.DataFrame([{'variable':'msshf_0001'},{'variable':'mslhf_0001','source':'cds_era5'}])

        #self.lib_dataarrays.index.names
        if groupby is None:
            groupby = list(self.lib_dataarrays.index.names)
            for key in apply_merge_df.columns:
                groupby = list(filter((key).__ne__, groupby))

        if archive_out is not None:
            write_mode = 'add_to_external_archive'
        else:
            write_mode = 'add_to_current_archive'


        if write_mode == 'create_new_archive':
            archive_out = archive()
            archive_out.dump(path_archive_out)
        elif write_mode == 'add_to_current_archive':
            archive_out = self

        xr_outputs = {}
        for index,group in temp_df.groupby(groupby):

            group_columns = group.set_index(list(apply_merge_df.columns))
            if type(group_columns.index) == pd.core.indexes.base.Index:
                MultiIndex_from_Single_Index = lambda index: pd.MultiIndex.from_tuples([x.split()[::-1] for x in index])
                group_columns.index = MultiIndex_from_Single_Index(group_columns.index)

            all_variables_available_in_this_group = True
            if len(apply_merge_index.intersection(group_columns.index)) != len(apply_merge_index):
             all_variables_available_in_this_group = False
            # if not group_columns.loc[apply_merge_index][groupby].isnull().any().any():
            #  all_variables_available_in_this_group = False
            #  import pdb; pdb.set_trace()

            if all_variables_available_in_this_group:
                  dataarrays_for_func = []

                  for index_group,columns in group_columns.loc[apply_merge_index].iterrows():
                      index_array_dict =   {**dict(columns),**dict(zip(apply_merge_index.names,index_group))}#**dict(zip(apply_merge_index.names,index_group))} **dict(zip(groupby,index)),
                      index_array_tuple_ordered =  tuple([index_array_dict[key] for key in self.lib_dataarrays.index.names])

                      if (self.mode == 'passive') and (not self.lib_dataarrays.loc[index]['absolute_path'].isnull().any() ):
                          logging.critical('to be implemented')
                          import pdb; pdb.set_trace()
                      else:
                          dataarrays_for_func.append(self.dataarrays[index_array_tuple_ordered])

                  filenames_out = []

                  attributes = []
                  ifile = 0
                  for index_group,group_columns in apply_merge_out_df.iterrows():
                      index_array_out_dict = {**dict(zip(groupby,index)),**dict(zip(apply_merge_out_df.columns,group_columns))}
                      attributes.append(index_array_out_dict)
                      index_array_out_tuple_ordered =  tuple([index_array_out_dict[key] for key in self.lib_dataarrays.index.names])


                      if inherit_attributes:
                          for key,value in dataarrays_for_func[min(len(dataarrays_for_func),ifile)].attrs.items():
                              if (key not in self.lib_dataarrays.index.names)  and \
                                 ( inherit_attributes or ((type(inherit_attributes) is list) and (key in inherit_attributes))) and \
                                 (key not in attributes[-1].keys()):
                                  attributes[-1][key] = value

                      for key,value in extra_attributes.items():
                          attributes[-1][key] = value

                      if (archive_out.file_pattern is None):
                          raise ValueError("I don't know how to write the data file to disk. Please set to file_pattern")
                      filenames_out.append(os.path.dirname(archive_out.path_pickle)+'/'+''.join(np.array(list(zip(archive_out.file_pattern.split('"')[::2],[attributes[-1][key] for key in archive_out.file_pattern.split('"')[1::2]]+['']))).ravel()))
                      ifile +=1

                  for ixr_out,filename_out in enumerate(filenames_out):
                      index_array_out_tuple_ordered =  tuple([attributes[ixr_out][key] for key in archive_out.lib_dataarrays.index.names])
                      if index_array_out_tuple_ordered in archive_out.dataarrays.keys():
                          self.remove_by_index(index_array_out_tuple_ordered,delete_on_disk=True)


                  for filename_out in filenames_out:
                      os.system('mkdir -p '+os.path.dirname(filename_out))

                  if not keep_in_memory_during_processing:
                    xarray_function_wrapper(func,dataarrays_wrapper(*tuple(dataarrays_for_func)),filenames_out=filenames_out,attributes = attributes, release=True, **kwargs)
                  else:
                    temp_dataarrays = xarray_function_wrapper(func,dataarrays_wrapper(*tuple(dataarrays_for_func)),**kwargs)

                    if type(temp_dataarrays) != tuple:
                        logging.info('this is a workaround in case we get a single dataarray instead of tuple of dataarrays from the wrapper function. This needs revision')
                        idataarray = 0
                        for key,value in attributes[idataarray].items():
                            if key not in self.not_dataarray_attributes:
                                temp_dataarrays.attrs[key] = value
                            if key == 'variable':
                                temp_dataarrays.name = value
                        #import pdb;pdb.set_trace()
                        os.system('rm '+filenames_out[idataarray])
                        temp_dataarrays.to_netcdf(filenames_out[idataarray])
                        temp_dataarrays.close()
                    else:
                        for idataarray in range(len(temp_dataarrays)):
                            for key,value in attributes[idataarray].items():
                                if key not in self.not_dataarray_attributes:
                                    temp_dataarrays[idataarray].attrs[key] = value
                                if key == 'variable':
                                    temp_dataarrays[idataarray].name = value

                        for idataarray in range(len(temp_dataarrays)):
                            os.system('rm '+filenames_out[idataarray])
                            temp_dataarrays[idataarray].to_netcdf(filenames_out[idataarray])
                            temp_dataarrays[idataarray].close()

                  for ixr_out,filename_out in enumerate(filenames_out):
                      archive_out.add_dataarray_old(filename_out)

        if write_mode == 'create_new_archive':
            return archive_out



#             if type(group_columns_out.index) == pd.core.indexes.base.Index:
#                 MultiIndex_from_Single_Index = lambda index: pd.MultiIndex.from_tuples([x.split()[::-1] for x in index])
#                 group_columns_out.index = MultiIndex_from_Single_Index(group_columns_out.index)
#
#             # not_all_arrays_available_in_this_group = group_columns.loc[apply_merge_out_index][groupby].isnull().any().any()
#             #       dataarrays_for_func = []
#
#             import pdb;pdb.set_trace()
#             for index_group,columns in group_columns_out.loc[apply_merge_out_index].iterrows():
#                 index_array_dict =   {**dict(columns),**dict(zip(apply_merge_out_index.names,index_group))}#**dict(zip(apply_merge_index.names,index_group))} **dict(zip(groupby,index)),
#                 index_array_tuple_ordered =  tuple([index_array_dict[key] for key in self.lib_dataarrays.index.names])
#                 # dataarrays_for_func.append(self.dataarrays[index_array_tuple_ordered])
#




                #group_columns.loc[apply_merge_index]


        # temp_df.groupby(self.lib_dataarrays.index.names
        # if kwargs.keys() != 1:
        #     raise ValueError('length different from 1 is not allowed')


    def update(self,
               library_path=None,
               query=None,
               force_overwrite_dataarrays = False,
               force_overwrite_pickle=False,
               extra_attributes={},
               dump_floating_dataarrays=False,**kwargs):

        """
            perform quick attribute parameter updates and/or dump unsynced information to disc on request.
        """
        if library_path is not None:
            if library_path[-4:] == '.pkl':
                lib_basename = os.path.basename(os.path.abspath(library_path))
                lib_dirname = os.path.dirname(os.path.abspath(library_path))
            elif library_path[-1:] == '/':
                lib_basename = 'master.pkl'
                lib_dirname = os.path.abspath(library_path)
            else:
                raise ValueError('Please provide a path that either ends on ".pkl" (indicating specific picklefile) or "/" (full archive dump at master.pkl)')


            if os.path.isfile(lib_dirname+'/'+lib_basename) and (not force_overwrite_pickle):
                raise IOError('pickle file exists. Please use force_overwrite_pickle = True, or specify the full pickle name as the library_path.')

            # when the self.path_pickle exists, it is assumed that this file needs to be updated
            self.path_pickle = lib_dirname+'/'+lib_basename
        # else:
        #     lib_dirname = os.path.dirname(self.path_pickle)


        # if self.path_pickle is None:
        #     raise ValueError("I can't track the pickle name. Please specify a pkl file or directory with library_path")

        for key,value in kwargs.items():
            if key not in self.settings_keys:
                raise ValueError('"'+key+'" is not a known setting.')
            else:
                self.__dict__[key] = value

        if query is not None:
            read_lib_dataarrays = self.lib_dataarrays.query(query).copy()
        else:
            read_lib_dataarrays = self.lib_dataarrays.copy()

        if len(extra_attributes) > 0:
            for idx,row in read_lib_dataarrays.iterrows():

                # # this should become part of a save procedure updating the attributes of the netcdf on disc
                # if ('absolute_path' in row.keys()) and ('ncvariable' in row.keys()):
                #     variable=idx[self.lib_dataarrays.index.names.index('variable')]
                #     self.remove_by_index(idx)
                #     ncin = nc4.open_dataset(row['absolute_path'])
                #     for attribute_key,attribute_value in extra_attributes.items():
                #         ncin[idx[self.lib_dataarrays.index.names.index('variable')]].setncattr(attribute_key,attribute_value)
                #     ncin.close()
                #     self.add_dataarray(row['absolute_path'],variable,ncvariable=row['ncvariable'])
                # else:
                dataarray_temp = self.dataarrays[idx]

                attributes_temp = {**dict(zip(read_lib_dataarrays.index.names,idx)),**row}
                for attribute_key,attribute_value in extra_attributes.items():
                    attributes_temp[attribute_key] = attribute_value

                # extra_attributes_plus_path = extra_attributes.copy()
                # if ( ('path' in row.keys()) and (type(row['path']) == str)):
                #     extra_attributes_plus_path['path'] =row['path']
                # if ( ('absolute_path' in row.keys()) and (type(row['absolute_path']) == str)):
                #     extra_attributes_plus_path['absolute_path'] =row['absolute_path']

                self.remove_by_index(idx,update_pickle=False)
                self.add_dataarray_old(dataarray_temp,**attributes_temp)

        # for key,value in extra_attributes.items():
        #     self.lib_dataarrays.loc[read_lib_dataarrays.index][key] = value

        if 'path_pickle' in self.__dict__.keys():
            os.system('mkdir -p '+os.path.dirname(self.path_pickle))

        read_lib_dataarrays = self.lib_dataarrays.copy()

        for idx,columns in read_lib_dataarrays.iterrows():
            if ( ('absolute_path' not in columns.keys()) or (type(columns['absolute_path']) != str)) or \
               ( ('path' not in columns.keys()) or (type(columns['path']) != str)):

                if dump_floating_dataarrays:
                    #parse filename according to file_pattern
                    if 'path_pickle' not in self.__dict__.keys():
                        raise ValueError ('self.path_pickle is not set')
                    fnout = os.path.dirname(self.path_pickle)+'/'+''.join(np.array(list(zip(self.file_pattern.split('"')[::2],[{**dict(zip(self.lib_dataarrays.index.names,idx)),**columns}[key] for key in self.file_pattern.split('"')[1::2]]+['']))).ravel())
                    logging.info("File pointer for "+idx+" is not known, so I'm dumping a new file under "+fnout)
                    #fnout = self.lib_dataarrays.loc[idx]['absolute_path']
                    if (not force_overwrite_dataarrays) and (os.path.isfile(fnout)):
                        raise IOError(fnout+' exists. Use force_overwrite_dataarrays to overwrite file')
                    os.system('mkdir -p '+os.path.dirname(fnout))
                    # self.dataarrays[idx].attrs['absolute_path'] = fnout
                    for key,value in dict(columns).items():
                        self.dataarrays[idx]

                    for key,value in dict(columns).items():
                        if key not in [
                            'variable',
                            'absolute_path',
                            'absolute_path_for_reading',
                            'absolute_path_as_cache',
                            'path',
                            'available'
                            #'dataarray_pointer',
                        ]:
                            if type(value) == bool:
                                self.dataarrays[idx].attrs[key] = int(value)
                            else:
                                self.dataarrays[idx].attrs[key] = value
                        if key == 'variable':
                            self.dataarrays[idx].name = value

                    os.system('rm '+fnout)
                    self.dataarrays[idx].to_netcdf(fnout);logging.info('file written to: '+fnout)
                    self.remove_by_index(idx,update_pickle=False)
                    self.add_dataarray_old(fnout)
                    #self.dataarrays[idx]

                    # key = 'path'
                    # if key not in self.lib_dataarrays.columns:
                    #     self.lib_dataarrays[key] = ''
                    # self.lib_dataarrays.loc[idx]['path'] = './'

                    # note that path and absolute_path are not written to the netcdf file above, but it is available virtually for convenience
                    #self.lib_dataarrays['absolute_path'].loc[idx] = fnout
                #self.dataarrays[idx].attrs['path'] = self.lib_dataarrays.loc[idx]['path']
            else:

                logging.info("Assuming variable for "+str(idx)+" exists in file "+str(columns['absolute_path']))
                if 'path' not in self.lib_dataarrays.columns:
                    self.lib_dataarrays['path'] = None

            if 'path_pickle' in self.__dict__.keys():
                # if (('absolute_path' in columns) and (columns['absolute_path'] is not None) and (type(columns['absolute_path']) is str)):
                #     if 'path' not in self.lib_dataarrays.columns:
                #         self.lib_dataarrays['path'] = None
                #     self.lib_dataarrays['path'].loc[idx] =  os.path.relpath(os.path.realpath(columns['absolute_path']),os.path.dirname(os.path.realpath(self.path_pickle)))
                #     logging.info("relative file path to "+os.path.dirname(self.path_pickle)+" is "+self.lib_dataarrays['path'].loc[idx])
                # #os.path.commonprefix([columns['absolute_path'],lib_dirname])
                if ((columns['path'] is not None) and (type(columns['path']) is str)) and ('absolute_path_for_reading' in columns):
                    if ((columns['absolute_path_for_reading'] is None) or (type(columns['absolute_path_for_reading']) is not str)):
                        if 'absolute_path_for_reading' not in self.lib_dataarrays.columns:
                            self.lib_dataarrays['absolute_path_for_reading'] = None
                        self.lib_dataarrays['absolute_path_for_reading'].loc[idx] =  os.path.dirname(self.path_pickle)+'/'+columns['path']
                    #print("absolute file path to "+os.path.dirname(self.path_pickle)+" is "+self.lib_dataarrays['path'].loc[idx])

        if ('path_pickle' in self.__dict__.keys()):
            self.lib_dataarrays.to_pickle(self.path_pickle)
            with open(self.path_pickle+'.yaml','w') as file:
                yaml.dump([[key,self.__dict__[key]] for key in self.settings_keys],file)

    def load(
            self,
            path,
            path_settings = None,
            #file_pattern = lambda columns: columns['variable']+'_'+columns['source']+'_'+columns['time']+'_'+columns['space']+'.nc',
            file_pattern = None,
            query= None,
            extra_attributes={},
            skip_unavailable=False,
            add_file_pattern_matches=False,
            release_dataarray_pointer =False,
            cache_to_tempdir=False,
            cache_to_ram=False,
            reset_space=False,
            initialize_if_missing=True,
            **kwargs):


        if type(path).__name__ == 'list':
            # eg., -> files_wildcard = '*_*_*_*.nc'
            logging.info('Guessing files from file list...(this procedure may need revision)')
            #allkwargs = {**dict(zip(lib_dataarrays_temp.index.names, index)),**dict(dataarray),**kwargs}
            filenames = path
            for filename in filenames:

                self.add_dataarray_old(xr.open_dataarray(filename),absolute_path=filename,skip_unavailable=False,release_dataarray_pointer =True,cache_to_tempdir=False,reset_space=reset_space,**extra_attributes)

        # elif os.path.isfile(path):
        #     print('pkl file '+path+' detected. Listing files from there.' )
        #     lib_dataarrays_temp = pd.read_pickle(path)
        #     if query is not None:
        #         print('performing query subselection: '+query)
        #         lib_dataarrays_temp = lib_dataarrays_temp.query(query,engine='python')
        #     for index,dataarray in lib_dataarrays_temp.iterrows():
        #         allkwargs = {**dict(zip(lib_dataarrays_temp.index.names, index)),**dict(dataarray),**kwargs}
        #         if dataarray.path[0] == '/':
        #             absolute_file_path = dataarray.path
        #         else:
        #             absolute_file_path = os.path.dirname(path)+'/'+dataarray.path
        #         allkwargs['absolute_path'] = absolute_file_path
        #         if os.path.isfile(absolute_file_path):
        #             print('adding '+absolute_file_path+' with additional attributes: ',allkwargs)
        #             try:
        #                 xrin = xr.open_dataarray(absolute_file_path)
        #             except ValueError:
        #                 # try to open the variable out of dataset
        #                 variable = index[0]
        #                 xrin = xr.open_dataset(absolute_file_path)[variable]
        #             self.add_dataarray(xrin,release_dataarray_pointer =False,**allkwargs)

        #         else:
        #             print('Warning. Could not add file '+absolute_file_path+' because it does not exist.')

        # elif os.path.isdir(path):
        #     print('Guessing files from directory...')
        #     self.lib_dirname = path
        #     self.lib_basename = 'master.pkl'

        elif type(path) == str:
            if path[-4:] == '.pkl':
                lib_basename = os.path.basename(os.path.abspath(path))
                lib_dirname = os.path.dirname(os.path.abspath(path))
            elif path[-1:] == '/':
                lib_basename = 'master.pkl'
                lib_dirname = os.path.abspath(path)
            else:
                raise ValueError('Please provide a path that either ends on ".pkl" (indicating specific picklefile) or "/" (full archive dump at master.pkl)')
        else:
            raise IOError ('path '+path+ ' does not exist.')
#         os.system('mkdir -p '+path)

        # file_pattern = '"variable"_"columns"_"time"_"space".nc'


        temp_path_pickle = lib_dirname+'/'+lib_basename


        logging.info('apply settings according to yaml file and kwargs')
        if path_settings is None:
            path_settings = temp_path_pickle+'.yaml'
        elif not os.path.isfile(path_settings):
            raise IOError('Settings file '+path_settings+ ' not found.')
        logging.info(temp_path_pickle)
        if os.path.isfile(temp_path_pickle):
            self.path_pickle = temp_path_pickle

        if (not os.path.isfile(temp_path_pickle)) and initialize_if_missing:
            if 'file_pattern' in kwargs.keys():
                self.update(temp_path_pickle,file_pattern=kwargs['file_pattern'])
            else:
                self.update(temp_path_pickle)


        if os.path.isfile(path_settings):
            logging.info('settings file found')
            with open(path_settings) as file:
                for key,value in yaml.safe_load(file):
                    if key in self.settings_keys:
                        self.__dict__[key] = value

        for key,value in kwargs.items():
            if key not in self.settings_keys:
                raise ValueError('"'+key+'" is not a known setting.')
            else:
                self.__dict__[key] = value


        # if file_pattern is not None:
        #     print('Overriding settings "file_pattern" with ', file_pattern,' from arguments')
        #     self.file_pattern = file_pattern


        logging.info('reading the dataarrays from the pickle file')

        if type(query) == str:
            read_lib_dataarrays = pd.read_pickle(temp_path_pickle).query(query,engine='python')
        elif query is not None:
            read_lib_dataarrays = query(pd.read_pickle(temp_path_pickle))
        else:
            #query is None
            read_lib_dataarrays = pd.read_pickle(temp_path_pickle)

        for column in read_lib_dataarrays.columns:
            if column not in self.lib_dataarrays.columns:
                logging.info('adding column '+column+' from the original set to avoid errors when the query result is empty and gets queried again')
                self.lib_dataarrays[column] = ""

        for index,columns in read_lib_dataarrays.iterrows():
            absolute_path = None

            if 'path' in read_lib_dataarrays.columns:
                absolute_path = lib_dirname+'/'+columns['path']
            elif 'absolute_path' in read_lib_dataarrays.columns:
                absolute_path = columns['absolute_path']
                columns['path'] =  os.path.relpath(columns['absolute_path'],os.path.dirname(self.path_pickle))

            if (absolute_path is not None) and (absolute_path not in self.lib_dataarrays.absolute_path):
                #if index[0] == 'mslhf_0001':
                logging.info('Opening file : '+absolute_path)
                self.add_dataarray_old(absolute_path,skip_unavailable=skip_unavailable,release_dataarray_pointer =True,cache_to_tempdir=False,cache_to_ram=cache_to_ram,**({**dict(zip(read_lib_dataarrays.index.names,index)),**columns}),**extra_attributes)

        if add_file_pattern_matches and (self.file_pattern is not None):
            files_wildcard = lib_dirname+'/'+''.join(np.array(list(zip(self.file_pattern.split('"')[::2],['*']*len(self.file_pattern.split('"')[1::2])+['']))).ravel())
            logging.info('file_pattern is '+self.file_pattern+' and add_file_pattern_matches == True, so scanning and adding files that match the wildcard: ',files_wildcard+' that are not in the library yet')
            # eg., -> files_wildcard = '*_*_*_*.nc'
            filenames = glob.glob(files_wildcard)
            for filename in filenames:
                if filename not in self.lib_dataarrays.absolute_path:
                    path = os.path.relpath(filename,os.path.dirname(temp_path_pickle))
                    logging.info('Opening file : '+filename)
                    self.add_dataarray_old(filename,skip_unavailable=skip_unavailable, release_dataarray_pointer = True, cache_to_tempdir=False,path=path,cache_to_ram=cache_to_ram,reset_space=reset_space,**extra_attributes)

        # import pdb; pdb.set_trace()
        if type(query) == str:
            read_lib_dataarrays = self.lib_dataarrays.query(query,engine='python').copy()
        elif (query is not None):
            read_lib_dataarrays = query(self.lib_dataarrays)
        else:
            read_lib_dataarrays = self.lib_dataarrays.copy()

        for idx,columns in self.lib_dataarrays.iterrows():
            self.remove_by_index(idx,update_pickle=False)
        for idx,columns in read_lib_dataarrays.iterrows():
            absolute_path = None
            # import pdb; pdb.set_trace()
            if 'path' in read_lib_dataarrays.columns:
                absolute_path = lib_dirname+'/'+columns['path']
            elif 'absolute_path' in read_lib_dataarrays.columns:
                absolute_path = columns['absolute_path']

            if (absolute_path is not None) and (absolute_path not in self.lib_dataarrays.absolute_path):
                #if index[0] == 'mslhf_0001':
                logging.info('Opening file : '+absolute_path)
                self.add_dataarray_old(absolute_path,skip_unavailable=skip_unavailable,release_dataarray_pointer =release_dataarray_pointer,cache_to_tempdir=cache_to_tempdir,cache_to_ram=cache_to_ram,reset_space=reset_space,**({**dict(zip(read_lib_dataarrays.index.names,idx)),**columns}),**extra_attributes)




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
            archive_out.add_dataarray_old(dataarray_out)
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


