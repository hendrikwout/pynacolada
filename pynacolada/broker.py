import logging
from . import archive,collection
import os


class broker (object):
    def __init__(self,**kwargs):
        for key,value in kwargs.items():
            self.__dict__[key] = value
        self.global_settings()#default settings

    def propagate_arguments_to_parents(self):
        logging.info('BEGIN --- propagate arguments to parent processes --- ')
        for ibroker_requires, broker_requires in enumerate(self.requires):
            for key in list(broker_requires.keys()):
                if (broker_requires[key] is None) or (type(broker_requires[key]) is int):
                    iselect= (0 if (broker_requires[key] is None) else broker_requires[key])
                    if key in self.provides[iselect].keys():
                        self.requires[ibroker_requires][key] = self.provides[iselect][key]
                    # elif key in vars(args).keys():
                    #     self.requires[ibroker_requires][key] = vars(args)[key]
                    else:
                        del self.requires[ibroker_requires][key]
                elif type(broker_requires[key]) is type(lambda x:x):
                    pass_parameters = []
                    for ibroker_provides, broker_provides in enumerate(self.provides):
                        pass_parameters.append(broker_provides[key])
                    self.requires[ibroker_requires][key] = self.requires[ibroker_requires][key](pass_parameters)
        logging.info('END --- propagate arguments to parent processes --- ')

    def global_settings(self,**kwargs):
        self.global_settings_defaults = {
            'conda_environment_export': 'blabla',
            'dummy': 'False',
            'reset_archive': 0,
            'reset_lock':0,
            'force_recalculate':0,
        }

        for setting in self.global_settings_defaults.keys():
            if setting in kwargs.keys():
                self.__dict__[setting] = kwargs[setting]

    def retrieve_input(self):
        logging.info('--- BEGIN Collecting or generating coarse input data -- ')
        for ibroker_requires, broker_requires in enumerate(self.requires):

            request_parent = self.requires[ibroker_requires].copy()
            for key, item in list(request_parent.items()):
                if type(item) is type(lambda x: x):
                    del request_parent[key]
                if key in ['archive', 'process','root']:
                    del request_parent[key]

            if 'process' in broker_requires:
                #print('broker_requires', broker_requires)
                #print('request_parent', request_parent)
                # parent_execute = ['conda','init','bash','&&','conda','activate','KLIMPALA','&&', 'python',broker_requires['process']]
                # parent_execute = ['source','/projects/C3S_EUBiodiversity/bio_load.sh',';','python',broker_requires['process']]
                # parent_execute = ['python',broker_requires['process']]
                parent_execute = [
                    './bin/launch_python_conda.sh',
                    self.conda_environment_export,
                    broker_requires['process']
                ]
                # parent_execute = ['source', '/projects/C3S_EUBiodiversity/bio_load.sh','&&', 'python',broker_requires['process']]
                parent_arguments = []

                for key, value in request_parent.items():
                    # if key != 'spaces':
                    parent_arguments.append('--' + key)
                    # if type(value) not in [int,bool]:
                    #     parent_execute.append('"'+str(value)+'"')
                    # else:
                    parent_arguments.append(str(value))

                logging.info('propagate environment settings')
                arguments_propagate_to_parent = ['conda_environment_export', 'dummy']
                for argument in arguments_propagate_to_parent:
                    parent_arguments.append('--' + key)
                    parant_arguments.append(self.__dict__[key] if (self.__dict__[key] != '') else '""')

                arguments_propagate_reduce_to_parent = [
                    'reset_archive',
                    'reset_lock',
                    'force_recalculate'
                ]
                for key in arguments_propagate_reduce_to_parent:
                    parent_execute.append('--' + key)
                    parent_execute.append(max(self.__dict__[key]-1, 0))

                # p = Popen(parent_arguments , stdout=PIPE, stderr=PIPE)

                for iarg, arg in enumerate(parent_execute):
                    if arg == "":
                        parent_execute[iarg] = '""'
                # logging.info('Executing parent process:'+ (' '.join(parent_execute))+'"'+ '" "'.join(parent_arguments))
                loggin.info('parent_execute: ', parent_execute)
                logging.info('parent_arguments: ', parent_arguments)
                tempbasename = tempfile.mktemp(prefix=broker_requires['process'].split('/')[-1] + '_')
                self.requires[ibroker_requires]['stderr'] = open(tempbasename + '_e.log', 'w')
                self.requires[ibroker_requires]['stdout'] = open(tempbasename + '_o.log', 'w')

                logging.info('- stdout: ' + self.requires[ibroker_requires]['stdout'].name)
                logging.info('- stderr: ' + self.requires[ibroker_requires]['stderr'].name)
                print(parent_execute + parent_arguments)
                self.requires[ibroker_requires]['executing_subprocess'] = \
                    Popen(
                        parent_execute + parent_arguments,
                        # ' '.join(parent_execute), #' "' + ('" "'.join(parent_arguments))+'" ',
                        # (' '.join(parent_execute) + '"' + '" "'.join(parent_arguments)),
                        stdout=self.requires[ibroker_requires]['stdout'],
                        stderr=self.requires[ibroker_requires]['stderr'],
                        )
        for ibroker_requires,broker_requires in enumerate(self.requires):
            if 'executing_subprocess' in broker_requires:
                broker_requires['executing_subprocess'].wait()
                self.requires[ibroker_requires]['stderr'].close()
                self.requires[ibroker_requires]['stdout'].close()
                return_from_subprocess = open(self.requires[ibroker_requires]['stdout'].name,'r').readlines()[-1][:-1]
                print('retrieving from parent process '+self.requires[ibroker_requires]['stdout'].name+': '+return_from_subprocess)

                self.requires[ibroker_requires].update(literal_eval(return_from_subprocess))

        for ibroker_provides, broker_provides in enumerate(self.provides):
            if ('archive' in broker_provides) and (type(broker_provides['archive']) is type(lambda x:x)):
                values_input = [ item['archive'] for item in self.requires]
                self.provides[ibroker_provides]['archive'] = broker_provides['archive'](values_input)

        if self.reset_archive > 0 :
            logging.info('Resetting archive of current broker')
            for broker_current in self.provides:
                if 'archive' in broker_current.keys():
                    if type(broker_current['archive']) is str:
                        archive_out = archive(broker_current['root'] + '/' + broker_current['archive'])
                        archive_out.remove(records=True)
                        archive_out.close()
                    elif type(broker_current['archive']) is archive:
                        broker_current['archive'].remove(records=True)
                        broker_current['archive'].close()

        self.parent_collection = collection(
            [archive(broker_requires['root'] + '/' + broker_requires['archive']) for broker_requires in
             self.requires]
        )
        logging.info('--- END Collecting or generating coarse input data --- ')

    def apply_func(self,apply_groups_out_extra=None,*args,**kwargs):
        #for ibroker_provides, broker_provides in enumerate(self.provides):
        archive_out_filename =self.provides[0]['root'] + '/' + self.provides[0]['archive']
        lockfile = archive_out_filename + '_lock'
        if self.reset_lock > 0:
            os.system('rm ' + lockfile)
        while os.path.isfile(lockfile):
            logging.info('archive locked with "' + lockfile + '" . Waiting for another process to release it.')
            sleep(10)


        requests_parents = [broker_requires.copy() for broker_requires in self.requires]

        for irequest_parent,request_parent in enumerate(requests_parents):
            for key,value in list(request_parent.items()):
                if \
                        (key in ['archive','process','executing_subprocess','stderr','stdout']) or \
                            (key not in self.parent_collection.get_lib_dataarrays().columns and \
                        key not in self.parent_collection.get_lib_dataarrays().index.names) or \
                        type(value) is type(lambda x: x):
                    del requests_parents[irequest_parent][key]

        apply_groups_out_list = list()
        for igroups_out in range(len(self.provides)):
            apply_groups_out_list.append(dict())
            for key in self.provides[igroups_out].keys():
                if key not in ['archive','root','chain']:
                    apply_groups_out_list[igroups_out][key] = self.provides[igroups_out][key]
        import pdb; pdb.set_trace()
        self.parent_collection.apply_func(
            self.operator,
            apply_groups_in = requests_parents,
            apply_groups_out=apply_groups_out_list,
            archive_out = archive_out_filename,
            *args,
            **self.operator_properties,
            **kwargs)
        os.system('rm '+lockfile)

        logging.info('Dumping return_request as last line in stdout being used for processes depending on it')
        request_return = dict()
        for apply_groups_out in apply_groups_out_list:
            for key in apply_groups_out.keys():
                if key not in request_return.keys():
                    request_return[key] = []
                request_return[key].append(apply_groups_out[key])

        import pdb; pdb.set_trace()
        request_return['archive'] = self.provides[0]['archive']
        for key, values in list(request_return.items()):
            for ivalue,value in enumerate(values):
                if type(value) is type(lambda x:x):
                    del request_return[key][ivalue]
        import pdb; pdb.set_trace()
        return str(request_return).replace(' ','')
