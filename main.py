from pathlib import Path
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd
import yaml

import sequence as seq
import model

class Train():
    def __init__(self, args):
        self.args = args

        # copy dict is used to copy all input files during run time 
        self.copy_dict = {}
        for ff in [__file__, seq.__file__, model.__file__, model.metrics.__file__]:
            path = Path(ff)
            self.copy_dict[path.name] = path.read_text()


    def prepare_output(self, args):
        ''' generate a subfolder run_# in args['folder'] for the output
        and copy all files there '''
    
        folder = Path(args['folder'])
    
        if not folder.exists():
            folder.mkdir(parents=True)
            count = 0
        else:
            all_folder = [sf.name for sf in folder.iterdir() if
                            sf.is_dir() and args['sub'] in sf.name]
            all_counts = [int(sf[len(args['sub']) + 1: ]) for sf in all_folder]
            count = max(all_counts) + 1
    
        print('run # {}'.format(count))
    
        subfolder = folder / (args['sub'] + '_' + str(count))
        subfolder.mkdir()
    
        for key, value in self.copy_dict.items():
            path = subfolder / key
            with open(path, 'w') as ff:
                ff.write(value)
            path.chmod(0o444)
        
        return subfolder

    def get_sequences(self, args):
        ''' generate a training and validation sequence based on the the files
        specified in the args file '''
    
        ## train sequence
        train_seq  = seq.CustomSequence({**args['general'], **args['train']})
        if not Path(args['train']['bundles_path']).exists():
            train_seq.generate_bundles()
            Path(args['train']['bundles_path']).chmod(0o444)
        train_seq.prepare()
        
        ## valid sequence
        valid_seq  = seq.CustomSequence({**args['general'], **args['valid']})
        if not Path(args['valid']['bundles_path']).exists():
            valid_seq.generate_bundles()
            Path(args['valid']['bundles_path']).chmod(0o444)
        valid_seq.prepare()
    
        return train_seq, valid_seq
    
    def replace_str(self, args, pattern, replace):
        ''' replace all 'patterns' in the args file with 'replace'
        in order to specify the output folder at run time '''
        for key, value in args.items():
            if isinstance(value, dict):
                self.replace_str(value, pattern, replace)
            elif not isinstance(value, str):
                continue
            else:
                args[key] = value.replace(pattern, str(replace))
        return args
    
    def get_scheduler(self, epoch, args):
        ''' returns a scheduler based on the arguments given in the args file '''
        if args['type'] == 'step':
            lr = args['learning_rate'] * args['drop']**((epoch +1) // args['epochs_drop'])
            if 'min' in args.keys():
                lr = np.max([args['min'], lr])
        elif args['type'] == 'const':
            lr = args['learning_rate']
        elif args['type'] == 'exp':
            lr = args['learning_rate'] * np.exp(-args['factor']*epoch)
        else:
            raise ValueError('scheduler not listed')
    
        tf.summary.scalar('learning rate', data=lr, step=epoch)
        return lr
    
    def get_callbacks(self, args):
        ''' generates all callbacks specified in the args file '''
        output = []
        for cb in args.keys():
            if cb[:4] == 'ckpt':
                callback = keras.callbacks.ModelCheckpoint
            elif cb == 'tensor_board':
                callback = keras.callbacks.TensorBoard
            elif cb == 'scheduler':
                schedule = lambda epoch: self.get_scheduler(epoch, args['scheduler'])
                output.append(keras.callbacks.LearningRateScheduler(schedule))
                continue
            else:
                raise ValueError('callback not listed')
            names = np.array(callback.__init__.__code__.co_varnames[1:-1])
            values = list(callback.__init__.__defaults__)
            variables = dict(zip(names[:-len(values)], values))
            
            for ag in args[cb].keys():            
                if ag in names:
                    variables[ag] = args[cb][ag]
                else:
                    raise ValueError(f'variable {ag} not in callback')
            output.append(callback(**variables))
        
        return output
        
    
    def train_model(self):
        ''' train the model '''
    
        out_folder = self.prepare_output(self.args['output'])
        print(out_folder)
        self.args = self.replace_str(args, '<OUTPUTDIR>', out_folder)
    
        file_writer = tf.summary.create_file_writer(self.args['training']['callbacks']['tensor_board']['log_dir'])
        file_writer.set_as_default()
        
        callbacks = self.get_callbacks(self.args['training']['callbacks'])
        train_seq, valid_seq = self.get_sequences(self.args['data']) 
    
        mdl = model.Model(self.args['model']).get_model() 
        
        ## train
        history = mdl.fit_generator(
            train_seq,
            epochs=self.args['training']['epochs'],
            steps_per_epoch=train_seq.__len__(),
            validation_data=valid_seq,
            validation_steps=valid_seq.__len__(),
            callbacks=callbacks,
            workers=self.args['training']['workers'],
        )


if __name__ == '__main__':
    pdir = Path('./parameter')
    pfiles = sorted([ff for ff in pdir.iterdir() if ff.suffix == '.yml'])
    
    for pf in pfiles:
        with open(pf, 'r') as ff:
            par = ff.read()
            args = yaml.safe_load(par)
    
        training = Train(args)
        training.copy_dict['par.yml'] = par
    
        training.train_model()
